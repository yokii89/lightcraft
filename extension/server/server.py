#!/usr/bin/env python3
"""LightCraft Auto Keywords local model service.

One photo's prompt and JPEG go in, the model's text comes back out. The service binds
127.0.0.1 only, requires the shared token on every request, and shuts itself down after
`--idle-timeout` seconds without a call, so a started-but-forgotten server never keeps the
machine busy.

stdout carries only the protocol lines LightCraft parses:

    LISTENING <port>    the HTTP server is up (the port was picked by the OS)
    READY <device>      the model is loaded and /v1/chat/completions can be served
    LOAD_ERROR <msg>    the model could not be loaded (one line; newlines replaced)
    IDLE_SHUTDOWN       the idle watchdog is ending the process

Everything else (request logs, download progress, tracebacks) goes to stderr.

The HTTP layer uses only the standard library; torch / transformers / pillow are needed
except in `--stub` mode, which serves canned replies so LightCraft's integration tests can
exercise the whole start → health → analyze → idle-shutdown loop without the ~4.3 GB model.
The model runs on a GPU only: with no CUDA/MPS device the service fails its load instead of
falling back to the CPU, which would turn every photo into a minute-long wait. One model per
machine: an OS file lock refuses a second service while the first is loading or serving.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The default repository on the Hugging Face hub (Apache-2.0). A 2B Instruct VLM fits a
# consumer 8 GB card in bf16 and follows the JSON keyword prompt as written.
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# The most new tokens one reply may take (the keyword prompt needs far fewer). Capping it also
# caps the KV cache, so a hostile `max_tokens` cannot grow the GPU footprint.
MAX_NEW_TOKENS_DEFAULT = 256
MAX_NEW_TOKENS_LIMIT = 512
# The largest request body accepted: a 1024 px JPEG at quality 85 is well under this.
MAX_BODY_BYTES = 20 << 20  # 20 MiB
# The most pixels the vision tower is handed, whatever the photo's size: a bound on the
# image-token count, so one large photo cannot grow the KV cache past a small GPU's VRAM.
# Qwen-VL stores this area bound in the image processor's size["longest_edge"] (max_pixels),
# whose checkpoint default is 16 MP. ~1.0 MP ≈ 1024 image tokens after the 2×2 merge of 16 px
# patches (32 px/token); the send path already caps the long edge at 1024 px, so this only
# reins in the processor's own upscaling.
MAX_IMAGE_PIXELS = 1_048_576  # 1024 × 1024
# Free VRAM the load needs before it starts (weights ~4.0 GiB + CUDA context + image
# activations). Measured peak on a 8 GB card after warmup is ~5.3 GiB; refusing early gives
# an actionable message instead of a CUDA out-of-memory traceback.
MIN_FREE_VRAM = 5 << 30  # 5 GiB
# The environment variable naming a directory that holds `<repo id>/` weight folders, so a
# repository checkout's `extension/server/models` can be reused instead of the HF cache.
MODELS_DIR_ENV = "LIGHTCRAFT_AUTO_KEYWORDS_MODELS"
# Linux gets expandable CUDA segments: a 1 MP photo makes several short-lived vision
# batches, and the default allocator can otherwise fragment an 8 GB card across requests.
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class NoGpuError(RuntimeError):
    """No usable GPU: the model must not be loaded on the CPU."""


def emit(line: str) -> None:
    """One protocol line on stdout. A closed pipe (LightCraft died) must not kill the
    service early: the idle watchdog is what ends it."""
    try:
        print(line, flush=True)
    except OSError:
        pass


def log(message: str) -> None:
    """A diagnostic line on stderr; timestamps keep a long model download readable."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def use_linear_patch_embed(model, torch) -> bool:
    """Route Qwen3-VL's patch embed through its equivalent linear projection.

    The embed is a Conv3d whose kernel equals its stride and covers the whole patch
    (3×2×16×16), so every patch is one dot product. On CUDA, cuDNN 9.10 picks an
    engine that takes ~45 s for one 1024 px photo (22 W, ~40 % utilisation), while
    the same contraction as a linear layer takes ~2 ms. Weights are unchanged; only
    the op changes. Returns whether the fast path was installed.
    """
    visual = getattr(getattr(model, "model", None), "visual", None)
    embed = getattr(visual, "patch_embed", None)
    proj = getattr(embed, "proj", None)
    if not isinstance(proj, torch.nn.Conv3d):
        return False
    if tuple(proj.kernel_size) != tuple(proj.stride):
        return False
    if any(proj.padding) or any(d != 1 for d in proj.dilation) or proj.groups != 1:
        return False
    original = embed.forward

    def forward(hidden_states):
        weight = proj.weight
        per_patch = weight[0].numel()
        if hidden_states.shape[-1] != per_patch:
            return original(hidden_states)
        flat = hidden_states.reshape(-1, per_patch).to(dtype=weight.dtype)
        return torch.nn.functional.linear(flat, weight.reshape(weight.shape[0], per_patch), proj.bias)

    embed.forward = forward
    return True


class Service:
    """The shared state the request handlers and the watchdog read and write."""

    def __init__(self, model_id: str, token: str, idle_timeout: float, device: str, stub: bool) -> None:
        self.model_id = model_id
        self.token = token
        self.idle_timeout = idle_timeout
        self.device_choice = device
        self.stub = stub
        self.status = "loading"  # loading | ready | error
        self.error = ""
        self.device_used = ""
        self.started = time.time()
        # The watchdog compares this against the idle timeout; it starts now, so a server
        # that is started but never called also frees the machine after the timeout.
        self.last_activity = time.time()
        self.in_flight = 0
        self.lock = threading.Lock()
        # The model is not thread-safe for concurrent generate(); LightCraft sends one
        # request at a time, but a second client must queue, not corrupt.
        self.generate_lock = threading.Lock()
        self.processor = None
        self.model = None
        # The exclusive lock held for the whole process life (real model only): a second
        # service must not load a second copy of the weights into the same GPU.
        self.instance_lock = None
        self.torch = None

    def touch(self) -> None:
        with self.lock:
            self.last_activity = time.time()

    def enter_request(self) -> None:
        with self.lock:
            self.last_activity = time.time()
            self.in_flight += 1

    def leave_request(self) -> None:
        with self.lock:
            self.last_activity = time.time()
            self.in_flight -= 1

    def authorized(self, header: str | None) -> bool:
        expected = f"Bearer {self.token}"
        return header is not None and header == expected

    # ------------------------------------------------------------- model loading

    def resolve_device(self, torch) -> str:
        """`auto` picks a GPU only. An explicit `--device cpu` stays available for debugging,
        but the app never asks for it: a 2B VLM on the CPU takes minutes per photo, so a
        missing GPU is a load failure with an actionable message, not a silent fallback."""
        if self.device_choice == "cpu":
            return "cpu"
        if self.device_choice == "cuda":
            if not torch.cuda.is_available():
                raise NoGpuError(
                    "CUDA is not available to this Python. Install the CUDA build of torch "
                    "(https://pytorch.org) and an NVIDIA driver; the model is not loaded on the CPU."
                )
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        raise NoGpuError(
            "no GPU is available to this Python (CUDA or MPS). Install the CUDA build of torch "
            "(https://pytorch.org) and an NVIDIA driver; the model is not loaded on the CPU."
        )

    def resolve_model_source(self) -> str:
        """Where the weights actually come from: `--model` is either a Hugging Face repo id
        or a directory that already holds the weights — an absolute path from the page's
        settings, `models/<repo id>` next to this script, or `<models dir>/<repo id>` under
        `LIGHTCRAFT_AUTO_KEYWORDS_MODELS`. A local copy loads offline and never touches the hub."""
        candidate = Path(self.model_id).expanduser()
        if candidate.is_dir():
            return str(candidate)
        roots = [Path(__file__).resolve().parent / "models"]
        from_env = os.environ.get(MODELS_DIR_ENV, "").strip()
        if from_env:
            roots.append(Path(from_env).expanduser())
        for root in roots:
            local = root / self.model_id
            if (local / "config.json").is_file():
                return str(local)
        return self.model_id

    def require_free_vram(self, torch) -> None:
        """Fail before allocating when the card cannot hold the weights plus activations: the
        page then says how much is free and what to close, instead of showing CUDA's OOM."""
        free, total = torch.cuda.mem_get_info()
        log(f"GPU free VRAM {free / (1 << 30):.1f} GiB of {total / (1 << 30):.1f} GiB")
        if free < MIN_FREE_VRAM:
            raise NoGpuError(
                f"the GPU has {free / (1 << 30):.1f} GiB free of {total / (1 << 30):.1f} GiB; "
                f"the model needs about {MIN_FREE_VRAM / (1 << 30):.0f} GiB. Close other GPU apps and retry."
            )

    def log_vram(self, torch, label: str) -> None:
        """One stderr line per load/generation, so VRAM growth is visible while tuning."""
        if not torch.cuda.is_available():
            return
        log(
            f"VRAM {label}: allocated {torch.cuda.memory_allocated() / (1 << 30):.2f} GiB, "
            f"reserved {torch.cuda.memory_reserved() / (1 << 30):.2f} GiB, "
            f"peak {torch.cuda.max_memory_allocated() / (1 << 30):.2f} GiB"
        )

    def bound_image_pixels(self, processor) -> None:
        """Cap the vision tower's input area at [`MAX_IMAGE_PIXELS`] so one photo cannot grow
        the image-token count (and the KV cache) past a small GPU. Qwen-VL processors keep the
        area bound in `size["longest_edge"]` (max_pixels) or a top-level `max_pixels`; whichever
        the checkpoint uses is capped, a processor with neither is left alone."""
        image_processor = getattr(processor, "image_processor", None)
        size = getattr(image_processor, "size", None)
        capped = False
        # transformers 5's SizeDict is a mapping-like object, not a dict subclass; item
        # access is the stable way to read and write it across versions.
        for key in ("longest_edge", "max_pixels"):
            try:
                value = size[key]
            except (TypeError, KeyError, IndexError):
                continue
            if isinstance(value, int):
                size[key] = min(value, MAX_IMAGE_PIXELS)
                capped = True
        if isinstance(getattr(image_processor, "max_pixels", None), int):
            image_processor.max_pixels = min(image_processor.max_pixels, MAX_IMAGE_PIXELS)
            capped = True
        if not capped:
            log("the processor has no image area cap; the send path's 1024 px long edge still bounds the input")

    def load(self) -> None:
        """Load processor and model in a background thread; the HTTP server is already
        listening, so /health reports `loading` while weights download on first run."""
        try:
            # One model per machine: hold an OS lock until this process exits. Stub services
            # load nothing, so they skip it and the Rust integration tests can run in parallel.
            self.instance_lock = acquire_instance_lock()
            import torch  # noqa: PLC0415 (imported here so --stub needs nothing)
            from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415

            self.Image = Image
            self.torch = torch
            device = self.resolve_device(torch)
            if device == "cuda":
                # TF32 matmuls run on the tensor cores (Ampere and later); the weights are
                # bf16 anyway, but the vision tower keeps fp32 work of its own.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.require_free_vram(torch)
            source = self.resolve_model_source()
            # bf16 (the checkpoint's dtype) on the GPU; float32 runs faster on most CPUs,
            # which lack native bf16 arithmetic.
            dtype = "float32" if device == "cpu" else "auto"
            log(f"loading {source} on {device} ({dtype}) …")
            started = time.time()
            processor = AutoProcessor.from_pretrained(source)
            self.bound_image_pixels(processor)
            # On the GPU, place every shard straight on the device (device_map): no full-size
            # CPU copy first, so peak host RAM stays at one shard and a small card is never
            # asked to hold the model twice. CPU/MPS materialize then move as before.
            load_kwargs: dict = {"low_cpu_mem_usage": True, "attn_implementation": "sdpa"}
            if device == "cuda":
                load_kwargs["device_map"] = {"": 0}
            try:
                model = AutoModelForMultimodalLM.from_pretrained(source, dtype=dtype, **load_kwargs)
            except TypeError:  # older transformers spell the argument torch_dtype
                model = AutoModelForMultimodalLM.from_pretrained(source, torch_dtype=dtype, **load_kwargs)
            if device != "cuda":
                model = model.to(device)
            model.eval()
            if device == "cuda" and use_linear_patch_embed(model, torch):
                log("vision patch embed: Conv3d -> equivalent linear (the cuDNN Conv3d engine takes ~45 s/photo)")
            with self.lock:
                self.processor = processor
                self.model = model
                self.device_used = device
            log(f"weights ready in {time.time() - started:.1f} s, warming up …")
            self.log_vram(torch, "after load")
            self.warmup()
            with self.lock:
                self.status = "ready"
            emit(f"READY {device}")
            log(f"model ready in {time.time() - started:.1f} s")
            self.log_vram(torch, "after warmup")
        except BaseException as exc:  # a failed import or download must surface, not hang /health
            message = " ".join(str(exc).split()) or exc.__class__.__name__
            with self.lock:
                self.status = "error"
                self.error = message
            emit(f"LOAD_ERROR {message[:500]}")
            log(f"model load failed: {exc}")
            # 快释放：加载失败的服务不占着端口和进程等看门狗，立即退出
            os._exit(1)

    def warmup(self) -> None:
        """One tiny generate() before READY: CUDA compiles kernels and grows the caching
        allocator on first use, which would otherwise land on the first photo and make the
        page's remaining-time estimate start with a lie. A warmup failure is logged, not
        fatal — the first request simply retries the same work."""
        if self.device_used == "cpu":
            return
        try:
            started = time.time()
            image = self.Image.new("RGB", (64, 64), (128, 128, 128))
            self.answer("Warm up. Answer with one word.", [image], 16)
            log(f"warmed up in {time.time() - started:.1f} s")
        except Exception as exc:  # generation failures here must not undo a successful load
            log(f"warmup skipped ({exc}); the first request may be slower")

    def answer(self, prompt: str, images: list, max_new_tokens: int) -> str:
        """One generate() call, serialized by `generate_lock`; images are PIL images."""
        if self.stub:
            time.sleep(0.05)  # a reply takes *some* time, for the cancel/timeout paths
            return '{"category":"风景","keywords":["雪山","湖泊"]}'
        import torch  # noqa: PLC0415

        content: list = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with self.generate_lock, torch.inference_mode():
            started = time.time()
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = int(outputs.shape[-1] - inputs["input_ids"].shape[-1])
        image_tokens = int((inputs["input_ids"] == self.model.config.image_token_id).sum()) if hasattr(self.model.config, "image_token_id") else 0
        log(f"generated {new_tokens} tokens in {time.time() - started:.1f} s (prompt {inputs['input_ids'].shape[-1]} tok, image {image_tokens} tok)")
        self.log_vram(torch, "after generate")
        text = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return text.strip()


SERVICE: Service | None = None  # set once in main() before the server starts


class Handler(BaseHTTPRequestHandler):
    """One request at a time per connection; bodies are read by Content-Length only."""

    protocol_version = "HTTP/1.1"
    server_version = "LightCraftAutoKeywords/1"
    # Small JSON replies must not sit in a Nagle-delayed buffer waiting for an ACK.
    disable_nagle_algorithm = True

    # ------------------------------------------------------------- plumbing

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:  # the client hung up while we were thinking
            pass

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json(status, {"error": {"message": message, "type": "auto_keywords"}})

    def read_body(self) -> bytes | None:
        length = self.headers.get("Content-Length")
        try:
            n = int(length) if length else 0
        except ValueError:
            self.send_error_json(400, "the Content-Length header is not a number")
            return None
        if n < 0 or n > MAX_BODY_BYTES:
            self.send_error_json(413, "the request body is too large")
            return None
        return self.rfile.read(n) if n > 0 else b""

    # ------------------------------------------------------------- routes

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler's name)
        assert SERVICE is not None  # set before serve_forever
        if not SERVICE.authorized(self.headers.get("Authorization")):
            self.send_error_json(401, "missing or wrong bearer token")
            return
        if self.path.split("?")[0] == "/health":
            with SERVICE.lock:
                payload = {
                    "status": SERVICE.status,
                    "model": SERVICE.model_id,
                    "device": SERVICE.device_used,
                    "uptime": round(time.time() - SERVICE.started, 1),
                    "error": SERVICE.error,
                }
            self.send_json(200, payload)
            return
        self.send_error_json(404, "unknown path")

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler's name)
        assert SERVICE is not None
        SERVICE.enter_request()
        try:
            if not SERVICE.authorized(self.headers.get("Authorization")):
                self.send_error_json(401, "missing or wrong bearer token")
                return
            if self.path.split("?")[0] != "/v1/chat/completions":
                self.send_error_json(404, "unknown path")
                return
            with SERVICE.lock:
                status, error = SERVICE.status, SERVICE.error
            if status == "loading":
                self.send_error_json(503, "the model is still loading")
                return
            if status == "error":
                self.send_error_json(503, f"the model failed to load: {error}")
                return
            body = self.read_body()
            if body is None:
                return
            try:
                request = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.send_error_json(400, "the request body is not JSON")
                return
            prompt, images, max_new_tokens = extract_request(request)
            if prompt is None:
                self.send_error_json(400, "the request has no text prompt")
                return
            try:
                content = SERVICE.answer(prompt, images, max_new_tokens)
            except Exception as exc:  # generation failed: one photo's failure, not the server's
                log(f"generation failed: {exc}")
                self.send_error_json(500, f"generation failed: {' '.join(str(exc).split())[:300]}")
                return
            self.send_json(
                200,
                {
                    "id": f"chatcmpl-{int(time.time() * 1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": SERVICE.model_id,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                },
            )
        finally:
            SERVICE.leave_request()


def extract_request(request: dict) -> tuple[str | None, list, int]:
    """Pull the prompt text, the images and the token budget out of an OpenAI-shaped chat
    request. Images arrive as `data:` URLs (the only form LightCraft sends) or PIL-ready
    fields; anything unreadable is skipped, one text part must survive."""
    max_new_tokens = request.get("max_tokens", MAX_NEW_TOKENS_DEFAULT)
    try:
        max_new_tokens = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS_LIMIT))
    except (TypeError, ValueError):
        max_new_tokens = MAX_NEW_TOKENS_DEFAULT
    prompt_parts: list[str] = []
    images: list = []
    messages = request.get("messages")
    if not isinstance(messages, list):
        return None, images, max_new_tokens
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in (None, "user"):
            continue
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}] if isinstance(content, str) else []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                prompt_parts.append(part["text"])
            url = None
            if part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                url = part["image_url"].get("url")
            elif part.get("type") == "image":
                url = part.get("url") or part.get("image")
            image = decode_image(url)
            if image is not None:
                images.append(image)
    prompt = "\n".join(prompt_parts).strip()
    return (prompt or None), images, max_new_tokens


def decode_image(source) -> object | None:
    """A `data:image/…;base64,` URL (or raw base64) as a PIL image, or None."""
    if not isinstance(source, str) or not source:
        return None
    if source.startswith("data:"):
        _, _, source = source.partition(",")
    try:
        raw = base64.b64decode(source, validate=False)
        from PIL import Image  # noqa: PLC0415

        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def acquire_instance_lock():
    """One model per machine: an exclusive OS lock on a temp file, held until this process
    exits. The lock is released by the OS even after a crash, so a stale file never blocks a
    restart; a second *live* service is refused instead of loading a second copy of the weights
    into the same GPU. Per-user on POSIX, where /tmp is shared."""
    name = "lightcraft-auto-keywords-model"
    if hasattr(os, "getuid"):
        name = f"{name}-{os.getuid()}"
    path = Path(tempfile.gettempdir()) / f"{name}.lock"
    handle = path.open("a+b")
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt  # noqa: PLC0415

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl  # noqa: PLC0415

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            "another LightCraft model service is already loading or serving a model on this "
            "computer. Stop it (or wait for its idle shutdown) and retry; only one model may "
            "be loaded at a time."
        ) from exc
    log(f"holding the single-model lock {path}")
    return handle


def watchdog(service: Service) -> None:
    """End the process when no request is in flight and the idle timeout has passed. An
    in-flight request is never interrupted: `touch` runs on arrival and on completion."""
    while True:
        time.sleep(1.0)
        with service.lock:
            idle = time.time() - service.last_activity
            busy = service.in_flight > 0
        if not busy and idle >= service.idle_timeout:
            emit("IDLE_SHUTDOWN")
            log(f"idle for {idle:.0f} s (limit {service.idle_timeout:.0f} s), shutting down")
            os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="LightCraft Auto Keywords local model service")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id, or a local weights directory")
    parser.add_argument("--token", required=True, help="shared secret every request must carry")
    parser.add_argument("--idle-timeout", type=float, default=300.0, help="seconds without a call before self-shutdown")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="where to run the model")
    parser.add_argument("--stub", action="store_true", help="no model: canned replies, for tests")
    args = parser.parse_args()
    if args.idle_timeout <= 0:
        parser.error("--idle-timeout must be positive")

    global SERVICE
    SERVICE = Service(args.model, args.token, args.idle_timeout, args.device, args.stub)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    emit(f"LISTENING {server.server_address[1]}")

    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    if args.stub:
        SERVICE.status = "ready"
        SERVICE.device_used = "stub"
        emit("READY stub")
    else:
        threading.Thread(target=SERVICE.load, name="model-load", daemon=True).start()

    threading.Thread(target=watchdog, args=(SERVICE,), name="idle-watchdog", daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        os._exit(0)


if __name__ == "__main__":
    main()
