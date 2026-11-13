#!/usr/bin/env python3
"""LightCraft Auto Keywords model benchmark.

Loads the model the same way `server.py` does (same dtype, SDPA attention, TF32
settings) and times one photo's ask at several send sizes, with per-token
timestamps so prefill and decode can be told apart. With `--photos` it runs on
photo-like images (dumped from the demo scenes by
`cargo run -p lightcraft-extension --example dump_scenes`); without, on synthetic
gradients — not real photo content, so use them for timing only.

Not part of the product or the service protocol: a development tool, run by hand.

    python bench_model.py --photos ../target/bench-photos --sizes 1024,768,640 --num 5
    python bench_model.py --photos ../target/bench-photos --probe          # per-token probe
    python bench_model.py --photos ../target/bench-photos --batch 4        # batching experiment
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from photo_prompt import photo_prompt

PROMPT = photo_prompt()

IMAGE_TOKEN_ID = 151655  # config.json's image_token_id
# The vision-input area cap server.py applies (size["longest_edge"]); mirrored here so the
# bench measures the production image-token count, not the checkpoint's 16 MP default.
MAX_IMAGE_PIXELS = 1_048_576  # 1024 × 1024


def make_photo(long_edge: int, seed: int):
    """A synthetic 3:2 photo (timing only: not real content to read keywords off)."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    w, h = long_edge, long_edge * 2 // 3
    img = Image.new("RGB", (w, h))
    top, bottom = (rng.randint(40, 200), rng.randint(40, 200), rng.randint(40, 200), ), (rng.randint(40, 200), rng.randint(40, 200), rng.randint(40, 200))
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        row = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        for x in range(w):
            px[x, y] = row
    draw = ImageDraw.Draw(img)
    for _ in range(24):
        x, y = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(w // 40, w // 8)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=tuple(rng.randint(0, 255) for _ in range(3)))
    noise = img.tobytes()
    img.frombytes(bytes(b ^ rng.randint(0, 24) for b in noise))
    return img


def load_photo(path: Path, long_edge: int):
    """A photo file as the Rust worker would send it: long edge held to the send
    size, JPEG quality 85 round-trip (§12's send path)."""
    from io import BytesIO

    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        scale = long_edge / max(img.size)
        if scale < 1.0:
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=85)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def load(model_arg: str, device_arg: str, attn: str):
    """The exact load path of server.py (Service.load): BF16 on the GPU,
    float32 on the CPU, SDPA attention (or what --attn asks), TF32 matmuls."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    device = "cpu" if device_arg == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    candidate = Path(model_arg).expanduser()
    source = str(candidate) if candidate.is_dir() else model_arg
    local = Path(__file__).resolve().parent / "models" / model_arg
    if (local / "config.json").is_file():
        source = str(local)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    started = time.time()
    print(f"loading {source} on {device} ({dtype}, {attn}) …", flush=True)
    processor = AutoProcessor.from_pretrained(source)
    # SizeDict (transformers 5) is mapping-like but not a dict subclass.
    size = getattr(getattr(processor, "image_processor", None), "size", None)
    for key in ("longest_edge", "max_pixels"):
        try:
            value = size[key]
        except (TypeError, KeyError, IndexError):
            continue
        if isinstance(value, int):
            size[key] = min(value, MAX_IMAGE_PIXELS)
    kwargs = {"dtype": dtype, "low_cpu_mem_usage": True, "attn_implementation": attn}
    if device == "cuda":
        kwargs["device_map"] = {"": 0}  # straight onto the GPU, no full-size CPU copy
    try:
        model = AutoModelForMultimodalLM.from_pretrained(source, **kwargs)
    except (TypeError, ValueError):
        model = AutoModelForMultimodalLM.from_pretrained(source, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation=attn)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    if device == "cuda":
        from server import use_linear_patch_embed  # same fast patch embed as the service

        use_linear_patch_embed(model, torch)
    print(f"weights ready in {time.time() - started:.1f} s", flush=True)
    return model, processor, device


class Ask:
    """One ask's measurements."""

    def __init__(self) -> None:
        self.prep_s = 0.0
        self.generate_s = 0.0
        self.prompt_tokens = 0
        self.image_tokens = 0
        self.new_tokens = 0
        self.eos_pos: int | None = None
        self.text = ""
        self.arrivals: list[float] | None = None


def timed_ask(model, processor, device, image, prompt: str, max_new_tokens: int, stream: bool) -> Ask:
    """One ask. With `stream`, per-token arrival times split prefill (first token)
    from decode (the rest)."""
    import torch

    content = [] if image is None else [{"type": "image", "image": image}]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    ask = Ask()
    started = time.perf_counter()
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)
    ask.prep_s = time.perf_counter() - started
    ask.prompt_tokens = int(inputs["input_ids"].shape[-1])
    ask.image_tokens = int((inputs["input_ids"] == IMAGE_TOKEN_ID).sum())
    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    if stream:
        from threading import Thread

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)
        thread = Thread(target=model.generate, kwargs=kwargs)
        torch.cuda.synchronize() if device == "cuda" else None
        started = time.perf_counter()
        thread.start()
        arrivals: list[float] = []
        for chunk in streamer:
            arrivals.append(time.perf_counter())
            ask.text += chunk
        thread.join()
        ask.generate_s = time.perf_counter() - started
        ask.arrivals = arrivals
        ask.new_tokens = len(arrivals)
        ask.eos_pos = ask.new_tokens
    else:
        torch.cuda.synchronize() if device == "cuda" else None
        started = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize() if device == "cuda" else None
        ask.generate_s = time.perf_counter() - started
        generated = outputs[0][ask.prompt_tokens :]
        ask.new_tokens = int(generated.shape[-1])
        ask.eos_pos = next((i + 1 for i, tok in enumerate(generated.tolist()) if tok in eos_ids), None)
        ask.text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return ask


def probe_rows(asks: list[Ask]) -> None:
    """The probe: prefill (up to the first token) and the per-token decode gaps."""
    for ask in asks[:1]:
        if not ask.arrivals:
            continue
        # generate() started right after prep ended: rewind from the first arrival
        start = ask.arrivals[0] - ask.prep_s
        prefill = ask.arrivals[0] - start
        gaps = [ask.arrivals[i] - ask.arrivals[i - 1] for i in range(1, len(ask.arrivals))]
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
        tail = sum(gaps[-20:]) / max(1, len(gaps[-20:])) if gaps else 0.0
        print(f"  prompt {ask.prompt_tokens:>5} tok (image {ask.image_tokens:>4}) | prefill {prefill:>5.2f} s | {ask.new_tokens:>4} tok | mean {mean_gap * 1000:>6.1f} ms/tok | tail20 {tail * 1000:>6.1f} ms/tok")
        if ask.text:
            print(f"    reply: {ask.text[:120]}")


def batch_bench(model, processor, device, image, n: int) -> None:
    """N copies of one image in one generate() vs the same N one at a time. Left
    padding so all rows finish at the same step; the processor pads the pixel
    batch itself."""
    import torch

    conversations = [[{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}] for _ in range(n)]
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True).to(device)
    except Exception as exc:
        print(f"batch {n}: the processor does not batch ({exc})")
        return
    torch.cuda.synchronize() if device == "cuda" else None
    started = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    torch.cuda.synchronize() if device == "cuda" else None
    batched = time.perf_counter() - started
    del inputs, outputs
    torch.cuda.empty_cache()
    sequential = 0.0
    for _ in range(n):
        sequential += timed_ask(model, processor, device, image, PROMPT, 96, stream=False).generate_s
    print(f"batch {n}: batched {batched:.2f} s vs sequential {sequential:.2f} s → {sequential / batched:.2f}× throughput")


def main() -> None:
    parser = argparse.ArgumentParser(description="LightCraft Auto Keywords model benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct", help="HF repo id or local weights directory")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa | eager | flash_attention_2)")
    parser.add_argument("--photos", default="", help="directory of photo JPEGs (dump_scenes output); empty = synthetic")
    parser.add_argument("--num", type=int, default=5, help="photos per size")
    parser.add_argument("--sizes", default="1024,768,640", help="send long edges to try, comma separated")
    parser.add_argument("--reps", type=int, default=1, help="timed runs per photo")
    parser.add_argument("--max-tokens", type=int, default=512, help="the decode cap to test (the Rust side sends 512)")
    parser.add_argument("--batch", default="", help="batching experiment: batch sizes to try, comma separated (e.g. 2,4)")
    parser.add_argument("--probe", action="store_true", help="per-token timing probe on the first photo of each size")
    parser.add_argument("--text-only", action="store_true", help="also time one text-only ask per size")
    args = parser.parse_args()

    model, processor, device = load(args.model, args.device, args.attn)
    import torch

    try:
        import flash_attn  # noqa: F401

        print("flash_attn: available")
    except ImportError:
        print("flash_attn: not installed")

    photos = sorted(Path(args.photos).glob("*.jpg")) if args.photos else []
    if args.photos and not photos:
        raise SystemExit(f"no .jpg found in {args.photos}")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    for size in sizes:
        images = [load_photo(p, size) for p in photos[: args.num]] if photos else [make_photo(size, seed=size)]
        print(f"\n== long edge {size} ({len(images)} photo{'s' if len(images) != 1 else ''}) ==", flush=True)
        # one warmup per size: the first generate at a new shape pays cudnn autotune
        timed_ask(model, processor, device, images[0], PROMPT, 16, stream=False)
        asks = []
        for _ in range(args.reps):
            for image in images:
                asks.append(timed_ask(model, processor, device, image, PROMPT, args.max_tokens, stream=args.probe))
        if args.probe:
            probe_rows(asks)
        else:
            for ask in asks:
                at_eos = f"{ask.eos_pos}" if ask.eos_pos is not None else "cap"
                decode_s = max(ask.generate_s - ask.prep_s, 1e-9)
                print(f"  prep {ask.prep_s * 1000:>4.0f} ms | gen {ask.generate_s:>6.2f} s | prompt {ask.prompt_tokens:>5} tok (image {ask.image_tokens:>4}) | new {ask.new_tokens:>4} | EOS at {at_eos:>4} | {ask.new_tokens / decode_s:>5.1f} tok/s")
            mean_gen = sum(a.generate_s for a in asks) / len(asks)
            print(f"  mean gen {mean_gen:.2f} s")
        if args.text_only:
            ask = timed_ask(model, processor, device, None, PROMPT, 64, stream=False)
            print(f"  text-only: gen {ask.generate_s:.2f} s | prompt {ask.prompt_tokens} tok | new {ask.new_tokens} | EOS at {ask.eos_pos}")
    if device == "cuda":
        print(f"\npeak VRAM {torch.cuda.max_memory_allocated() / (1 << 30):.2f} GiB")

    if args.batch:
        size = sizes[0]
        image = load_photo(photos[0], size) if photos else make_photo(size, seed=size)
        timed_ask(model, processor, device, image, PROMPT, 16, stream=False)
        for n in (int(b) for b in args.batch.split(",") if b.strip()):
            batch_bench(model, processor, device, image, n)


if __name__ == "__main__":
    main()
