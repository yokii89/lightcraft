#!/usr/bin/env python3
"""Download model weights fast where a single connection is throttled (stdlib only).

The cross-border link to the Hugging Face CDN throttles one connection to well under
100 KB/s, while N parallel ranged connections each get their own slice, so the weights
arrive in minutes instead of hours. Every file is fetched in fixed-size chunks into
`<name>.part.NNNN` siblings, streamed to disk as it arrives, so an interrupted run
resumes where it stopped (`Ctrl+C` is safe, and so is a lost network). When every chunk
is in, the parts are concatenated and, when the hub published an LFS checksum, verified
against it.

Usage (from anywhere; the default target directory sits next to this script):

    python download_model.py                     # default repo, into <script>/models
    python download_model.py --workers 16        # fewer parallel connections
    python download_model.py --out D:/models     # anywhere else
    python download_model.py --url <URL> --out <file>   # one file, no hub API involved

The default output layout `models/Qwen/Qwen3-VL-2B-Instruct/…` is exactly what
`AutoModelForMultimodalLM.from_pretrained` accepts, so the Auto Keywords page's
模型仓库 field can point at it directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO = "Qwen/Qwen3-VL-2B-Instruct"
# Seconds a read may stall before a chunk request is given up and retried.
READ_TIMEOUT = 45.0
# Attempts per chunk before the whole run fails (finished chunks stay on disk).
CHUNK_RETRIES = 12
# 32 MiB: small enough for fine-grained resume, large enough to amortize the redirect hop.
CHUNK_BYTES_DEFAULT = 32 << 20
# The public mirror that keeps working where huggingface.co does not; tried in this order.
ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]

_log_lock = threading.Lock()


def log(message: str) -> None:
    with _log_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def fetch(url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0):
    """GET with redirects; the caller reads or discards the body. 4xx/5xx raise HTTPError."""
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "lightcraft-download/1", **(headers or {})})
    context = ssl.create_default_context()
    return urllib.request.urlopen(request, timeout=timeout, context=context)


def endpoint_for(repo: str, override: str | None) -> str:
    """The first hub front that answers the repo's API; the mirror is the fallback, so this
    also keeps working where huggingface.co is unreachable."""
    if override:
        return override.rstrip("/")
    for endpoint in ENDPOINTS:
        try:
            with fetch(f"{endpoint}/api/models/{repo}", timeout=10.0) as response:
                if response.status == 200:
                    log(f"hub endpoint: {endpoint}")
                    return endpoint
        except (OSError, urllib.error.URLError, TimeoutError):
            continue
    raise SystemExit(f"no hub endpoint answered (tried {', '.join(ENDPOINTS)}); check the network or pass --endpoint")


def list_files(repo: str, endpoint: str) -> list[dict]:
    """The repo's files on `main`, each with its size and, for LFS files, its sha256."""
    with fetch(f"{endpoint}/api/models/{repo}/tree/main?recursive=true", timeout=30.0) as response:
        tree = json.load(response)
    files = [entry for entry in tree if entry.get("type") == "file"]
    if not files:
        raise SystemExit(f"the hub lists no files for {repo}")
    return files


def remote_size(url: str) -> int:
    """The file's full size, from the Content-Length of an opened (unread) response."""
    with fetch(url, timeout=30.0) as response:
        return int(response.headers.get("Content-Length") or 0)


def human(bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if bytes < 1024 or unit == "GiB":
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} GiB"


class Progress:
    """One shared byte counter for all workers; a reporter thread prints it."""

    def __init__(self, total_bytes: int) -> None:
        self.total = total_bytes
        self.done = 0
        self.lock = threading.Lock()
        self.started = time.time()
        self.stop = threading.Event()

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n

    def report_loop(self, label: str) -> None:
        while not self.stop.wait(5.0):
            with self.lock:
                done = self.done
            rate = done / max(time.time() - self.started, 0.001)
            if self.total > 0 and rate > 1:
                eta = (self.total - done) / rate
                log(f"{label}: {human(done)} / {human(self.total)} ({human(rate)}/s, ~{eta / 60:.0f} min left)")
            else:
                log(f"{label}: {human(done)} / {human(self.total)}")


def chunk_plan(size: int, chunk_bytes: int) -> list[tuple[int, int]]:
    """The (start, end) byte ranges covering a file, in order."""
    return [(start, min(start + chunk_bytes, size) - 1) for start in range(0, size, chunk_bytes)]


def part_path(target: Path, index: int) -> Path:
    return target.with_name(f"{target.name}.part.{index:05d}")


def download_range(url: str, path: Path, start: int, end: int, progress: Progress) -> None:
    """One ranged GET streamed to disk in 1 MiB blocks, so progress is visible and a crash
    never loses more than a megabyte. The file must land with exactly the asked size."""
    expected = end - start + 1
    for attempt in range(1, CHUNK_RETRIES + 1):
        try:
            with fetch(url, headers={"Range": f"bytes={start}-{end}"}, timeout=READ_TIMEOUT) as response:
                # Small, non-LFS files may be served by a CDN that ignores Range and answers
                # 200 with the whole file. A single-chunk range starting at 0 is that whole
                # file, so accept it; a later chunk of a big file cannot be trusted this way.
                if response.status != 206 and not (response.status == 200 and start == 0):
                    raise OSError(f"expected 206 for a ranged request, got {response.status}")
                remaining = expected
                with path.open("wb") as out:
                    while remaining > 0 and (block := response.read(min(1 << 20, remaining))):
                        out.write(block)
                        progress.add(len(block))
                        remaining -= len(block)
            if path.stat().st_size != expected:
                raise OSError(f"short read: {path.stat().st_size} of {expected} bytes")
            return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == CHUNK_RETRIES:
                raise SystemExit(f"bytes {start}-{end} of {path.name} failed {CHUNK_RETRIES} times: {exc}") from exc
            time.sleep(min(2.0 ** attempt, 15.0))


def run_queue(items: list[dict], workers: int) -> None:
    """Fetch the queued ranges on `workers` threads; a failure stops the queue early."""
    pending = iter(items)
    lock = threading.Lock()
    failures: list[str] = []

    def take() -> dict | None:
        with lock:
            return next(pending, None)

    def run() -> None:
        while item := take():
            if failures:
                return
            try:
                download_range(item["url"], item["path"], item["start"], item["end"], item["progress"])
            except SystemExit as exc:
                with lock:
                    failures.append(str(exc).strip())

    threads = [threading.Thread(target=run, name=f"dl-{index}") for index in range(min(workers, len(items)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"{len(failures)} range(s) failed; rerun the same command to resume the rest")


def concatenate(parts: list[Path], target: Path) -> None:
    with target.open("wb") as out:
        for part in parts:
            with part.open("rb") as src:
                while block := src.read(4 << 20):
                    out.write(block)


def verify_sha256(target: Path, expected: str) -> bool:
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        while block := fh.read(4 << 20):
            digest.update(block)
    return digest.hexdigest() == expected


def assemble(parts: list[Path], target: Path, expected_sha: str | None) -> None:
    """Concatenate a file's chunks, drop them, and verify the LFS checksum when the hub
    published one. A checksum mismatch deletes the file so a rerun refetches it."""
    concatenate(parts, target)
    for part in parts:
        part.unlink()
    if expected_sha and not verify_sha256(target, expected_sha):
        target.unlink(missing_ok=True)
        raise SystemExit(f"{target.name}: sha256 does not match the hub's checksum; deleted, rerun to retry")
    log(f"{target.name}: done{', sha256 verified' if expected_sha else ''} ({human(target.stat().st_size)})")


def queue_file(url: str, target: Path, size: int, progress: Progress, chunk_bytes: int, items: list[dict]) -> list[Path]:
    """Queue every missing chunk range of one file and return its part paths in order.
    Chunks already on disk with the right size are kept, so a rerun resumes."""
    parts: list[Path] = []
    for index, (start, end) in enumerate(chunk_plan(size, chunk_bytes)):
        part = part_path(target, index)
        parts.append(part)
        if part.is_file() and part.stat().st_size == end - start + 1:
            progress.add(end - start + 1)
            continue
        items.append({"url": url, "path": part, "start": start, "end": end, "progress": progress})
    return parts


def lfs_checksum(entry: dict) -> str | None:
    """The sha256 the hub stores for an LFS file, or None (plain git files have none)."""
    lfs = entry.get("lfs") or {}
    return lfs.get("oid") if lfs.get("size") == entry.get("size") else None


def main() -> None:
    parser = argparse.ArgumentParser(description="download model weights with parallel ranged requests")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repository id")
    parser.add_argument("--endpoint", default=None, help="hub front (default: try huggingface.co, then hf-mirror.com)")
    parser.add_argument("--out", default=None, help="target directory (repo mode) or file path (--url mode)")
    parser.add_argument("--url", default=None, help="fetch this one URL directly instead of listing a repo")
    parser.add_argument("--workers", type=int, default=32, help="parallel connections")
    parser.add_argument("--chunk-mb", type=int, default=CHUNK_BYTES_DEFAULT >> 20, help="chunk size in MiB")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_mb < 1:
        parser.error("--workers and --chunk-mb must be positive")
    chunk_bytes = args.chunk_mb << 20

    items: list[dict] = []
    assemblies: list[tuple[Path, list[Path], str | None]] = []
    progress = Progress(0)
    reporter = threading.Thread(target=progress.report_loop, args=("total",), daemon=True)
    started = time.time()

    if args.url:
        if not args.out:
            parser.error("--url mode needs --out pointing at the target file")
        target = Path(args.out).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        size = remote_size(args.url)
        if size <= 0:
            raise SystemExit(f"the server does not announce a size for {args.url}")
        log(f"{target.name}: {human(size)} in one file")
        progress.total = size
        parts = queue_file(args.url, target, size, progress, chunk_bytes, items)
        assemblies.append((target, parts, None))
        reporter.start()
        run_queue(items, args.workers)
    else:
        endpoint = endpoint_for(args.repo, args.endpoint)
        out = Path(args.out).expanduser() if args.out else Path(__file__).resolve().parent / "models"
        target_dir = out / args.repo
        target_dir.mkdir(parents=True, exist_ok=True)
        files = list_files(args.repo, endpoint)
        progress.total = sum(entry.get("size") or 0 for entry in files)
        log(f"{args.repo}: {len(files)} files, {human(progress.total)} -> {target_dir}")
        reporter.start()
        for entry in files:
            name = entry["path"]
            target = target_dir / name
            if "/" in name:  # nested paths keep the repo's layout
                target.parent.mkdir(parents=True, exist_ok=True)
            size = entry.get("size") or 0
            if target.is_file() and target.stat().st_size == size:
                progress.add(size)
                log(f"{name}: already present, skipping")
                continue
            url = f"{endpoint}/{args.repo}/resolve/main/{name}"
            parts = queue_file(url, target, size, progress, chunk_bytes, items)
            assemblies.append((target, parts, lfs_checksum(entry)))
        run_queue(items, args.workers)

    for target, parts, expected_sha in assemblies:
        assemble(parts, target, expected_sha)
    progress.stop.set()
    reporter.join(timeout=6.0)
    if args.url:
        log(f"done: {human(progress.total)} in {(time.time() - started) / 60:.1f} min -> {Path(args.out).expanduser()}")
        print(f"DONE {Path(args.out).expanduser()}", flush=True)
    else:
        target_dir = Path(args.out).expanduser() / args.repo if args.out else Path(__file__).resolve().parent / "models" / args.repo
        if not (target_dir / "config.json").is_file():
            raise SystemExit("config.json is missing from the download; the repo may not be a loadable model")
        log(f"done: {human(progress.total)} in {(time.time() - started) / 60:.1f} min -> {target_dir}")
        print(f"DONE {target_dir}", flush=True)


if __name__ == "__main__":
    main()
