#!/usr/bin/env python3
"""Prompt-behaviour lab for the Auto Keywords model: which phrasing makes
Qwen3-VL-2B-Instruct answer with the JSON the Rust side parses, with the most
useful Simplified Chinese keywords.

One model load, several prompts × photos, replies and decode rates printed.
Development tool, run by hand — not part of the service protocol.

    python prompt_lab.py --photos ../target/bench-photos
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# The Rust side's prompt (prompt.rs), verbatim.
CURRENT = (
    "你是照片图库的自动打标助手。看这张照片，只输出一个 JSON 对象，不要解释：\n"
    '{"category":"风景","keywords":["雪山","湖泊","日落"]}\n'
    "\n"
    "要求：\n"
    "1. category 只能是以下之一：人物、动物、风景、建筑、食物、物体、文档、其他。\n"
    "2. keywords 为 2 到 8 个简体中文词，具体、可检索；不要句子、不要相机参数、不要猜测真实姓名或具体地点。\n"
    "3. 不要重复 category 或 keywords 中已有的词。"
)

# Variants to try, in order.
VARIANTS = [
    ("describe", "描述这张照片，两句话。"),
    ("current", CURRENT),
    (
        "no-markup",
        "你是照片图库的自动打标助手。看这张照片，只输出一个 JSON 对象，不要输出任何其他文字、不要使用 <ref>、<box>、<OBJ> 等任何标记：\n"
        '{"category":"风景","keywords":["雪山","湖泊","日落"]}\n'
        "\n"
        "要求：\n"
        "1. category 只能是以下之一：人物、动物、风景、建筑、食物、物体、文档、其他。\n"
        "2. keywords 为 2 到 8 个简体中文词，具体、可检索；不要句子。\n"
        "3. 除 JSON 外不输出任何字符。",
    ),
    (
        "subject-first",
        "先看照片的主体和场景，再按下面的格式输出。只输出一个 JSON 对象，不要解释、不要 Markdown 代码块：\n"
        '{"category":"风景","keywords":["雪山","湖泊","日落","松树","倒影"]}\n'
        "\n"
        "要求：\n"
        "1. category 只能是：人物、动物、风景、建筑、食物、物体、文档、其他。\n"
        "2. keywords 为 4 到 8 个简体中文词：先主体，再环境、光线、季节；只写照片里看得到的东西，不要猜测地点和姓名。\n"
        "3. 每个词 2 到 6 个字，不要句子，不要英文，不要重复 category。",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto Keywords prompt lab")
    parser.add_argument("--photos", default="../../target/bench-photos")
    parser.add_argument("--num", type=int, default=3, help="photos to try each prompt on")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()

    import torch  # noqa: PLC0415

    from bench_model import load, load_photo  # noqa: PLC0415

    model, processor, device = load(args.model, "auto", "sdpa")
    # One tiny generate first: CUDA compiles kernels and grows the allocator on first use,
    # which would otherwise be charged to the first timed prompt.
    from PIL import Image  # noqa: PLC0415

    warm = Image.new("RGB", (64, 64), (128, 128, 128))
    warm_messages = [{"role": "user", "content": [{"type": "image", "image": warm}, {"type": "text", "text": "Warm up. Answer with one word."}]}]
    inputs = processor.apply_chat_template(warm_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    print(f"warmup done; peak VRAM {torch.cuda.max_memory_allocated() / (1 << 30):.2f} GiB", flush=True)

    photos = sorted(Path(args.photos).glob("*.jpg"))[: args.num]
    if not photos:
        raise SystemExit(f"no photos in {args.photos}")
    for name, prompt in VARIANTS:
        print(f"\n===== {name} =====", flush=True)
        for path in photos:
            image = load_photo(path, args.size)
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
            messages = [{"role": "user", "content": content}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            torch.cuda.synchronize()
            gen_s = time.perf_counter() - started
            new_tokens = int(out.shape[-1] - inputs["input_ids"].shape[-1])
            text = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            rate = new_tokens / max(gen_s, 1e-9)
            print(f"\n[{path.name}] {new_tokens} tok in {gen_s:.2f} s ({rate:.1f} tok/s)\n  {text[:300]}", flush=True)


if __name__ == "__main__":
    main()
