"""Opt-in production-prompt smoke check using an original procedural demo image.

python extension/server/description_smoke.py --photo target/bench-photos/01-lake-at-first-light.jpg --out target/semantic-search/description-smoke.json
Uses the service's GPU checks and exclusive model lock. No product dependency on this tool.
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image
from photo_prompt import photo_prompt
from server import Service

parser = argparse.ArgumentParser()
parser.add_argument("--photo", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
args = parser.parse_args()
service = Service(args.model, "development-smoke", 300, "auto", False)
service.load()
if service.status != "ready":
    raise SystemExit(service.error)
with Image.open(args.photo) as source:
    image = source.convert("RGB")
image.thumbnail((512, 512))
started = time.perf_counter()
raw = service.answer(photo_prompt(), [image], 512)
result = json.loads(raw)
assert result["category"] in ["人物", "动物", "风景", "建筑", "食物", "物体", "文档", "其他"]
assert isinstance(result["description"], str) and result["description"].strip()
assert 1 <= len(result["keywords"]) <= 8
payload = {"elapsed_seconds": time.perf_counter() - started, "result": result}
Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
