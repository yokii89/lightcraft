"""Read the production prompt for development benchmarks; no duplicated prompt text."""
from pathlib import Path


def photo_prompt(max_keywords=8):
    source = Path(__file__).resolve().parent.parent / "src" / "auto_keywords" / "prompt.txt"
    return (source.read_text(encoding="utf-8").strip()
            .replace("{categories}", "人物、动物、风景、建筑、食物、物体、文档、其他")
            .replace("{max_keywords}", str(max(1, min(16, max_keywords)))))
