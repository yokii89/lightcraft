"""Development-only reference vectors from official Transformers BERT inference.

python extension/server/text_reference.py target/semantic-search/model
No network calls, no remote code, one CPU thread. The native app does not use Python.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

directory = Path(sys.argv[1])
torch.set_num_threads(1)
tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
model = AutoModel.from_pretrained(directory, local_files_only=True).eval()
cases = [
    ("一只白色小狗在草地上奔跑。", False),
    ("在户外跑动的宠物", True),
    ("Café，红色 car 2026！", False),
    ("猫", False),
    ("绿色的树木与红色的房屋。" * 15, False),
    ("夜景", True),
    ("猫" * 254, False),
]
results = []
with torch.inference_mode():
    for text, query in cases:
        actual = "为这个句子生成表示以用于检索相关文章：" + text if query else text
        inputs = tokenizer(actual, return_tensors="pt", truncation=False)
        vector = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0], p=2, dim=1)[0]
        results.append({"text": text, "query": query, "vector": vector.tolist(), "ids": inputs["input_ids"][0].tolist()})
(directory / "reference.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(results)} reference vectors")
