"""Create a local, self-contained comparison from quantization_bench.py JSON files."""
import argparse
import base64
import html
import json
from pathlib import Path

MODES = ["bf16", "int8", "nf4"]


def escape(value):
    return html.escape(str(value))


def picture(path):
    if not path.is_file():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:{mime};base64,{data}" alt="{escape(path.name)}">'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("target/quantization-final"))
    parser.add_argument("--timing", type=Path, default=Path("target/quantization-throughput"))
    parser.add_argument("--photos", type=Path, default=Path("target/quantization-suite/photos"))
    parser.add_argument("--output", type=Path, default=Path("target/quantization-comparison.html"))
    args = parser.parse_args()
    reports = {mode: json.loads((args.results / (mode + ".json")).read_text(encoding="utf-8")) for mode in MODES}
    timing = {mode: json.loads((args.timing / (mode + ".json")).read_text(encoding="utf-8")) if (args.timing / (mode + ".json")).exists() else reports[mode] for mode in MODES}
    content = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Qwen quantization comparison</title>',
               '<style>body{font:15px system-ui,sans-serif;max-width:1600px;margin:32px auto;padding:0 20px;color:#17202b;background:#f6f8fb}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}td,th{border:1px solid #d9e0e8;padding:14px;vertical-align:top;text-align:left}th{background:#e9eef6}img{max-width:250px;max-height:220px;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px system-ui,sans-serif;margin:10px 0}.pass{color:#087f5b}.fail{color:#c92a2a}.small{font-size:13px;color:#526170}details{margin:16px 0}h1{font-size:28px}td:first-child{width:20%}</style>',
               '<h1>千问量化本机对照</h1><p>Qwen3-VL-2B-Instruct · RTX 4060 Ti 8GB · CUDA · 同一提示词 · 贪心解码 · 照片长边 1024 px。</p>',
               '<p>INT8 / NF4 量化语言解码器的线性层；视觉编码器、词嵌入和绑定输出层保留 BF16。显存为 PyTorch 分配峰值，不含其他程序、CUDA 上下文和全部缓存池。质量与吞吐分开运行。</p>',
               '<p>16 道人工设定答案的视觉题只覆盖有限能力，不代表通用智力。基础题中的 Markdown 包裹会先去除；格式遵守另列。照片描述保留完整原文供人工判断，没有用 BF16 输出当作正确答案。</p>',
               '<table><tr><th>指标</th><th>BF16</th><th>INT8</th><th>NF4 4-bit</th></tr>']
    rows = [
        ("峰值分配显存 / GiB", lambda mode: f'{timing[mode]["summary"]["peak_allocated_gib"]:.2f}'),
        ("照片生成 / tokens/s（含视觉 prefill）", lambda mode: f'{timing[mode]["summary"]["aggregate_photo_tokens_per_second"]:.2f}'),
        ("单张平均生成 / 秒", lambda mode: f'{timing[mode]["summary"]["mean_photo_seconds"]:.2f}'),
        ("视觉题内容正确", lambda mode: f'{reports[mode]["summary"]["quality_passed"]}/{reports[mode]["summary"]["quality_total"]}'),
        ("照片字段符合要求（去 Markdown）", lambda mode: f'{reports[mode]["summary"]["schema_after_fence_removal"]}/{reports[mode]["summary"]["description_total"]}'),
        ("照片字段符合要求（严格 JSON）", lambda mode: f'{reports[mode]["summary"]["valid_descriptions"]}/{reports[mode]["summary"]["description_total"]}'),
    ]
    for label, value in rows:
        content.append('<tr><th>' + label + '</th>' + ''.join('<td>' + escape(value(mode)) + '</td>' for mode in MODES) + '</tr>')
    content.append('</table><h2>逐题答案</h2><table><tr><th>输入与标准答案</th><th>BF16</th><th>INT8</th><th>NF4</th></tr>')
    for index, case in enumerate(reports['bf16']['quality']):
        content.append('<tr><td>' + picture(args.results / (case['id'] + '.png')) + '<pre>' + escape(case['prompt']) + '</pre><pre>标准答案：' + escape(json.dumps(case['expected'], ensure_ascii=False)) + '</pre></td>')
        for mode in MODES:
            row = reports[mode]['quality'][index]
            content.append(f'<td><b class="{"pass" if row["passed"] else "fail"}">{"正确" if row["passed"] else "错误"}</b><pre>{escape(row["text"])}</pre></td>')
        content.append('</tr>')
    content.append('</table><h2>照片描述与关键词</h2><table><tr><th>图片</th><th>BF16</th><th>INT8</th><th>NF4</th></tr>')
    for index, case in enumerate(reports['bf16']['descriptions']):
        content.append('<tr><td>' + picture(args.photos / case['photo']) + '<p>' + escape(case['photo']) + '</p></td>')
        for mode in MODES:
            row = reports[mode]['descriptions'][index]
            okay = row['schema_after_fence_removal']
            content.append(f'<td><b class="{"pass" if okay else "fail"}">{"字段符合要求" if okay else "字段不符合要求"}</b><pre>{escape(row["text"])}</pre></td>')
        content.append('</tr>')
    content.append('</table><details><summary>运行版本与量化范围</summary><pre>' + escape(json.dumps({m: {k: reports[m].get(k) for k in ['packages', 'torch', 'quantization_scope', 'quantized_module_count', 'prompt_sha256']} for m in MODES}, ensure_ascii=False, indent=2)) + '</pre></details></html>')
    args.output.write_text('\n'.join(content), encoding='utf-8')
    print(args.output.resolve())


if __name__ == "__main__":
    main()
