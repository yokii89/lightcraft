# Optional text search model

No model weights or tokenizer assets are bundled in this repository. The user can download
the following original, unmodified model to the application configuration directory.

| Resource | Creator | Source | Licence | Added | Modifications |
|---|---|---|---|---|---|
| `bge-small-zh-v1.5/{model.safetensors,tokenizer.json}` (downloaded at runtime) | BAAI / FlagEmbedding authors | https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/7999e1d3359715c523056ef9478215996d62a620 | MIT; text in `licenses/BGE-MIT.txt`, from https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE | 2026-11-13 | none; the native encoder implements BERT inference using the pinned configuration |

The application copies the licence beside downloaded weights. Both model files have pinned
SHA-256 hashes and sizes in `src/semantic_search/model.rs`. Development model files and
reference vectors stay under the gitignored `target/` directory.
