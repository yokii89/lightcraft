# LightCraft 自动关键字 · 本地模型服务

「自动关键字」扩展页用这个目录里的 `server.py` 在本机起一个视觉模型服务（默认模型
[`tencent/WeVisOne-VL-2B`](https://huggingface.co/tencent/WeVisOne-VL-2B)，Apache-2.0）。
应用替你管理它的生命周期：

- **懒启动**：在扩展页点「启动模型」才拉起进程、加载模型；第一次运行会把模型权重
  （约 4.9 GB）下载到 Hugging Face 用户缓存（`%USERPROFILE%\.cache\huggingface`，
  macOS/Linux 在 `~/.cache/huggingface`）。
- **快释放**：服务 **5 分钟**没有收到新调用就自动退出，不占机器性能；也可以随时点
  「停止模型」。即使应用退出后进程成了孤儿，空闲看门狗同样会让它退出。

## 一次性准备

```bash
pip install -r extension/server/requirements.txt
```

- Python 3.10 及以上；依赖缺失时页面会显示准确的安装命令，不会自动改动你的 Python 环境。
- 有 NVIDIA 显卡：先按 [pytorch.org](https://pytorch.org) 的命令装 CUDA 版 `torch`，
  推理快得多；CPU 也能跑，单张约几十秒，建议把页面里的单张超时调高（如 300 s）。
- 模型脚本只使用 Python 标准库的 HTTP 服务；权重下载走 Hugging Face 官方源。

## 本地权重目录（可选）

模型默认按需下载到 Hugging Face 用户缓存（`%USERPROFILE%\.cache\huggingface`）。想把它
固定放在 `server/models/`（跨境网络慢、或想离线使用）时，用本目录的 `download_model.py`：

```bash
python extension/server/download_model.py
# -> extension/server/models/tencent/WeVisOne-VL-2B/
```

- 只用 Python 标准库：多连接分段下载（单连接被限速时并行是唯一解）、断点续传
  （`Ctrl+C` 安全）、LFS sha256 校验。
- 默认先试 huggingface.co，不通自动改用 `hf-mirror.com`；`--endpoint`/`--repo`/`--out`
  可覆盖，`--url <URL> --out <文件>` 可直下任意单个大文件（如 torch 的 wheel）。
- 用法二选一：
  1. 把页面「自动关键字 → 模型仓库」填成本地目录路径，例如
     `D:\Code\Project\lightcraft\extension\server\models\tencent\WeVisOne-VL-2B`；
  2. 或保持默认仓库 id 不动——`server.py` 会先找脚本旁边的 `models/<仓库 id>`，
     命中就离线加载，不联网。
- `models/` 整个目录被 gitignore，权重永远不会进仓库。

## 隐私与安全

- 服务只绑定 `127.0.0.1` 随机端口，每次启动用一个随机 token，所有请求必须携带；
  应用侧也只允许回环地址，图片不出本机。
- 没有遥测、没有云端调用。

## 手动运行（调试用）

应用内点「启动模型」即可，一般不需要手动跑。要单独调试：

```bash
python extension/server/server.py --token 123456
# stdout 只输出协议行：LISTENING <端口> / READY / LOAD_ERROR <原因> / IDLE_SHUTDOWN
# 下载进度与请求日志在 stderr

# 测试模式：不加载模型，返回固定回复（用于集成测试）
python extension/server/server.py --token 123456 --stub --idle-timeout 2
```

参数：`--model`（Hugging Face 仓库 id 或本地权重目录）、`--token`（必填）、
`--idle-timeout`（空闲秒数，默认 300）、`--device auto|cuda|cpu`、`--stub`。
