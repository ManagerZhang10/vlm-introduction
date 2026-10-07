# VLM Introduction：多模态大模型原理（EP01–04）

一套讲「看图的大模型是怎么工作的」的入门系列：四集竖屏讲解视频，配三份 16:9 讲义。
全系列拿一个真实模型 **Qwen3-VL-8B** 从头走到尾——图怎么切块、怎么编码、怎么交给语言模型、整个模型怎么训出来——数字按它的公开 config、源码和技术报告核对，核不到的地方在讲义里标明。

- 视频：竖屏 1080×1920，每集 3～4 分钟，适合手机看。
- 讲义：16:9 多页 HTML，比视频讲得更细，浏览器里直接翻页（← / → 键，ESC 回总览）。

## 四集一览

| 集 | 主题 | 视频 | 讲义（在线翻页） | 口播稿 |
| --- | --- | --- | --- | --- |
| 01 | AI 怎么看懂图片和视频（总览：切块 → 投影 → 编码 → 对齐 → 拼接） | [下载 / 播放](https://github.com/ManagerZhang10/vlm-introduction/releases/download/v1/vlm-introduction-ep01.mp4)（3:24） | 无单独讲义，内容在 02–04 展开 | [ep01.md](transcripts/ep01.md) |
| 02 | 视觉编码：缩放、切块、位置编码、ViT、合并器、视频 | [下载 / 播放](https://github.com/ManagerZhang10/vlm-introduction/releases/download/v1/vlm-introduction-ep02.mp4)（4:18） | [lecture/ep02](https://managerzhang10.github.io/vlm-introduction/lecture/ep02/) | [ep02.md](transcripts/ep02.md) |
| 03 | 图怎么交给大模型：占位符替换、因果注意力、MRoPE 三个坐标、视频时间戳、KV cache | [下载 / 播放](https://github.com/ManagerZhang10/vlm-introduction/releases/download/v1/vlm-introduction-ep03.mp4)（4:11） | [lecture/ep03](https://managerzhang10.github.io/vlm-introduction/lecture/ep03/) | [ep03.md](transcripts/ep03.md) |
| 04 | 训练：87 亿个参数从哪来（训练目标、四段预训练、SFT / 蒸馏 / 强化学习） | [下载 / 播放](https://github.com/ManagerZhang10/vlm-introduction/releases/download/v1/vlm-introduction-ep04.mp4)（4:15） | [lecture/ep04](https://managerzhang10.github.io/vlm-introduction/lecture/ep04/) | [ep04.md](transcripts/ep04.md) |

视频都在 [Release v1](https://github.com/ManagerZhang10/vlm-introduction/releases/tag/v1) 的附件里，不进 Git 历史。
口播稿另有只含字幕原文的 JSON 版（`transcripts/ep0N.json`）。

## 事实来源与阅读提示

- 事实主要来自三处：Qwen3-VL 技术报告（[arXiv 2511.21631](https://arxiv.org/abs/2511.21631)）、Hugging Face transformers 的 `qwen3_vl` 源码、HF 上 `Qwen/Qwen3-VL-8B-Instruct` 的 config。EP04 另外引用了 SAPO、DeepSeekMath、Qwen3、LLaVA、GKD、MiniLLM 等论文，出处和页码见 [`docs/sources/`](docs/sources/)。
- 讲义里标了「**推断**」「**报告未写**」的地方，是报告和源码都没有直接给出的内容，请按标注理解，不要当成官方结论。
- [`docs/topic-position-encoding.md`](docs/topic-position-encoding.md) 是讲 EP02 时追问出来的位置编码拓展问题，标「待核」的条目尚未查证。
- 讲义和口播里的数字以 8B 为准；`docs/sources/` 里有一部分 4B 实测记录，已在文中注明。

## 制作说明

- 口播用的是作者本人声音的克隆；出镜画面是本人录像加对口型。
- 背景音乐由 ElevenLabs Music 生成。
- 示意图里的折耳猫照片在整个系列反复使用，随本仓库一起以 CC BY 4.0 发布。
- 制作流水线（配音、对口型、渲染脚本）不在本仓库。

## 目录

```text
vlm-introduction/
├── lecture/
│   └── ep02/ ep03/ ep04/
│       ├── index.html     ← 翻页外壳，打开它即可
│       ├── slides/        ← 每页一个 HTML
│       ├── thumbs/        ← 总览用的缩略图
│       └── assets/        ← 讲义用到的图片
├── transcripts/           ← 四集口播稿（Markdown + 仅字幕原文的 JSON）
└── docs/
    ├── topic-position-encoding.md
    └── sources/           ← 事实核对笔记（结构、训练、EP04 训练方法）
```

本地看讲义：克隆后直接双击 `lecture/ep04/index.html`；或在仓库根目录跑 `python3 -m http.server` 后打开 `http://localhost:8000/lecture/ep04/`。讲义字体从 Google Fonts 加载，离线时会退回系统字体。

## 许可

内容（讲义、口播稿、笔记、视频）按 [CC BY 4.0](LICENSE) 发布，转载请注明出处。
讲义中引用的论文和源码的版权归原作者所有。
