# AGENTS.md — vlm-introduction

本仓库是公开仓库，发布「多模态大模型原理」系列的讲义、口播稿和事实核对笔记。下面的规则只针对本仓库。

## 目录结构

| 位置 | 放什么 |
| --- | --- |
| `lecture/epNN/` | 一集的讲义：`index.html`（翻页外壳）、`slides/`（每页一个 HTML）、`thumbs/`（总览缩略图）、`assets/`（图片） |
| `transcripts/epNN.md`、`transcripts/epNN.json` | 口播稿；JSON 每句只保留 `id`、`scene`、`text`（字幕原文） |
| `docs/sources/` | 事实核对笔记：每个数字的出处（论文章节页码、源码行号、config 字段） |
| `docs/` 其他文件 | 拓展专题等补充材料 |
| `README.md` | 唯一入口：系列简介 + 每集一行的表格（视频、讲义、口播稿链接）+ 来源与制作说明 |

## 加新一集

1. 讲义放 `lecture/epNN/`，四个部分齐全；所有引用用相对路径（例如 `../assets/cat.png`），不得引用仓库外文件。
2. 口播稿放 `transcripts/epNN.md` 和 `.json`，只保留字幕原文，不放配音用的改写文本。
3. 成片作为 GitHub Release 附件上传（命名 `vlm-introduction-epNN.mp4`），不提交进 Git。
4. README 表格加一行；未定稿的集要在 README 注明是初版及已知问题。
5. 提交后用 GitHub Pages 地址打开新讲义，确认图片和字体正常加载。

## 不能提交的内容

- 第三方论文原文：PDF、全文抽字文本、整段转贴。笔记里只写结论、页码和必要的短引文。
- 视频文件（`*.mp4`）和任何 PDF（`.gitignore` 已排除）。
- 制作流水线的中间文件、配音与对口型缓存、声音克隆配置、录像素材。
- 任何个人电脑上的绝对路径、密钥、私人联系方式。提交前全文搜一遍用户目录路径前缀和常见密钥字段名，结果必须为空。

## 内容口径

- 数字以 Qwen3-VL-8B 为准，来源是技术报告 arXiv 2511.21631、transformers `qwen3_vl` 源码和 HF config。
- 报告和源码都没写、靠推理得出的内容，在讲义里标「推断」或「报告未写」。
- 公式用纯文本写法，不依赖 LaTeX 渲染。
- 许可为 CC BY 4.0。
