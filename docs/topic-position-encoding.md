# 拓展专题：位置编码（候选放在编码那一章，或课程尾部的拓展环节）

2026-10-05 讲 EP02 讲义时追问出来的一串问题，都是重要技术点，单独存下来，后续展开。
标「已核」的有出处；标「待核」的是凭记忆或推断，写进正式讲义前要查原文。

## 1. 经典 ViT 的位置表是怎么来的

- ViT-B/16：训练输入 224×224，块边长 16 → 14×14 = 196 块，加 [CLS] 共 197 格，表形状 197×768，可学习。（已核：ViT 原论文 arXiv 2010.11929）
- 块按从左到右、从上到下排成一维序号；原论文试过二维编号，没有明显差别，所以用一维。（待核：原文消融的具体表述）
- 换分辨率微调时把表二维插值一次（例：384 像素 → 24×24），之后固定。

## 2. SigLIP-2 和 Qwen3-VL 的位置表

- SigLIP-2 官方（块边长 16）：384 版 24×24，512 版 32×32，NaFlex 版 16×16 并按图插值。（已核：HF config + transformers siglip2 源码，num_patches 默认 256）
- Qwen3-VL：48×48 = 2304 格（config `num_position_embeddings: 2304`），没有 [CLS] 格。（已核）
- 报告原话：ViT 从 SigLIP-2 官方权重初始化，"continue training it with dynamic input resolutions"；"employ 2D-RoPE and interpolate absolute position embeddings based on input size, following the methodology of CoMP"。（已核：arXiv 2511.21631 §2）
- 为什么是 48：报告没写。推断：48×16 = 768 像素，比 SigLIP-2 官方最大的 512 更大；表大一些，中等尺寸图拉伸更少。（推断）

## 3. 「弹性地图」：插值不是外推

- 表铺满整张图：左上角对 (0,0)，右下角对 (47,47)，中间按比例落点，取四格加权平均（align_corners=True）。
- 例：1920×1088 → 68×120 块；第 31 行 → 31×47/67 = 21.75，第 12 列 → 12×47/119 = 4.74，四格权重 0.07/0.19/0.19/0.55。
- 48×48 不限制输入大小（那由 smart_resize 的像素上限管），只决定位置分得多细：列数多于 48 时，相邻几列拿到的位置向量差不多。
- 它表达的是「我在这张图的哪个比例位置」，不是「第几个像素」；大图小图的左上角是同一个向量。永远落在表内，所以没有外推问题。

## 4. 为什么位置表 + 2D RoPE 一起用

| | 插值位置表 | 2D RoPE |
|---|---|---|
| 告诉模型 | 我在图的哪个比例位置 | 我们隔几行几列 |
| 精度 | 48 档，大图偏粗 | 真实行列号，多大都精确 |
| 作用位置 | 输入处加一次 | 每层注意力转 Q、K |
| 来历 | 继承自 SigLIP-2 | 新加 |

- 只留位置表：大图位置太粗。只留 RoPE：可行（Qwen2-VL 的 ViT 就只用 2D RoPE，待核原文），但从 SigLIP-2 接着训，删表等于换掉它认位置的方式，容易冲坏预训练。合起来是「保旧加新」。（推断，报告未解释）
- 官方有没有做「去掉位置表、只留 RoPE」的消融：Qwen3-VL 报告 §5.12 只有 ViT vs SigLIP-2（打包比较）、DeepStack、大海捞针三组，没有这一项。（已核）
- Qwen2-VL 报告原话 "removing the original absolute position embeddings"，视觉端只用 2D RoPE。（subagent 查证，arXiv 2409.12191）
- **CoMP（arXiv 2503.18931）Table 5(a) 有直接对照**（SigLIP-So400M + Qwen2-0.5B，LLaVA-NeXT-SFT 1 epoch；AI2D / ChartQA / DocVQA）：
  只用学习表 384：48.8 / 22.8 / 24.3；只用学习表 768：47.2 / 28.8 / 29.9；**只用 RoPE-2D 768：47.5 / 8.24 / 11.9**；**表 + RoPE（CPE）768：48.0 / 32.2 / 33.2**；
  扩到 8M 数据、原生分辨率（换 Qwen2.5-0.5B）：只用 RoPE 57.8 / 29.5 / 34.3，CPE 61.9 / 66.7 / 75.9。
  论文结论：只用 RoPE "neither data-efficient nor training-friendly"——从现成 SigLIP 接着训时，删表会伤图表、文档这类精细任务。（subagent 查证，写进讲义前再对一遍原表）
- RoPE-ViT（arXiv 2403.13298）Table A.9：ImageNet 上「只 RoPE」和「RoPE + 绝对表」互有胜负——低分辨率加表更好，高分辨率外推只用 RoPE 更好。
- 主流 VLM 三条路线并存（2026-10-05 subagent 调研）：① 切图 + 固定表：InternVL、LLaVA-OV、MiniCPM-V、DeepSeek-VL2、Phi-4-MM、Llama 3.2；② 只用 2D RoPE：Qwen2/2.5-VL、Pixtral；③ 插值表 + 2D RoPE：Qwen3-VL（及 Qwen3.5/3.8 同款编码器）、Kimi-VL（64×64 bicubic）、GLM-4.1V/4.5V（24×24）。Llama 4 是表 + RoPE + 切图；Gemma 4 用 x、y 分解查表 + RoPE。

## 5. 语言模型为什么只用相对位置

1. 生成时不知道总长度，「我在全文百分之几」算不出；硬算的话每多一个字前面所有比例位置都变，KV cache 全部作废。图片编码前就知道总块数，所以能算。
2. 语言里「隔多远」比「在第几个」更重要；图片里「在画面上方」本身有意义（天空在上、框坐标）。
3. 因果注意力 + 起始 token，模型能自己推断离开头多远；不加位置编码的因果 Transformer（NoPE）也能学会位置。（待核：Haviv et al. 2022 等）
4. 早期可学习位置表（GPT-2 1024、BERT 512）表多长就只能处理多长，不能外推，所以被淘汰。
5. 语言模型也用插值，但插的是 RoPE 的位置号：Position Interpolation（Chen et al. 2023）、YaRN，把更长序列压回训练见过的范围；Qwen3-VL 把上下文从 256K 扩到 1M 用的就是 YaRN。（YaRN 已核：报告 §5.12.3；PI 待核）

一句话：图片的比例位置算得出、也有用；语言生成时算不出，算出来也不太有用，所以只留相对位置，需要拉长时对 RoPE 插值。
