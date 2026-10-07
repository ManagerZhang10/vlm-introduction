# Qwen3-VL-8B-Instruct 结构事实核对

来源：HF `Qwen/Qwen3-VL-8B-Instruct` 的三个 config；transformers main@0a896aa（2026-10-02）。源码从 GitHub 上的 transformers 仓库读取。
缩写：M = models/qwen3_vl/modeling_qwen3_vl.py；VU = src/transformers/vision_utils.py（main 把 qwen3_vl 的网格计算抽到了这里）；IP = models/qwen2_vl/image_processing_qwen2_vl.py（preprocessor_config 指定的是 Qwen2VLImageProcessorFast）；VP = video_processing_qwen3_vl.py；P = processing_qwen3_vl.py；M457 = v4.57.1 版 modeling（对应 config 里的 transformers_version 4.57，用来交叉核对）。

## A. 编码

1. **patch_embed**：Conv3d(3→1152, kernel=stride=(2,16,16), bias=True)。输入先 view 成 (-1,3,2,16,16)，所以每块是 3×2×16×16=1536 个数，等价于一个 1536→1152 的线性层。— M:94-101；IP:206、VP:260（channel×temporal_patch_size×patch_size²）
2. **位置编码，两套叠加**：
   - 可学习的绝对位置表 `pos_embed = nn.Embedding(2304, 1152)`，也就是 48×48 的网格。按每张图的 patch 网格 (grid_h, grid_w) 做双线性插值，align_corners=True；同一视频的各时间组共用同一份。插值函数在 main 里是 `get_vision_interpolation_indices_and_weights`（VU:231，调用处 M:654-658、698-710），在 v4.57.1 里是 `fast_pos_embed_interpolate`（M457:642，用 linspace(0,47,h)）。
   - 2D RoPE：head_dim=72，18 个频率给 h、18 个给 w，拼成 36 维再复制成 72 维，整个头都参与旋转；θ=10000（默认值）。坐标是 patch 级的 (h,w)，不含 t。— M:106-167（docstring 108-110 写的是 "axial 2D rope"）、VU:81-127、modeling_rope_utils.py:739、M457:82
3. **图片尺寸（smart_resize）**：长和宽各四舍五入到 32（=patch 16 × merge 2）的倍数；总像素超过上限就等比缩小、向下取整，低于下限就等比放大、向上取整；长宽比超过 200 直接报错。下限 65536、上限 16777216 来自 preprocessor_config 的 `size.shortest_edge` / `size.longest_edge`，覆盖了代码默认值 56×56 和 28×28×1280。token 数 N=(H'/32)×(W'/32)，范围 64～16384。例：1920×1080 → 1920×1088 → 2040 个 token。— IP:63-90、96、238；P:76-79
   **视频的像素预算不一样**：shortest_edge 4096、longest_edge 25165824，而且算的是**所有帧加起来**（t×h×w）。由此推出单个视频的视觉 token 上限约为 25165824/(2×32×32)=12288。— video_preprocessor_config；VP:75-108、201-218
   main 分支另外新增了 `cap_pixels_per_frame`：目前默认关闭并打印警告，从 v5.22 起改为默认开启。— VP:53-60、283-292
4. **PatchMerger**：先做 LayerNorm(1152)（在拼接之前）→ 把 4 个 patch 拼成 4608 维 → Linear(4608,4608) → GELU（`nn.GELU`，精确版）→ Linear(4608,4096)。
   **DeepStack** 有 3 个独立的 merger（`deepstack_merger_list`）。结构和主 merger 一样，区别是 LayerNorm 放在拼接之后，作用在 4608 维上（`use_postshuffle_norm=True`）。ViT 第 8/16/24 个 block（从 0 数）的输出各自过对应的 merger，然后在 LLM 第 0/1/2 层**算完之后**直接加到输出 hidden states 上，而且只加在视觉 token 的位置（`visual_pos_masks = image_mask | video_mask`）。— M:170-183、663-676、718-732、841-862、1209-1233
5. **ViT block**：pre-norm，LayerNorm eps=1e-6。注意力的 qkv 是一个 Linear(1152, 3456, bias)，16 头 × 72 维。MLP 是 1152→4304→1152，激活 gelu_pytorch_tanh，带 bias，不是门控结构。— M:73-83、244-256、327-354；config vision_config

## B. 注意力

6. **ViT 的注意力范围**：没有窗口注意力，qwen3_vl 的源码和 config 里都找不到 window 或 fullatt_block_indexes。它是"分段的全局注意力"：每张图、每个时间组（两帧合成的一个 temporal patch）各算一段，段内互相全看得见，段与段之间看不见。所以视频里不同时间组在 ViT 里互相看不见，多张图之间也看不见。
   ```
   seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
   return F.pad(seqlens.cumsum(dim=0, dtype=dtype), (1, 0), value=0)
   ```
   — VU:42-65（merge_temporal 默认 False，docstring 原文是 "each frame is its own attention segment"）；M:707、281-317；M457:727-735 写法不同、意思一样
7. **LLM 侧**：32 个 Q 头、8 个 KV 头（GQA，4:1），head_dim 128，rope_theta 5000000；q 和 k 各过一层 RMSNorm（QK-Norm）。用的是因果注意力，视觉 token 之间也是因果的：源码只调用了 create_causal_mask，没有给图块开双向掩码。— config text_config；M:482-501、486、815
8. **MRoPE**：
   - position_ids 的形状是 3×batch×seq（进入 LLM 时可以再加一行文本位置，变成 4 行）。
   - 图像 token：t 全部等于起点 s，h = s + 行号，w = s + 列号（按合并后的网格算）。图后面的文字从 s + max(H',W') 开始数，也就是图内最大位置 +1。
   - interleaved 的分配：64 个频率对按下标轮流分给 T/H/W。下标 1,4,…,58 给 H（20 个），2,5,…,59 给 W（20 个），剩下的 0,3,…,57 和 60～63 给 T（24 个），排出来就是 THW THW … THW TTTT，最后复制成 128 维。v4.57.1 的 docstring 写成 "[THTHWHTHW...TT]"，是笔误。
   - 视频：先把 grid 按 t 拆成多组，每组 t=1，组与组之间插入时间戳文本。v4.57.1 的注释原文是 "Qwen3VL use timestamps rather than absolute time position ids"；main 的注释里误写成了 "Qwen3.5"。
   — M:414-423、881-931、933-1024（967-970）、800-811；M457:923-925、299-314
9. **推理时的 attention 实现**：ViT 用 flash_attention_2 时走 varlen（cu_seq_lens_q/k = cu_seqlens）；用 sdpa 或 eager 时按 cu_seqlens 切段，逐段计算。LLM 声明同时支持 flash 和 sdpa，不指定时 transformers 默认选 sdpa。— M:281-317、625-626；modeling_utils.py:1733-1734

## C. 规模与成本

10. 按 config 和源码逐项算：
    - 视觉侧 576,388,336 个参数（ViT 本体 0.416B + 主 merger 0.040B + 3 个 DeepStack merger 0.120B）
    - LLM 8,190,735,360 个参数（36 层 6.946B + 词嵌入 0.622B + 不和词嵌入共享的 lm_head 0.622B，tie_word_embeddings=false）
    - 合计 8,767,123,696。乘以 2 字节，正好等于 model.safetensors.index.json 里的 total_size 17,534,247,392，两边对得上。
11. KV cache：36 × 2 × 8 × 128 × 2 = 147,456 字节 = 144 KiB/token，核对无误。照此算，一张 2040 token 的图约占 287 MiB，32K 上下文约 4.5 GiB。

## 讲视频时值得讲、但容易讲错的点

1. 说"ViT 把整段视频一起看"是错的：ViT 只在每两帧一组的内部做注意力，跨时间的交互全靠 LLM。
2. MRoPE 的 t 不是真实时间：每组的 t 只占 1 格，组与组之间的位置靠 token 数往前推（包括时间戳文字）；真实秒数只出现在 `<x.x seconds>` 这段文本里。
3. DeepStack 不是"把多层特征拼起来"，而是 3 个独立 merger 的结果分别**加**到 LLM 第 0/1/2 层输出的视觉 token 位置上，文本位置不动。
4. 2304 不是"最大序列长度"（configuration_qwen3_vl.py:33-34 的 docstring 这么写，会误导人），而是 48×48 的位置表；边长超过 768 像素的图，是在插值放大这张表。
5. 对齐单位是 32，不是 Qwen2-VL 的 28。ViT 的序列长度是视觉 token 数的 4 倍，而且没有窗口，所以 16384 token 的大图在 ViT 里是一段 65536 长的全局注意力。
6. 单张图会被复制成两帧，所以 Conv3d 的两个时间切片实际上等于相加后再用（这是推论：两帧相同时 W0·x+W1·x=(W0+W1)·x）。另外 merger 用的是精确 GELU，ViT 的 MLP 用的是 tanh 近似的 GELU，讲的时候别混在一起。
