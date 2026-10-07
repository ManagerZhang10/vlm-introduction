# EP04 训练方法事实核对

> 核对日期：2026-10-07。全部来自 arXiv 原文 PDF（原文不随本仓库发布），用 pypdf 抽字后逐段核对。
> 页码 = PDF 文件的页序（第 N 页）。这几篇的页序和论文页脚印的页码一致或基本一致。
> 版本：SAPO 2511.20347v2（2025-12-01）；DeepSeekMath 2402.03300（下载到的最新版）；Qwen3 2505.09388v1；
> LLaVA-1.5 2310.03744v2；LLaVA 2304.08485v2；GKD 2306.13649v3；MiniLLM 2306.08543v6（2026-01-31）。
> 换版本的话页码可能会变。
> 公式全部用纯文本写。记号：pi = 当前策略，pi_old = 采样时的旧策略，sigma(x) = 1/(1+e^-x)。

---

## 1. SAPO：Soft Adaptive Policy Optimization（arXiv 2511.20347，Gao et al.，Qwen Team）

**一句话**：SAPO 去掉了 GRPO/GSPO 里“超出范围就把梯度直接清零”的硬裁剪，换成一个平滑的 sigmoid 门。
重要性比率离 1 越远，梯度就越小，但不会突然变成 0。advantage 为负的 token 用更大的温度，让它的梯度衰减得更快。

### 1.1 目标函数（p3，式 5–6）

```
J(theta) = E_{q~D, {y_i}~pi_old} [ (1/G) * sum_i (1/|y_i|) * sum_t  f_{i,t}( r_{i,t}(theta) ) * A_{i,t} ]

r_{i,t}(theta) = pi(y_{i,t} | q, y_{i,<t}) / pi_old(y_{i,t} | q, y_{i,<t})          # 逐 token 重要性比率（p2 式 2）

f_{i,t}(x) = sigma( tau_{i,t} * (x - 1) ) * 4 / tau_{i,t}                          # 软门（替代 clip）

tau_{i,t} = tau_pos   若 A_{i,t} > 0
          = tau_neg   否则（A <= 0）
```

- 和 GRPO 相比：没有 `min(r*A, clip(r,1-eps,1+eps)*A)`，换成 `f(r)*A`。
- 目标函数（式 5）里**没有 KL 惩罚项**（论文的 SAPO 目标里没写 KL 项；有没有在工程上另加 KL：未核实）。

### 1.2 梯度权重 / 门函数形式（p3 式 7–8；p5 式 16）

```
grad J = E[ (1/G) sum_i (1/|y_i|) sum_t  w_{i,t} * r_{i,t} * grad log pi(y_{i,t}|...) * A_{i,t} ]

w_{i,t} = 4 * p_{i,t} * (1 - p_{i,t}),   p_{i,t} = sigma( tau_{i,t} * (r_{i,t} - 1) )
        = sech^2( (tau_{i,t}/2) * (r_{i,t} - 1) )        # 即 f'(r)
```

- r = 1 时 w 正好等于 1，和不裁剪的目标 `r*A` 梯度相同，与 tau 无关。这就是系数要写成 4/tau 的原因（p3）。
- r 偏离 1 时 w 平滑地、近似按指数衰减，于是形成一个“连续的信任域”（p3）。
- 对照 GRPO 的门（p6 式 24）：A>0 时 r<=1+eps 取 1，否则取 0；A<=0 时 r>=1-eps 取 1，否则取 0，也就是“全有或全无”。

### 1.3 正负 advantage 用不同温度吗？—— 是（p1、p4、p7）

- 设 `tau_neg > tau_pos`，tau 越大衰减越快（p4）。
- 理由（p4 式 9）：对 logit z_v 求导，A>0 会提高采样 token 的 logit、压低其余全部 token；A<0 则反过来，会抬高词表里大量无关 token 的 logit。
  词表有几十万个 token，所以负梯度会扩散出去，更容易让训练不稳定。
- 实验取值（p7 §5.1）：`tau_pos = 1.0`，`tau_neg = 1.05`。
- 消融（p7–p8 图 5）：tau_neg=1.05 > tau_pos 最稳；两者相等时居中；tau_neg=0.95 < tau_pos 最不稳。

### 1.4 advantage 怎么算 —— 与 GRPO 相同，在组内归一化（p2 式 2；p3 写明“computed as in Equation (2)”）

```
A_{i,t} = A_i = ( R_i - mean({R_j}_{j=1..G}) ) / std({R_j}_{j=1..G})     # 同一回答内所有 token 共用
```

### 1.5 与 GRPO / GSPO 的关系（p2 式 3–4；p4 式 10–14；p5 式 16–23）

- 统一写法：`J = E[(1/G) sum_i (1/|y_i|) sum_t f_{i,t}(r_{i,t}) * A_i]`，三种算法只是 f 不同（p4）。
  - GRPO：A>0 时 `f = min(r, 1+eps)`，A<=0 时 `f = max(r, 1-eps)`。token 级硬门。
  - GSPO（arXiv 2507.18071）：把同一个 min/max 作用到序列级比率上，
    `s_i = (pi(y_i|q)/pi_old(y_i|q))^(1/|y_i|) = exp( (1/|y_i|) * sum_t log r_{i,t} )`。
    同一序列里所有 token 的门都一样。
  - SAPO：token 级软门。
- 和 GSPO 的关系：在两个假设下，(A1) r≈1（步子小、接近 on-policy），(A2) 同一序列内各 token 的 log r 方差小，
  token 门的平均值约等于序列门 `g(log s_i) = sech^2( (tau/2) * log s_i )`，此时 SAPO 退化成“带连续信任域的 GSPO”（p5 式 23）。
  误差上界：`D_i <= (tau^2 / 4) * Var_i`（p5 式 21）。
- 假设成立吗（p5–p6 图 2–3）：在 Qwen3-30B-A3B（MoE）和 Qwen3-4B（dense）的冷启动 checkpoint 上统计了超过 1e5 条序列、1e9 个 token。
  r 集中在 1 附近，Var_i 通常小于 0.02。MoE 的分布更宽，dense 更集中。
- 好处（p1、p3、p6）：如果一条序列里只有少数 token 严重 off-policy，GSPO 会把整条序列的梯度都压掉；SAPO 只压这几个 token，其余 token 的信号保留。
  和 GRPO 比，SAPO 用平滑缩放代替 0/1 门，不会出现梯度突然消失。

### 1.6 主要实验结论（p7–p9）

- 受控实验（p7 §5.1，图 4）：从 Qwen3-30B-A3B-Base 冷启动后做数学 RL。每批 rollout 切成 4 个 mini-batch 更新。
  评测 AIME25、HMMT25、BeyondAIME（16 次采样平均 Pass@1）。
  对比对象是 GSPO 和 GRPO-R2（GRPO 加 routing replay），超参沿用 GSPO 论文。
  结论：GSPO 和 GRPO-R2 在训练早期就崩了，SAPO 一直稳定，最终分数更高；而且 **SAPO 不需要 routing replay**。
- 温度消融（p7–p8 图 5）：见 1.3。
- Qwen3-VL（p8 §5.2，p9 图 6）：用于训练 Qwen3-VL 系列，覆盖不同尺寸以及 MoE 和 dense 两种架构，任务混合了数学、代码、逻辑和多模态。
  在 Qwen3-VL-30B-A3B 的初步冷启动 checkpoint 上，评测 AIME25（Pass@1，32 次采样）、LiveCodeBench v6（Pass@1，8 次采样）、ZebraLogic、MathVision 四项均分。
  算力相同时，SAPO 高于 GSPO 和 GRPO-R2。每批切 2 个 mini-batch。
- 注意：论文**没有给数值表**，结论全部来自曲线图（图 4–6）。从图上读出的具体分数：未核实，不建议引用。

---

## 2. GRPO（DeepSeekMath，arXiv 2402.03300，§4.1，p11–p14）

**一句话**：同一道题让模型答 G 遍，用这 G 个答案的平均分当基线，比平均好的回答就加强、比平均差的就削弱。
这样就不需要再训练一个和策略模型差不多大的价值模型（critic）。

### 2.1 组内 advantage（p14 §4.1.2，结果监督）

```
r = {r_1, ..., r_G}                         # 奖励模型给 G 个输出打的分
A_hat_{i,t} = r_tilde_i = ( r_i - mean(r) ) / std(r)     # 该输出的所有 token 共用同一个值
```

- 过程监督（p14 §4.1.3）：对每一步的奖励按全体步骤的 mean/std 归一化，`A_hat_{i,t} = sum_{index(j) >= t} r_tilde_i^{index(j)}`，即当前 token 之后所有步骤归一化奖励之和。

### 2.2 裁剪目标函数（p13 式 3）

```
J_GRPO(theta) = E_{q~P(Q), {o_i}_{i=1..G} ~ pi_old(O|q)}
   (1/G) * sum_i (1/|o_i|) * sum_t {
       min( rho_{i,t} * A_hat_{i,t},  clip(rho_{i,t}, 1-eps, 1+eps) * A_hat_{i,t} )
       - beta * D_KL( pi_theta || pi_ref )
   }

rho_{i,t} = pi_theta(o_{i,t} | q, o_{i,<t}) / pi_old(o_{i,t} | q, o_{i,<t})
```

对照 PPO（p11 式 1）：形式相同，但 PPO 的 A_t 来自 GAE，要用一个学出来的价值函数 V_psi。

### 2.3 KL 项（p13–p14 式 4）

- 位置：PPO 的常见做法是把逐 token 的 KL 惩罚加进奖励里，`r_t = r_phi(q, o_<=t) - beta * log(pi_theta/pi_ref)`（p13 式 2）。
  GRPO 改为**直接加在 loss 里**，这样就不会把 advantage 的计算搞复杂（p13）。
- 估计器（Schulman 2020 的无偏估计，p14 式 4），保证非负：

```
D_KL(pi_theta || pi_ref) = pi_ref(o_{i,t}|...) / pi_theta(o_{i,t}|...) - log( pi_ref(o_{i,t}|...) / pi_theta(o_{i,t}|...) ) - 1
```

- 方向：`D_KL(pi_theta || pi_ref)`，也就是“当前策略 || 参考策略”，属于 reverse 方向，并且是在当前策略的采样上估计。
- pi_ref 通常是初始的 SFT 模型（p13）；做迭代 RL 时，每轮把 pi_ref 重设为当前策略（p14 算法 1）。

### 2.4 为什么不要 critic（p13）

1. PPO 的价值函数通常和策略模型一样大，显存和算力负担都很重。
2. 在 LLM 场景里，奖励模型往往只给最后一个 token 打分，很难把每个 token 位置的价值都学准。
3. GRPO 用“同一问题多个采样的平均奖励”作为基线，省掉了价值函数近似。它在组内做相对比较，也正好贴合奖励模型“在同一问题的回答之间做比较”的训练方式。

### 2.5 训练超参（p15 §4.2）

基座 DeepSeekMath-Instruct 7B；约 144K 道 GSM8K/MATH 相关 CoT 题；策略学习率 1e-6；**KL 系数 beta = 0.04**；
**每题采样 G = 64 个输出**；最大长度 1024；训练批大小 1024；每个探索阶段之后策略只更新一次。

---

## 3. Qwen3 技术报告（arXiv 2505.09388）：Strong-to-Weak Distillation

**一句话**：小模型不重新走旗舰模型的四阶段后训练，而是两步走。先拿大模型写好的回答做 SFT（off-policy），
再让小模型自己生成，用大模型的 logits 逐 token 纠正（on-policy）。效果比直接做 RL 好，GPU 时间约为 RL 的 1/10。

### 3.1 适用对象（p12 §4.5）

5 个 dense 模型：Qwen3-0.6B、1.7B、4B、8B、14B，加 1 个 MoE 模型 Qwen3-30B-A3B。
旗舰模型（Qwen3-235B-A22B、Qwen3-32B）走四阶段流程：Long-CoT 冷启动 → 推理 RL → 思考模式融合 → 通用 RL（p10）。

### 3.2 off-policy 蒸馏（p12 §4.5 (1)）

- 把教师模型在 `/think` 和 `/no_think` 两种模式下生成的**回答**合在一起，做“response distillation”，也就是在教师生成的文本上做 SFT。
- 目的：让学生先学会基本推理能力和模式切换，为后面的 on-policy 阶段打基础。
- 这一阶段的教师具体是哪一个：报告没有单独写明，未核实。

### 3.3 on-policy 蒸馏（p12 §4.5 (2)）

- 先采样 prompt，**学生自己**在 `/think` 或 `/no_think` 模式下生成回答，再把学生的 logits 对齐到教师（**Qwen3-32B 或 Qwen3-235B-A22B**）的 logits 上，目标是最小化 KL 散度。
- 每个学生对应哪位教师：报告只写了“Qwen3-32B or Qwen3-235B-A22B”，没有给出逐个模型的对应关系，未核实。
- **KL 方向：报告没有写**，原文只有 “to minimize the KL divergence”。说它是 reverse KL 属于推断，没有原文出处，未核实。
- 思考 / 非思考模式：两个阶段都同时覆盖两种模式。off-policy 阶段混合两种模式下的教师输出；on-policy 阶段学生在两种模式下都会生成。
  报告还说这种做法能“imparting robust mode-switching capabilities”（p12），并且能“maintaining fine-grained control over their reasoning processes”（p10）。

### 3.4 成本 / 效果对比

- p10 §4：直接蒸馏教师 logits，比每个小模型单独跑四阶段流程的 Pass@1 更高，Pass@64（探索能力）也更好，**GPU 小时只需四阶段流程的 1/10**。
- p20–p21 表 21（Qwen3-8B，起点同为 off-policy 蒸馏后的 checkpoint，只用数学和代码 query）：

| 方法 | AIME'24 (pass@64) | AIME'25 (pass@64) | MATH500 | LCB v5 | MMLU-Redux | GPQA-D | GPU 小时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| off-policy 蒸馏（起点） | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | - |
| + 强化学习 | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | **17,920** |
| + on-policy 蒸馏 | 74.4 (93.3) | 65.5 (86.7) | 97.0 | 60.3 | 88.3 | 63.3 | **1,800** |

- 结论（p21）：蒸馏分数明显更高，GPU 小时约为 RL 的 1/10（1,800 / 17,920 ≈ 0.10）。蒸馏还提高了 pass@64，RL 没有提高 pass@64。

---

## 4. LLaVA-1.5（arXiv 2310.03744）与 LLaVA 原版（2304.08485）

### 4.1 LLaVA-1.5

| 项 | 内容 | 出处 |
| --- | --- | --- |
| stage 1 数据量 | **558K**（表 3 “Sample Size – Pretrain” 列） | p5 表 3 |
| stage 2 数据量 | **665K** 条指令数据混合：LLaVA 158K + ShareGPT 40K + VQAv2 83K + GQA 72K + OKVQA 9K + OCRVQA 80K + A-OKVQA 66K + TextCaps 22K + RefCOCO 48K + VG 86K | p9 表 7 |
| 投影层 | 把原来的单层线性投影换成**两层 MLP**（“a two-layer MLP”） | p3 §3.3 |
| 激活函数 | 论文正文没写（常说的 GELU 来自代码，未核实） | — |
| 视觉编码器 / 分辨率 | CLIP-ViT-L-336px，336x336 | p1、p4 |
| 基座 LLM | Vicuna v1.5（7B / 13B） | p10 A.3 |
| 超参（表 9） | stage 1：batch 256，lr 1e-3；stage 2：batch 128，lr 2e-5。两阶段都用 cosine 衰减，warmup 比例 0.03，weight decay 0，1 epoch，AdamW；DeepSpeed stage 1 用 ZeRO-2，stage 2 用 ZeRO-3 | p10 A.3 表 9 |
| 预训练学习率减半 | 因为改用了 MLP 投影，预训练学习率减半，其余超参同 LLaVA | p10 A.3 |
| 算力 | 8x A100：约 6 小时预训练 + 约 20 小时指令微调，“~1 day on a single 8-A100 node” | p1、p4 §3.3 |

- **哪些参数可训：LLaVA-1.5 论文没有逐阶段重新写**，只说预训练数据和超参“与 LLaVA 相同”（p4、p10）。按沿用原版的理解：
  stage 1 只训投影层（MLP），stage 2 训投影层和 LLM，视觉编码器始终冻结。这一点由原版论文推出，1.5 原文里没有直接核实到，引用时宜写“沿用 LLaVA”。
- 558K 是什么数据：1.5 论文正文**没有写数据名**。通常说的 “LCS-558K（LAION/CC/SBU 子集，BLIP 生成 caption）” 来自官方仓库，论文中未核实。
- 小矛盾：1.5 的 §3.3（p4）说 “we use the same pretraining dataset”，但表 3 写的是 558K，原版是 595K。两处不一致，讲义里直接用 558K 并标注出处为表 3 即可。

### 4.2 LLaVA 原版（2304.08485）对照（p5 §4.2、§5；p20 附录 C）

- stage 1（Feature Alignment）：**把 CC3M 过滤成 595K 图文对**，视觉编码器和 LLM 都冻结，**只训投影矩阵 W**（单层线性）。论文把这一步比作“为冻结的 LLM 训练一个兼容的视觉 tokenizer”。
- stage 2（端到端微调）：视觉编码器仍冻结，训 W 和 LLM 参数 phi；Chatbot 场景用 LLaVA-Instruct-158K。
- 超参：stage 1 训 1 epoch，lr 2e-3，batch 128；stage 2 训 3 epoch，lr 2e-5，batch 32；8x A100，预训练不超过 4 小时，微调约 10 小时。
- 注意：1.5 说 batch size 和原版相同，但原版论文写的预训练 batch 是 128，1.5 表 9 写的是 256。以各自论文原文为准，两处不一致。

---

## 5. SFT ↔ forward KL，on-policy 蒸馏 ↔ reverse KL

### 5.1 推导（纯文本）

```
记 p = 数据分布或教师分布，q = 学生 q_theta。

(1) SFT 的交叉熵 = forward KL + 常数
    KL(p || q) = E_{y~p}[ log p(y|x) - log q(y|x) ]
               = -H(p)  +  E_{y~p}[ -log q(y|x) ]
                 ^^^^^^    ^^^^^^^^^^^^^^^^^^^^^
                 与 theta 无关       = 交叉熵 = SFT 的 loss（p 取经验数据分布时就是 NLL）
    => min_theta CE  等价于  min_theta KL(p || q)        # forward KL，样本来自 p（数据或教师）

    逐 token 的软标签蒸馏同理：
    sum_v p_T(v|ctx) * (-log q(v|ctx)) = KL(p_T || q)(ctx) + H(p_T(.|ctx))

(2) on-policy 蒸馏常用 reverse KL
    KL(q || p_T) = E_{y~q}[ log q(y|x) - log p_T(y|x) ]
    期望是对学生自己的分布 q 取的，所以天然要在学生的采样上估计（on-policy）。
    逐 token 估计：在学生生成的 y 上，对每个位置 t 累加 log q(y_t|y_<t,x) - log p_T(y_t|y_<t,x)。
    等价于一个 RL 问题：每步奖励 r_t = log p_T(y_t|...) - log q(y_t|...)（MiniLLM 式 2）。

(3) 为什么 reverse KL 是 mode-seeking，forward KL 是 mass-covering / mean-seeking
    forward  KL(p||q)：在 p>0 而 q≈0 的地方，log(p/q) 会变得很大，所以 q 必须覆盖 p 的所有峰。
                       学生容量不够时，概率质量会被摊到峰与峰之间 p≈0 的区域，生成低质量样本。
    reverse  KL(q||p)：在 q>0 而 p≈0 的地方，log(q/p) 会变得很大，所以 q 不敢往教师认为不可能的地方放概率。
                       q 可以放弃 p 的一些小峰，只锁定大峰。结果是更“准”，但多样性更低。
```

### 5.2 权威出处

- **GKD**（arXiv 2306.13649v3，Agarwal et al.）
  - p3 §2：把 `D_KL(P||Q)` 定义为 forward KL，`D_KL(Q||P)` 定义为 reverse KL；
    原句为 “Forward KL under an empirical data distribution corresponds to maximum likelihood, which we optimize in supervised learning”，这条可以直接作为“SFT = forward KL”的出处。
  - p3 §2：SFT 的 loss 写作 `L_SFT = E_{(x,y)}[-log p_S(y|x)]`。
  - p5 §3.1 “Choice of Divergence”：forward KL 要求学生覆盖教师的全部支撑，可能产生幻觉或低质量生成；
    reverse KL 等 mode-seeking 散度只盯教师给高概率的 token，代价是多样性下降。p6 图 4 用实验展示了从 forward 到 reverse 多样性逐渐下降。
  - 原文笔误：p3 那句 “minimizing the reverse and forward KL results in mean and mode-seeking behavior” 的对应顺序写反了，以 p5 和图 4 的表述为准，即 reverse 对应 mode-seeking。
  - **要注意**：GKD 的 on-policy KD 式 4（p4）是 `L_OD = E_x E_{y~p_S}[ D_KL(p_T || p_S)(y|x) ]`，
    也就是**在学生采样上算 forward KL**。GKD 把“数据从哪来（学生采样占比 lambda）”和“用哪种散度（forward/reverse/JSD）”当成两个独立的维度（p4 式 5–6），
    并在 p5 建议和 RL 结合时用 reverse KL 或 JSD(0.9)。
    所以讲义不要写成“on-policy 蒸馏 = reverse KL”这种定义式等号。准确说法是：on-policy 蒸馏**通常**搭配 reverse KL，因为 reverse KL 的期望本来就是对学生分布取的，只能用学生自己的样本来估计。
- **MiniLLM**（arXiv 2306.08543v6，Gu et al.）
  - p2 §1：标准 KD（包括序列级 KD）本质上是最小化 forward KL `KL[p||q_theta]`，会逼学生覆盖教师的所有 mode。
  - p3 §2：forward KL 写作 `KL[p||q] = E_{x~p_x, y~p'}[ log p(y|x)/q(y|x) ]`，其中 p' 取真实数据时是 word-level KD，取教师分布时是 sequence-level KD。
  - p3 §2.1 式 1：MiniLLM 的目标是 `min KL[q_theta || p] = min -E_{x, y~q_theta}[ log p(y|x)/q_theta(y|x) ]`，即 reverse KL。
    文中说 “Minimizing reverse KLD has been shown to cause the mode-seeking behavior”。
  - p3 §2.2 式 2（标题就叫 “On-Policy Distillation”）：用策略梯度定理求导，
    `grad L = -E_{y~q}[ sum_t (R_t - 1) * grad log q(y_t|y_<t,x) ]`，其中 `R_t = sum_{t'>=t} log( p(y_t'|...) / q(y_t'|...) )`。
  - p3 图 3：左边是序列级 KD，在教师样本上做 forward KL；右边是 MiniLLM，在学生样本上做 reverse KL。适合直接当讲义配图的参照。
