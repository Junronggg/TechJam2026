# TechJam 推荐系统研究记录

> 记录已经实际运行的实验、数值变化、判断依据与研究路线变化。  
> 除 rolling mean 外，分数均为 Validation Primary；研究阶段不根据 Test 结果选择模型。

## 一页结论

当前提交候选：

```text
FM + BPR                    60%
DeepFM + BCE                40%
Validation Primary     0.604713
相对 FM+BCE baseline  +0.003243
Rolling validation      3/3 folds 提升
Rolling mean delta      +0.001123（相对 FM+BPR）
```

值得保留、但不能直接替换冠军的单模型方向：

| 候选 | 单次 Primary | 稳定性证据 | 当前定位 |
|---|---:|---|---|
| Like-only Multi-task DeepFM | 0.604400 | rolling 3/3，平均 +0.000309 vs DeepFM | 有效辅助任务 |
| Low-rank DCNv2 | 0.604164 | rolling 3/3，平均 +0.000248 vs DeepFM | 有效单模型，但不适合当前融合 |
| FM+BPR + `global_context` | 0.604394 | rolling 3/3；paired seeds 3/4 | 有希望，但统计区间跨 0 |

已有充分负面证据的路线：全局 target rate、稀疏显式交叉、更多随机负样本、当前 semi-hard negative、BCE+BPR hybrid、3-day temporal counts、candidate-history 标量、completion/log-watch auxiliary，以及把 DCNv2 直接塞进现有 ensemble。

稳健性提醒：`0.604713` 是 standard-exposure validation 冠军；在开发期 random exposure 上，ensemble 比 FM+BPR 低 `0.000155`，不能把它描述成对曝光策略变化也稳定。

---
| 阶段 | 实验 | Primary / rolling 结果 | 相对变化 | 判断与原因 |
|---|---|---:|---:|---|
| 0 | FM + BCE baseline | 0.601470 | — | 官方基线，可复现 |
| 1 | FM + BPR，lr=0.001 | 0.603396 | +0.001926 | KEEP；ranking loss 更符合 GAUC/nDCG |
| 1 | FM + BPR，lr=0.0005 | 0.603696 | +0.002226 vs BCE | KEEP；较低学习率更稳定 |
| 1 | FM + BPR，lr=0.0003 | 0.603963 | +0.002493 vs BCE | KEEP；当前 BPR seed-0 基线 |
| 1 | BPR paired seeds 0–3 | mean +0.002344 vs BCE | 4 seeds 均值提升 | 证明不是只挑中了一个好 seed |
| 2 | 2 negatives / positive | 0.603379 | 基本无增益 | REJECT；更多 pair 没带来新信息 |
| 2 | 4 negatives / positive | 0.602794 | 下降 | REJECT；训练更重且后期退化 |
| 2 | Semi-hard pool=2 / 4 | 0.601855 / 0.587747 | 明显下降 | REJECT；当前 hard-negative 定义引入噪声 |
| 2 | BCE+BPR hybrid，权重 0.25/0.5/0.75 | 最高 0.603962 | 未超过纯 BPR | REJECT；pointwise loss 没有额外帮助 |
| 3 | User/item 全局 long-view rate | 0.600448 / 0.591682 | 下降 | REJECT；全局平均太粗，分桶也损失信息 |
| 3 | User×tab / user×author 显式交叉 | 0.602869 / 0.602180 | 下降 | REJECT；稀疏交叉记忆噪声大 |
| 3 | 两个显式交叉一起 | 0.601198 | 下降更大 | REJECT；堆 feature 不等于互补 |
| 4 | LightGBM base | 0.599817 | -0.001653 vs FM BCE | REJECT；当前 sparse ID 场景不占优 |
| 4 | LightGBM + 连续历史统计 | 0.590084 | -0.011386 | REJECT；统计本身价值不足，不只是 FM 表达问题 |
| 4 | LightGBM + user×tab rate | 0.597528 | 下降 | REJECT；细分 rate 仍不稳定 |
| 5 | DeepFM + BCE，lr=0.001 | 0.603862 | +0.002392 vs FM BCE | 保留；非线性交互有效，但单独未超过最佳 BPR |
| 5 | DeepFM + BPR | 0.603530 | -0.000332 vs DeepFM BCE | REJECT；BPR 对 DeepFM 没复制 FM 上的收益 |
| 6 | FM+BPR + DeepFM ensemble，DeepFM weight=0.3 | 0.604562 | +0.000599 vs FM+BPR | 有效但不是最好权重 |
| 6 | 同上，weight=0.4 | **0.604713** | **+0.000750** | KEEP；rolling 3/3，平均 +0.001123 |
| 6 | 同上，weight=0.5 | 0.604203 | +0.000240 | REJECT；DeepFM 权重过高 |
| 7 | 3-day user/item temporal counts | 单 split 最高 0.605010 | 表面上涨 | rolling 仅 1/3，平均 -0.000246，REJECT |
| 8 | Multi-task：click-only | 0.604034 | +0.000172 vs DeepFM | 小幅，nDCG 略降 |
| 8 | Multi-task：like-only | **0.604400** | **+0.000538** | KEEP standalone；rolling 3/3，平均 +0.000309 |
| 8 | Multi-task：completion-only | 0.603876 | +0.000014 | REJECT；与 long_view 定义高度重复 |
| 8 | Multi-task：click+like | 0.604259 | +0.000397 | 被 like-only 超过；click 稀释收益 |
| 8 | click+like+completion | 0.604382 | +0.000520 | 未超过 like-only，结构更复杂 |
| 9 | FM+BPR + prior-video-positive | 0.604205 | +0.000242 | 表面 rolling 3/3，但 placebo 更高；行为解释 REJECT |
| 9 | FM+BPR + author-positive-recency | 0.604199 | +0.000236 | 表面 rolling 3/3，但 placebo 更高；行为解释 REJECT |
| 9 | 两个 sequence feature 一起 | 0.604169 | +0.000206 | 不叠加，且没有超过 constant placebo |
| 10 | Low-rank DCNv2，2 layers/rank 16 | 0.604164 | +0.000302 vs DeepFM | KEEP standalone；rolling 3/3，平均 +0.000248 |
| 11 | 用 DCNv2 替换 ensemble 的 DeepFM | 0.604317 | -0.000396 | REJECT；rolling 1/3 |
| 11 | FM + DeepFM + DCNv2 | 0.604616 | -0.000097 | REJECT；rolling 0/3 |
| 12 | Prior same-video count | 0.603936 | -0.000027 | REJECT；覆盖更高但没有模型增益 |
| 12 | Previous interaction same author | 0.604009 | +0.000046 | REJECT；远小于正常波动 |
| 12 | Capped log-watch regression | 0.603891 | +0.000029 vs DeepFM | REJECT；绝对 watch time 仍无额外收益 |
| 13 | Constant-field placebo | 0.604394 | +0.000431 vs FM+BPR | 证明 sequence 小涨主要来自新增 FM field |
| 13 | 正式 `global_context` | 0.604394 | +0.000431 | 候选；rolling 3/3、平均 +0.000810，但 paired seeds 仅 3/4 |
| 13 | `global_context` paired seeds 0–3 | mean +0.000333 | 3/4 seeds 为正 | 区间 [-0.000427, +0.001092] 跨 0，不能宣称已统计确认 |
| 13 | Global-context FM + DeepFM | 0.604674 | -0.000039 vs champion | rolling 3/3、平均 +0.000460；暂列候选，不替换冠军 |

## 评价口径

这里所说的“涨分”不是 classification accuracy。

```text
Primary = (GAUC + nDCG@5) / 2
```

- GAUC：同一用户内部，正样本能否排在负样本前面。
- nDCG@5：正样本能否出现在用户排序前五名的靠前位置。
- KEEP：除了单次分数，还要看 rolling、多 seed、泄漏检查及复杂度。
- REJECT：不代表代码错误，而是当前表示、模型或训练方式没有可靠增益。
- Candidate：存在正向证据，但稳定性或统计证据还不够强。

关键参照点：

| 参照 | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| 官方 FM+BCE 复现 | 0.667133 | 0.535806 | 0.601470 |
| FM+BPR，lr=0.0003，seed=0 | 0.670593 | 0.537333 | 0.603963 |
| 当前 0.6/0.4 ensemble | 0.671480 | 0.537945 | **0.604713** |

---

## 研究过程

### A. 先修正训练目标

最初的 FM 使用 BCE，把每条曝光当作独立二分类样本；但比赛指标评价的是同一用户内部的排序。因此第一个假设是：只改变 loss，不改变模型和输入，BPR 会更贴近 GAUC/nDCG。

| 对照 | Primary | 相对 FM+BCE |
|---|---:|---:|
| FM+BCE | 0.601470 | — |
| FM+BPR，lr=0.001 | 0.603396 | +0.001926 |
| FM+BPR，lr=0.0005 | 0.603696 | +0.002226 |
| FM+BPR，lr=0.0003 | **0.603963** | **+0.002493** |

Paired seeds 0–3 中，BPR 相对 BCE 的平均增量为 **+0.002344**。这说明主要提升来自训练目标与排序指标更一致，而不是某一个幸运 seed。

结论：BPR 成为 FM 的默认目标；`lr=0.0003` 成为后续 seed-0 对照配置。

### B. 检查负样本能否继续增强 BPR

尝试增加每个正样本的负样本数量，以及用当前模型分数选择 semi-hard negatives。

| 改动 | Primary | 结果 |
|---|---:|---|
| 1 negative / positive | 0.603963 | 对照 |
| 2 negatives / positive | 0.603379 | -0.000584，拒绝 |
| 4 negatives / positive | 0.602794 | -0.001169，拒绝 |
| Semi-hard pool=2 | 0.601855 | 明显下降 |
| Semi-hard pool=4 | 0.587747 | 严重下降 |

更多 pairs 使一次 epoch 的更新量变大，但没有增加真正有用的信息；当前 semi-hard 方法又容易反复选择噪声大或过难的负样本。

随后测试 BCE+BPR hybrid，BPR 权重 `0.25 / 0.50 / 0.75`，最高只有 `0.603962`，没有超过纯 BPR。说明 pointwise BCE 在当前 FM 上没有提供额外正则收益。

路线变化：停止继续扫 negative 数量、hard pool 和 hybrid 权重。

### C. 验证“历史平均是否有用”

这一阶段分别测试了全局用户倾向、视频质量、细粒度 user×context 偏好，以及连续统计表达。

| 表示方法 | 模型 | Primary | 判断 |
|---|---|---:|---|
| `user_long_view_rate` 分桶 | FM | 0.600448 | 低于 baseline |
| `item_long_view_rate` 分桶 | FM | 0.591682 | 大幅下降 |
| 连续 user/item rates + log counts | LightGBM | 0.590084 | 大幅下降 |
| `user_tab_long_view_rate` + count | LightGBM | 0.597528 | 低于 LightGBM base |
| `user×tab` categorical cross | FM+BPR | 0.602869 | 低于 BPR |
| `user×author` categorical cross | FM+BPR | 0.602180 | 低于 BPR |
| 两种 cross 同时加入 | FM+BPR | 0.601198 | 稀疏性进一步恶化 |

LightGBM 原始字段本身为 `0.599817`，也没有超过 FM。这组结果说明问题不只是“FM 把连续值分桶了”，而是这些全局历史平均在当前时间切分和候选排序任务中价值有限。

路线变化：不再继续复制 user/item click rate、like rate 等相似全局统计；新特征必须提供不同信息源。

### D. 从单模型转向互补模型

DeepFM+BCE 在相同基础字段上得到 `0.603862`。它没有超过 FM+BPR，但两者训练目标和非线性交互方式不同，因此测试融合。

| DeepFM 权重 | Ensemble Primary | 相对 FM+BPR |
|---:|---:|---:|
| 0.3 | 0.604562 | +0.000599 |
| 0.4 | **0.604713** | **+0.000750** |
| 0.5 | 0.604203 | +0.000240 |

权重 0.4 在三个 rolling folds 中全部超过 FM+BPR：

| 时间折 | FM+BPR | Ensemble | Delta |
|---|---:|---:|---:|
| Fold 1 | 0.610742 | 0.611392 | +0.000650 |
| Fold 2 | 0.580129 | 0.581429 | +0.001301 |
| Fold 3 | 0.586890 | 0.588308 | +0.001418 |
| Mean | 0.592587 | **0.593710** | **+0.001123** |

结论：当前最好方案不是最强单模型，而是 ranking signal 与 nonlinear pointwise signal 的融合。

### E. 用 rolling validation 推翻单切分假象

严格过去的 3-day user activity 和 item exposure 在单个 validation 上把 ensemble 推到 `0.605010`，看起来超过冠军。

但 rolling 结果只在 1/3 folds 提升，平均相对原 ensemble 为 **-0.000246**。这说明近期计数依赖具体时间段，不能因为一个 validation 上涨就接受。

路线变化：从这一阶段开始，所有小幅提升都必须经过 rolling；单 split 只负责筛选候选。

### F. 拆开 multi-task 辅助信号

辅助任务只参与训练，不作为预测时输入。目标是判断其他用户行为是否能帮助模型学习 long-view。

| 辅助任务 | Primary | 相对 DeepFM | 解释 |
|---|---:|---:|---|
| Click | 0.604034 | +0.000172 | 有小涨，但 nDCG 略降 |
| Like | **0.604400** | **+0.000538** | 最有效；rolling 3/3，平均 +0.000309 |
| Completion ratio | 0.603876 | +0.000014 | 与 long_view 定义高度重复 |
| Click + Like | 0.604259 | +0.000397 | click 稀释 like 收益 |
| Click + Like + Completion | 0.604382 | +0.000520 | 更复杂但未超过 like-only |
| Capped log-watch regression | 0.603891 | +0.000029 | 绝对观看时长也没有新增益 |

EDA 中 watch ratio 与互动强相关，但相关性不等于辅助监督一定有效。这里 completion 和 log-watch 都与主标签高度耦合，真正提供不同信息的是稀疏的 like 行为。

结论：保留 like-only Multi-task DeepFM；不继续堆 watch-time 变换或组合辅助任务。

### G. 审计 candidate-specific history

原始假设是“用户以前是否看好当前视频/作者”比全局平均更精准。初始结果看起来支持这个方向：

| Feature | Primary | 相对 FM+BPR | 初始 rolling |
|---|---:|---:|---|
| `prior_video_positive` | 0.604205 | +0.000242 | 3/3 |
| `author_positive_recency` | 0.604199 | +0.000236 | 3/3 |
| 两者一起 | 0.604169 | +0.000206 | 没有叠加 |

但数据覆盖审计发现：

- `prior_video_positive` 在 official validation 只命中 **38 / 124,909** 行，即 **0.0304%**。
- `author_positive_recency` 非零覆盖约 **0.7269%**。
- `previous_author_same` 命中 739 行；命中组 long-view rate **23.95%**，未命中组约 **31.37%**，没有复现“紧邻同作者是强正信号”的外部结论。

随后加入完全不携带行为信息的 constant-field placebo：

```text
FM+BPR baseline                 0.603963
real prior-video field          0.604205
real author-recency field       0.604199
constant all-zero field         0.604394
```

Placebo 比两个真实行为特征都高，因此不能把小涨归因于 candidate history；更合理的解释是“新增一个 FM field 改变了 embedding interaction 和优化轨迹”。

进一步的 target-free 对照也没有支持历史假设：

| Feature | Primary | Delta |
|---|---:|---:|
| Prior same-video exposure count | 0.603936 | -0.000027 |
| Previous interaction same author | 0.604009 | +0.000046，噪声量级 |

结论：代码是严格因果、无标签泄漏的，但这些标量历史信号在本数据上没有可靠价值。以后新增 categorical field 必须同时配一个 matched placebo。

### H. 将 placebo 现象变成正式模型假设

把常量字段正式命名为 `global_context`。它为 FM 提供一个可学习的全局向量，该向量可以与每个原始字段产生二阶交互。

单次结果：

```text
FM+BPR                         0.603963
FM+BPR + global_context        0.604394
Delta                         +0.000431
```

Rolling validation：

| Fold | Baseline | Global context | Delta |
|---|---:|---:|---:|
| Fold 1 | 0.610742 | 0.611830 | +0.001088 |
| Fold 2 | 0.580129 | 0.580913 | +0.000784 |
| Fold 3 | 0.586890 | 0.587449 | +0.000559 |
| Mean | 0.592587 | 0.593397 | **+0.000810** |

Paired seeds 0–3：

| Seed | FM+BPR | + global context | Paired delta |
|---:|---:|---:|---:|
| 0 | 0.603963 | 0.604394 | +0.000431 |
| 1 | 0.603352 | 0.604245 | +0.000893 |
| 2 | 0.603757 | 0.604027 | +0.000270 |
| 3 | 0.604128 | 0.603864 | -0.000264 |
| Mean | — | — | **+0.000333** |

Paired std 为 `0.000478`，近似 95% 区间为 `[-0.000427, +0.001092]`。虽然 rolling 3/3、seeds 3/4 为正，但区间跨过 0，因此只能标记为 Candidate，不能宣称统计确认。

用 global-context FM 替换冠军中的普通 FM 后：official validation `0.604674`，相对冠军 **-0.000039**；rolling 仍为 3/3 提升、平均 `+0.000460`。证据冲突，因此不替换冠军，也不围绕 validation 细扫融合权重。

### I. 检查更强交互模型是否能加入融合

Low-rank DCNv2 使用 2 层 cross、rank 16，得到 `0.604164`，相对 DeepFM `+0.000302`；rolling 3/3，平均 `+0.000248`。作为单模型，这是正向结果。

但是加入 ensemble 后：

| 融合方式 | Primary | Rolling |
|---|---:|---|
| 用 DCNv2 替换 DeepFM | 0.604317 | 1/3，拒绝 |
| FM + DeepFM + DCNv2 | 0.604616 | 0/3，拒绝 |

DeepFM 与 DCNv2 的预测相关系数达到 `0.9925–0.9963`，说明它们几乎在犯相同的错误。单模型更高不代表能给 ensemble 带来新信息。

结论：以后选择 ensemble 成员时，先检查预测差异，再考虑分数与权重。

### J. 从总体相关性升级到条件互补性

新增固定切片：history availability、item popularity、duration 和 validation early/late。所有切片只使用预测前可见信息构造。

以当前冠军为 Model A，候选单模型为 Model B：

| Model B | Overall delta | 相关系数 | 最好 slice | 最差 slice |
|---|---:|---:|---|---|
| Like-only Multi-task DeepFM | -0.000313 | 0.9721 | short video +0.001340 | cold history -0.002673 |
| DCNv2 | -0.000549 | 0.9708 | cold history +0.009087 | medium history -0.002251 |

进一步观察：

- Like-only 在 long video 上 `+0.001263`、head item 上 `+0.000704`，但 early/late 方向相反。
- DCNv2 的 cold 优势只覆盖 `1,627 / 124,909` 行，即 1.30%；不能只看 `+0.009087`。
- Like-only 修复冠军约 7.36% 的错误 pairs，同时在冠军原本正确的 pairs 中引入约 3.59% 新错误。
- DCNv2 修复约 7.24%，同时引入约 3.56%。

根据 cold slice 结果，预先固定一个可解释 gate：history length ≤ 2 时使用 `0.5 champion + 0.5 DCNv2`，其他样本保持冠军。

```text
Official validation delta    -0.000017
Rolling wins                  2/3
Rolling mean delta           +0.000012
Decision                      REJECT / noise
```

这说明“候选模型在某个 slice 上更高”仍不等于 gate 会提升整体用户内排序。切片用于产生假设，真正决策仍必须运行完整 gated score、rolling 和 multi-seed。

研究流程因此升级为：

```text
Hypothesis
→ strict-time/leakage audit
→ cheap controlled experiment
→ constant/shuffled/random placebo
→ rolling validation
→ paired seeds
→ fixed slice analysis
→ pair-error recovery/introduction
→ real gated/ensemble evaluation
→ KEEP / REJECT / REINTERPRET
→ research memory
```

### K. 首个 leakage-safe sequence model

没有安装额外深度学习框架，而是在现有 NumPy Runner 中实现了一个便宜的机制验证模型：

```text
strict last 16 interactions
+ video embedding
+ author embedding
+ train-only past behavior state
+ time-gap bucket
+ position embedding
→ single-head candidate-conditioned attention
→ DeepFM score
```

同 timestamp 不互读；validation/test label 不会进入 history。与 DeepFM 使用相同基础字段、BCE、seed、learning rate、epoch budget、early stopping 和 evaluator。

| 模型 | Best epoch | Primary | Runtime |
|---|---:|---:|---:|
| DeepFM | 6 | 0.603862 | 32.7 秒 |
| Lightweight Sequence DeepFM | 5 | 0.603369 | 813.4 秒 |

Sequence 相对 DeepFM 为 **-0.000493**，训练成本约 25 倍，因此不进入 K=32 或 rolling。

它相对冠军的条件诊断：

```text
Overall delta                    -0.001344
Within-user score correlation      0.9646
Champion pair-error recovery        8.96%
New pair-error introduction          4.39%
Best slice: short video           +0.001460
Worst slice: tail item            -0.006721
```

虽然预测比 DCNv2/Like-only 稍有差异，但固定 `90% champion + 10% sequence` 仍为 **-0.000023**。因此它既不满足 standalone，也没有通过一次预声明的 ENSEMBLE_ONLY 检查。

结论：REJECT 当前 NumPy last-16 attention；保留严格历史张量和诊断框架。若将来尝试真正 causal self-attention，应先换高效后端并提供明显不同的结构假设，而不是继续扫 K 或权重。

### L. Standard 与 random exposure 稳健性

仅使用 2022-04-22 至 2022-04-28 的 random-exposure 开发数据；4 月 29 日之后的行为标签在过滤前不会被解析。

| Exposure | 模型 | GAUC | nDCG@5 | Primary |
|---|---|---:|---:|---:|
| Standard | FM+BPR | 0.670593 | 0.537333 | 0.603963 |
| Standard | Champion ensemble | 0.671480 | 0.537945 | 0.604713 |
| Random | FM+BPR | 0.592296 | 0.147733 | 0.370014 |
| Random | Champion ensemble | 0.591785 | 0.147934 | 0.369860 |

Standard 中 ensemble 相对 FM 为 **+0.000750**；random 中变成 **-0.000155**，相对收益变化 **-0.000905**。

Random 的 long-view rate 为 **8.06%**，standard 为 **31.33%**。两套 Primary 的绝对值不能简单横向比较，因为候选集合和正例密度不同；真正可比的是同一 exposure split 内的模型差值。

结论：DeepFM 提供的互补性依赖 standard logging policy。当前冠军仍适用于比赛 standard split，但 FM+BPR 在 policy shift 下更稳健；项目叙事必须明确区分“榜单冠军”和“随机曝光鲁棒性”。

---

## 当前资产与状态

| 能力 | 已实现 | 最终判断 |
|---|---|---|
| FM + BCE/BPR/hybrid | 是 | BPR 保留；hybrid 拒绝 |
| DeepFM + BCE/BPR | 是 | BCE 保留，BPR 拒绝 |
| Multi-task DeepFM | 是 | like-only 保留 |
| Low-rank DCNv2 | 是 | 单模型保留，不加入当前 ensemble |
| LightGBM | 是 | 当前特征下拒绝 |
| 0.6 FM + 0.4 DeepFM ensemble | 是 | 当前冠军 |
| Rolling validation | 是 | 新方法稳定性门槛 |
| Candidate-history causal encoding | 是 | 信号无效，但实现和审计保留 |
| Auxiliary click/like/completion/log-watch | 是 | 仅 like 有可靠价值 |
| Prediction correlation analysis | 是 | 用于判断 ensemble 互补性 |
| Conditional complementarity | 是 | 分析 user/history/popularity/duration/time 条件差异与 pair error recovery |
| 自动 placebo controls | 是 | real/constant/shuffled/random same-cardinality；失败则 `REINTERPRET` |
| Rule-based history gate | 是 | 首个 DCNv2 cold gate 已验证并拒绝 |
| Strict last-K sequence tensors | 是 | video/author/behavior/time-gap；same timestamp 不互读 |
| Lightweight Sequence DeepFM | 是 | 已完成 controlled ablation；分数下降且成本约 25 倍，拒绝 |
| Random-exposure robustness | 是 | standard ensemble 增益未保持；random 下 FM+BPR 更好 |
| LLM-compatible evidence memory | 是 | 已记录支持、拒绝、不确定结论 |
| Full causal self-attention / BST | 否 | 仅在新后端和新假设成立时再做 |

## Agent 自主性升级（代码能力，不是新增分数实验）

当前 `python run_agent.py` 已统一进入以下闭环：

```text
读取 validation-only memory
→ 生成合法候选池
→ 按 expected gain / evidence / novelty / cost / redundancy 排序
→ 自动选择并训练一个候选
→ overall + fixed slices + error complementarity
→ KEEP / REJECT / STOP_DIRECTION / ENSEMBLE_ONLY
→ 必要时自动排队 placebo controls
→ 更新 research_memory.json
→ 继续或按 ε / 时间 / 次数停止
```

- `candidate_selection` 保存前五个候选和各评分项，可审计 Agent 为什么选择该实验。
- 小幅 categorical history gain 自动运行 constant、shuffled、same-cardinality random；control 永不成为冠军。
- 若 placebo 超过 real，原结果改为 `REINTERPRET`，冠军自动回滚。
- `manual_interventions.jsonl` 真实记录 reason/action/avoidable；标准无人干预 run 为 0。
- 开发 run 默认不计算 test；只有显式 `--finalize-test` 才在研究完成后评估一次。
- 自动化行为由 `tests/test_autonomy.py` 覆盖，不能只靠文档宣称。

### 首次 Planner memory 真实消融

使用相同初始配置、静态候选先验、数据、seed 和 `max_iterations=5`，分别运行三种模式；三组都不计算 test。

| Mode | Best Primary | Best iteration | 无效候选 | 时间 | 人工干预 | 候选 family 顺序 |
|---|---:|---:|---:|---:|---:|---|
| `no_memory` | 0.604713 | 2 | 2/4 | 301.6 秒 | 0 | BPR → ensemble → multitask → DCNv2 |
| `raw_history` | 0.604713 | 2 | 2/4 | 287.8 秒 | 0 | BPR → ensemble → multitask → DCNv2 |
| `distilled_patterns` | 0.604713 | 2 | 2/4 | 281.5 秒 | 0 | BPR → ensemble → multitask → DCNv2 |

结论：**当前短程实验没有证明 distilled patterns 带来 downstream value。** 三种模式的选择、最好分数和无效实验数完全相同，运行时间差只能视为系统波动。

原因不是 pattern 计算报错，而是前四个候选依次属于四个从未在本次 run 中试过的 family；pattern 只有在获得同类历史后才能影响排序，而 `ε=0.002, N=3` 的收敛窗口很快结束。当前状态必须表述为：pattern 机制已实现，但有效性尚未被证明。

针对这个负结果，Planner 现在会在每次真实选择时同步记录三种 memory mode 的反事实 top choice：

```text
counterfactual_choices
memory_changed_choice
```

这不会启动额外训练。`summary.json` 中的 `memory_influenced_selections` 将直接报告本次 trajectory 有多少次选择真的因 memory 而改变；只有该值大于 0，才值得进一步讨论 downstream reward。

### 首次完整 autonomous trajectory

命令：

```bash
python scripts/run_agent.py \
  --researcher deterministic \
  --memory-mode distilled_patterns \
  --max-iterations 50
```

运行目录：`logs/run_20260830T095900Z`

```text
Iteration 0  FM+BCE baseline       0.601470  REFERENCE
Iteration 1  FM+BPR lr=0.0003      0.603963  KEEP_CANDIDATE
Iteration 2  FM+DeepFM ensemble    0.604713  KEEP_CANDIDATE
Iteration 3  Like-only Multi-task  0.604400  REJECT
Iteration 4  DCNv2                 0.604164  REJECT
```

| 审计项 | 结果 |
|---|---:|
| Stop reason | `converged` |
| Total / candidate experiments | 5 / 4 |
| Best iteration / Primary | 2 / 0.604713 |
| Convergence streak | 3 |
| Wall clock | 280.9 秒 |
| Manual / avoidable interventions | 0 / 0 |
| LLM requests / tokens | 0 / 0 |
| Test metrics | `null` |
| Memory-influenced selections | 0 |

结论分开报告：

- **端到端 autonomy 已通过真实运行验证**：Agent 从 baseline 出发，自主选择四个实验、更新冠军、拒绝失败候选，并按官方收敛规则停止；过程中没有人指定下一项实验。
- **Pattern downstream value 未通过本次运行验证**：所有 counterfactual memory mode 的 top choice 相同，因此不能说 distilled patterns 改善了这条 trajectory。

### 跨 run memory 的离线压力测试

短程真实运行尚未让同一 family 被重复选择，因此新增：

```bash
python scripts/replay_planner_memory.py --max-steps 12
```

该脚本只读取 26 份历史 `experiment_history.jsonl` 中的 validation 指标，将旧配置归一到当前 schema，并以 34 个有日志支持的配置构造离线 replay。候选被选中前只检查“是否存在日志支持”，不会读取其分数；test summary 不加载。它是 Planner 决策压力测试，不是新的独立训练结果。

| Memory mode | Replay 数 | 无效实验 | 最后保留的配置 | 结果 |
|---|---:|---:|---:|---|
| `no_memory` | 6 | 2 | temporal 单 split 0.605010 | 重复进入 rolling 已否定的方向 |
| `raw_history` | 6 | 2 | temporal 单 split 0.605010 | 与无记忆相同，当前 run 的原始历史不足以阻止它 |
| `distilled_patterns` | 4 | 2 | 稳健 ensemble 0.604713 | 跳过 2 个已知不稳定 temporal 实验 |

这次结果证明的是“决策质量和效率”，不是更高的单 split 分数：

- 无记忆 Planner 被 `+0.000218`、`+0.000080` 的 temporal 小涨吸引，但已有 rolling 结果为 1/3 folds、平均 `-0.000246`。
- distilled Planner 读取机器可解析的 `family_policies`，将该 family 标为 `stop_direction`，避免重新采用已被更强证据否定的结论。
- 因此 memory 确实改变了 trajectory，并节省 2 次实验；代价是放弃表面更高、但更可能过拟合单 validation 的 `0.605010`。
- 限制：replay 重用了历史日志，不是独立 fresh trials；它验证 Planner 是否使用记忆，不能替代下一次真实 autonomous run。

## 四篇 Agent 推荐论文：我们真正学到的内容

这些论文研究的是 LLM 推荐 Agent 或对话推荐，不是 KuaiRand 上的自动 ML 调参。因此这里只迁移“如何规划、记忆、反思和探索”的机制，不把论文中的推荐分数当成我们模型有效性的证据。这里的自主改进特指：**根据实验历史更新 memory、research pattern 和 planning policy**；当前没有宣称参数级 self-learning 或 RL。

原文：

- [RecMind, NAACL 2024 Findings](https://aclanthology.org/2024.findings-naacl.271/)
- [TAIRA, arXiv 2506.23485](https://arxiv.org/abs/2506.23485)
- [STARec, CIKM 2025 / arXiv 2508.18812](https://arxiv.org/abs/2508.18812)
- [SAPIENT, NAACL 2025](https://aclanthology.org/2025.naacl-long.133/)

| 论文 | 论文中真正有效的机制 | 对本项目的迁移 | 不直接照搬的部分 |
|---|---|---|---|
| RecMind | Planning + personalized/world memory + tools；Self-Inspiring 在生成新分支时保留被放弃路径的信息 | 失败实验不能从 memory 消失；候选选择同时读取成功、失败、noise、placebo 和其他分支 | 让 LLM 直接对大量视频名称排序；论文也报告长候选列表的上下文与位置偏差 |
| TAIRA | Manager 分阶段规划并调用 Executor；把成功经验、修正后的失败经验和专家经验蒸馏成可检索 Thought Pattern；匹配 pattern 后再规划 | 将逐条实验蒸馏为 family-level `research_patterns`；执行前检索同类模式，决定 exploit、stop、ensemble-only 或补证据 | 为当前单机实验拆出多个 LLM Agent；会增加调用成本，现有 Controller/Runner 的职责边界已经足够 |
| STARec | 快速执行与慢速反思分离；用预测和真实反馈的差异更新 memory | 借鉴 fast/slow separation：Runner 执行实验；Controller 完成 validation、归因和诊断后，结论才进入长期 research memory | 没有声称复现其 memory validation、SFT 或 GRPO；这些需要不同任务和训练资源 |
| SAPIENT | MCTS 用 selection、expansion、simulation、reward back-propagation 做非贪心规划；UCT 平衡 exploration/exploitation | 只借鉴 exploration/exploitation balancing：使用 cheap value estimate 和小型 tree frontier，真实执行一个候选 | 不照搬 full MCTS；一次真实训练不是便宜 rollout，反复模拟会直接耗尽比赛预算 |

### 已根据论文补进代码的机制

`research_memory.json` 现在同时保存两层内容：

```text
hypotheses:
  每次实验的 validation 证据、诊断、结论

research_patterns:
  按实验 family 蒸馏出的可复用研究策略
```

每个 pattern 包含：

```text
task_description
solution_description
thought_template
policy
evidence:
  trials / positive / negative_or_noise / failed
  reinterpreted / slice_or_diversity
  mean_delta_from_parent / best_delta_from_parent
```

当前五种自动 policy：

- `exploit_with_confirmation`：有正向证据，但必须再做 rolling 或 paired-seed 确认。
- `ensemble_only`：overall 不够好，但 slice/diversity 表明可能只适合融合。
- `retest_with_control`：metric 上涨但归因可疑、control 尚未完成或 coverage 很低；先自动运行 constant、shuffled、same-cardinality controls。
- `gather_evidence`：证据不足，只允许一个便宜、单变量的验证。
- `stop_direction`：同一 family 已有至少两次负面/噪声/失败证据且无条件优势，停止重复搜索。

Deterministic 和 LLM Researcher 都会看到这些 patterns；被选候选会记录 `retrieved_pattern`。失败和 `REINTERPRET` 路径不会被删除，placebo control 也不能伪装成正向经验。

这是我们自己的 Controller 设计，组合了四类思想，而不是按四篇论文拼接流程：

- reflective validation（受 STARec fast/slow separation 启发）；
- full-path memory（受 RecMind Self-Inspiring 启发）；
- distilled research patterns（受 TAIRA TPD 启发）；
- budget-aware exploration（受 SAPIENT planning 启发）。

### 论文启发的下一批实验

| 优先级 | 实验 | 要回答的问题 | 通过标准 |
|---|---|---|---|
| 完成 | Full autonomous run | 能否无人干预完成观察、选择、执行、反思、停止？ | 已完成：5 次实验后 `converged`，interventions=0，test=null |
| 完成 | Harder planner replay | 在包含同 family 多个候选的历史中，pattern 是否改变选择？ | 已完成：trajectory 改变，并少做 2 个 rolling 已否定的实验 |
| 完成 | Cross-run validation memory 接线 | 上一轮的可靠结论能否在新 run 避免已知失败方向？ | Deterministic/LLM 都读取结构化 `family_policies`；test 不进入该文件 |
| P3 | Memory downstream-value test | 一条 distilled pattern 是否真的改变后续选择并减少浪费？ | 使用 `memory_changed_choice` 和下一实验 reward；无影响的 pattern 删除或降权 |
| P5 | Budgeted one-step lookahead | cheap value estimate 是否比纯 greedy 更好？ | 最多估计 2–3 个候选，只真实执行一个；不得以 lookahead 为名增加真实训练次数 |

暂不做：LLM 直接替代 FM/DeepFM 排序、GRPO 微调、完整 MCTS、依赖用户模拟器的对话 reward。它们在论文任务上合理，但对当前离线自动研究目标成本高、迁移证据弱。

## 下一轮决策规则

轻量因果 sequence encoder 和 standard-vs-random robustness 均已完成。当前不再扩展新论文机制或堆普通 feature；下一阶段首先证明 Agent 机制确实减少无效实验和人工干预。

顺序固定为：

1. 短程真实 memory ablation 无差异；该负结果保留，不用 replay 覆盖它。
2. 完整 autonomous trajectory 已完成：人不指定下一模型或 feature，5 次实验后自动收敛。
3. 历史 replay 已证明跨 run pattern 能改变选择，并阻止重新进入 rolling 已否定方向。
4. 下一步跑一次接入 persistent evidence 后的 fresh autonomous trajectory，确认执行层行为与 replay 一致。
5. 只有 fresh run 也证明减少浪费后，才继续做 one-step lookahead。

暂时不继续：

```text
更多全局 target rate
更多 user×author/user×tab 显式交叉
更多 hard-negative pool
更多 completion/watch-time 变换
围绕 0.604713 做小数点级 ensemble 权重搜索
挑选表现最好的 seed
```

## 复现实验入口

```bash
# 当前 Agent
python run_agent.py --researcher deterministic

# Planner memory ablation（必须使用相同 max-iterations / 时间预算）
python run_agent.py --researcher deterministic --memory-mode no_memory
python run_agent.py --researcher deterministic --memory-mode raw_history
python run_agent.py --researcher deterministic --memory-mode distilled_patterns
python scripts/run_memory_ablation.py --max-iterations 5
python scripts/replay_planner_memory.py --max-steps 12

# Rolling validation
python scripts/run_rolling_validation.py

# Multi-task 辅助信号拆解
python scripts/run_auxiliary_ablation.py
python scripts/run_multitask_rolling.py

# Candidate history、placebo 与数据覆盖
python analysis/candidate_history_audit.py
python scripts/run_candidate_history_followup.py
python scripts/run_sequence_placebo.py

# Slice、错误互补性与规则 gate
python scripts/analyze_conditional_complementarity.py
python scripts/evaluate_history_gated_ensemble.py
python scripts/run_lightweight_sequence_ablation.py
python scripts/evaluate_random_exposure_robustness.py

# DCNv2 与融合互补性
python scripts/run_dcnv2_ablation.py
python scripts/run_dcnv2_rolling.py
python scripts/evaluate_dcnv2_ensemble.py

# Global context
python scripts/run_global_context_ablation.py
python scripts/run_constant_context_rolling.py
python scripts/run_global_context_multiseed.py
python scripts/evaluate_global_context_ensemble.py
```

最后更新结论：**冠军不变；现在的优先事项是验证跨 run memory 在真实运行中避免已知失败，而不是继续添加模型或 feature。**
