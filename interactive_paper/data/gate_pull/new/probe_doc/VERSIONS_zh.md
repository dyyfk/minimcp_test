# NVDA probe：v1 → v2 → v3 做了什么，哪些尝试值得

*NemotronLabs-VoiceChat-11B（第二 duplex 家族），截至 2026-09-01。数字来源：`RESULTS.md` §8ac、`probe_doc/results_*.json`。*

## 三个版本

| | v1（8ac，08-19） | v2（08-31 → 09-01 凌晨，"deployed"） | v3（09-01 夜，review round） |
|---|---|---|---|
| 校准数据 | frozen 600 | frozen + expansion + expansion2 = 2,550（committed 2,481） | 同 v2；另做第三次独立 replay（pass 3） |
| 读点 | 用户音频结束（eot，末 8 帧） | **模型自己的 commit-to-speak**：首个 ≥3 帧非 PAD run，取其后 8 帧（8be 的 NVDA 类比） | 同 v2；另加两个 causal 读（commit 帧之前 8 帧 `H_pre`；到 commit 为止的运行均值 `H_run`） |
| 层 / 特征 | L34 · eot_last ‖ eot_mean8 ‖ user_mean | L30 · onset_last ‖ onset_mean8 ‖ user_mean（同配方，scaler + LR C=1e-4） | 配方不变；v2-arch = 三层（L26/30/34）logit 平均 + commit 帧；causal = commit ‖ pre_mean8 ‖ run_mean @L34 |
| calib OOF | .790 | .820 | v2-arch .824；causal .812 |
| 外部四池均值（cold） | .757 | **.808**（.838/.851/.771/.774） | v2-arch .816（+.007 [−.000,+.015]，pass 3 复现 +.004）；causal .781（−.029 [−.048,−.009]） |
| 其他 | — | act head（stop-word bug 修复，false-fire 69%→0%）；demo 上线 | remix / fire-rate / LOPO 宏平均 / 池内 AUC 体检；论文附录段 + 表 + 图；PR #7 |

## v1 → v2 做了什么（值得，+.05 外部 AUC）

1. **校准扩容 600 → 2,550**（与 MiniCPM 同一批 wav）：外部均值 .757 → .782，每层 OOF 都涨（+.013 ~ +.071），层曲线形状不变。**值得**——是全程最大的单项收益之一，斜率 ≈ +.013/1k 行。
2. **读点对齐到原生 duplex 的 commit-to-speak**（8be 类比；dual capture H_eot + H_onset）：外部 .782 → .808，四池全涨。**值得**——而且是"免费"的：部署读点比旧读点好，不用取舍。校准侧 OOF 三种读点在 .011 内（噪声），所以这个收益只在外部迁移上看得到。
3. **act head**（信息求解 vs floor-management 第二线性头）：196 条 stop/backchannel 刺激 100% 会 commit，failure probe 在 balanced 档 69% 误升级；act head 之后各档 0%，只损失 0.73% 真升级。**值得**——这是 8bh 在 NVDA 上的直接复现，且证明"两头分离"优于"负例混入 failure probe"（后者掉 .010 AUC）。
4. **hillclimb 18 变体**（C、scaler、多层拼接、双读点、MLP-64）：最好 +.002。**不值得再做**，但值得做一次——它给了"配方到顶"的证据，后面 v3 的 85 个变体是它的放大版。
5. **strict refit 镜像**（frozen calib-360 discipline）：eot .800 / onset .814，赢家 mmean L26。**值得**——是跨家族表里无星号的行。

## v2 → v3 做了什么（诊断值得，配方改动不值得）

**值得（改变了我们对 probe 的理解）：**

1. **两位独立 reviewer 审文档 + 85 个变体统一评测**（选择只用 calibration 侧 LOPO，外部只报告）：结论是**线性 probe 在这组特征上已到天花板**——外部 AUC 最多 +.013，且没有一个变体改变固定预算下的准确率（remix 四池均值 @15/30/50%：deployed .552/.650/.756，v2-arch .550/.650/.757，随机 .507/.577/.670）。省下以后在配方上花的时间。
2. **causal re-capture（pass 3，4×H100，3,500 条）**：回答了"读点在 commit 后 640 ms 算不算 pre-answer"——不算。严格 causal（只读 commit 帧之前）外部 AUC 掉 .025–.03，每层一致；commit 之前 8 帧窗口并不比旧的 eot 窗口好，所以多出来的信号就是模型开口的前几个 token。**但换成固定预算准确率只低 0.5–1 个点**（.546/.640/.749）。这是论文里该写的数字。
3. **user_mean 的 train/serve 偏差**（离线 = 到音频结束；在线状态机 = 到 commit）：cos .996，换成在线定义只掉 −.004 [−.009,+.001]。**值得**——streaming port 可以直接用在线定义，不用重新校准。
4. **pooled OOF 的"池身份"成分**：只输出池失败率的 oracle AUC .752；池内均值 .713；LOPO 池内均值 .687。**值得**——审稿人会问；也告诉我们扩数据要往 MCQ 类知识池（最弱且最均衡）加。
5. **fire rate 体检**：固定校准阈值在 Llama-Q 只触发 1/4/12%，在 SD-QA 触发 7/27/53%（名义 15/30/50）。**值得**——和 MiniCPM 上遇到四次的问题一样，解法就是 8bn/8bp 的 per-pool / windowed 分位数；NVDA 侧还没移植。
6. **remix eval 本地化**（`remix_eval.py`，与 `scripts/18` 同协议）：以后任何 probe 改动都能在 20 s 内看到"固定预算准确率 vs 随机"，而不是只看 AUC。

**不值得（试过，可以不用再试）：**

- PCA / 白化 / PLS / shrinkage-LDA 单头：全部 ≤ 0 或更差。
- 池/类别重加权（去捷径）：pooled OOF 掉 .03，外部不涨。
- robust / winsorised scaling、C 扫描、elastic net：≈ 0。
- 逐帧 40k 维特征、帧差分、早/晚半窗：≈ 0；双读点拼接 / 三读点 logit 平均：≤ 0。
- 标签噪声：两次 replay 之间 label 翻转只有 1.3%，soft label +.0001——**标签不是瓶颈**，不用再花 judge 钱在重标上。
- L26 vs L30：L26 略高（+.004 OOF，+.009 外部）但 LOPO 宏平均不占优，属噪声；不值得为它换部署层。

**唯一一致的小收益：** 三层 logit 平均 + commit 帧（v2-arch）：所有 calibration 指标都涨（pooled LOPO +.023 最明显），外部 +.007，pass 3 复现 +.004；但固定预算准确率不变。**可以换，不必换。**

## 现在真正能动的杠杆

1. **数据：expansion3**（2,300 条 en）——NVDA 上约需 2.7 GPU 小时；源音频不在仓库中。MiniCPM 上 exp3 让外部 .709 → .736；NVDA 的斜率约 2 倍。
2. **阈值：per-pool / windowed 分位数**移植（`scripts/26_pool_thresholds.py`）。
3. **streaming port**：读点/特征已确认可直接用；配置（QA system prompt、bf16）必须固定，否则需要重新导出特征。

## 与 MiniCPM 部署配方的差异（跨家族表要注明）

NVDA 用 StandardScaler（+.015–.023），MiniCPM 部署配方不用；C 都是 OOF 网格选（NVDA 1e-4，MiniCPM 3e-4）；judge 相同（gpt-5.4-mini）；NVDA 校准 2,481 vs MiniCPM 5,228。
