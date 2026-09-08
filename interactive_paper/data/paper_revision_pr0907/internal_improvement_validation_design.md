# Internal-improvement 小批 validation 设计（2026-09-08，CPU 阶段交付）

**协议更新（作者 2026-09-08）：取消三次 repeats。** 冻结配置后每个
query/arm 只运行一次（schema 保留 repeat_id，固定 0）；跨运行波动引用
`repeat_variance.json`（v1 协议 r0/r1/r2），样本不确定性用同题配对
query bootstrap（不重跑模型）。先 internal，再按时间推进外部池。

## 0. 与 2026-09-08 任务单的对账（避免重复消耗）

| 任务单条目 | 状态 | 位置 |
|---|---|---|
| P0.1 阈值/去重导出 | 已交付 | thresholds.json（fresh no-leak 10–50% OOF grid + fit provenance）、calibration_oof.parquet、leak_audit.json；gate_native_noleak.json 从未部署（8cv 核实） |
| P0.2 native bench manifest | 已交付 | manifest.json（46 runs, experiment_family/in_paper/provenance）、artifact_index.jsonl（~530 volume files） |
| P0.3 三处评测边界 | 已核实：v1 全部 runs 存在；v2（modal_native_bench2.py）已修 | STATUS.md、8cv/8cw 段；v2 = 因果输入+sha、两路音频+audio_asr judge、单调时间戳、状态行、seed、run_id 键 |
| P1.4 因果输入/计时 | v2 已实现；trunc1 已实测（43 长 knowledge，expert acc .310→.512） | 8cx、trunc1_ids.txt、internal_gain_analysis.json |
| P1.5 档位冻结 | 未完成——validation sweep 需在输入+relay 冻结后重做（旧 sweep 是 v1 输入） | validation_sweep.parquet=旧版仅参考；8bz 15/25/40 未确认 |
| 本轮新增 CPU 改动 | 本 commit | 见 §1 |

## 1. 本轮 CPU 改动（已完成，未启动模型）

- `src/relay_fmt.py`：v2 formatter 原样移出（1064 条已录 expert 文本
  行为 diff = 0）+ 保守 v3（保留代码/表格/比较符内容，转换 LaTeX 展示
  宏，答案句优先，超预算从前端丢 rationale，长度上限同 v2）。
- `modal_native_bench2.py`：`--fmt v2|v3` 与 `--expert web|structured`
  按 run 选择；rec 记录 relay_fmt / expert_mode / gen_top_k。
- `src/escalate.py`：`ask_expert_structured`（final_answer /
  concise_explanation / spoken_answer，Structured Outputs，无工具，
  不缓存）。
- judge 内容缓存（Exp-4）：`/data/native_bench_v2/judge_cache.jsonl`，
  键 = judge 配置签名 + query + reference + answer；相同内容跨
  field/arm/run 复用同一 verdict；judge 失败不入缓存。
- `scripts/55_relay_fmt_offline.py` 离线诊断（$0，结果
  `relay_fmt_offline.json`）：
  - 回归案例：q0091 "64" v2✓v3✓；q0046 "36" v2✓v3✓；q0486
    `<button>` v2✗(输出为空) v3✓。
  - reference-token 保留率 v2≈v3≈raw expert（结论保留问题已被
    fccff8b 的 v2 conclusion-keep 解决）；v3 的净收益在：空输出
    1→0，TTS 会读出的残余展示语法 frozen 22→8、sreason 9→2，代码
    保留。**含义：formatter 单独救不回全部 relay-loss 行**，其余
    要靠候选一（structured expert）或本就是 judge 波动——与任务单
    预期一致。
  - striviaqa 保留率 .112 为度量伪影（参考答案为别名列表），
    raw expert 同值，不代表 formatter 损失。

审计文件缺口：任务单引用的
`review_sources/internal_improvement_audit_2026-09-08.json` 不在本仓库
（全盘查找无结果），本轮以 native_bench parquets + scripts/54 结果为
依据；若该文件在别处生成，请提供或忽略。

## 2. 分阶段小批 validation（GPU/API 按预算逐步批准）

冻结顺序：输入修复（已在 v2）→ relay → gate 档位 → 正式单次运行。
所有阶段：独立 run_id + 独立输出目录（`_todo()`/`judge_pool()` 按 id
跳过，绝不能与旧 run 共用 run_id）；smoke/validation 不进正式汇总。

### V1 输入边界 smoke（GPU ~1h，~$10–15）
固定 smoke set（不属于 internal test）：
- 长题：validation split 内 hard-knowledge/math 按 volume 实际
  audio_s 选 >30s 的全部（候选按文本长度代理排序：q0262 q0266 x0427
  q0277 q0182 x0463 q0161 x0410 q0113 q0272 x0502 q0088，运行时以
  audio_s 复核）；
- 短题对照 5：x0019 x0022 x0235 x0251 q0369；
- zh 对照 5：expansion4zh（校准来源，非 test）；
- early-onset 对照：以往记录 onset_chunk+1 < n_chunks 的 validation id 取 5。
检查：expert_input_sha256/秒数边界、题干-选项-数字完整（ASR uplink 对照
原题）、时间戳单调、恢复键、audio_asr 产出。经 v2 causal 输入不读未来
音频；early-onset 行确认 expert 只拿已消费前缀。

### V2 relay 比较（先文本后音频）
- V2a（仅 API，~$10–15）：已录 always-arm expert 文本 × {v2, v3}
  formatter → 统一盲化 judge（走内容缓存，相同文本自动去重）。产出
  文本级 Δacc；只作 relay 消融，不冒充 live arm。
- V2b（GPU+API，~$40–60）：validation split ~60 fired 子集，arms =
  {胜出 formatter, structured expert}；同 run 内合成实际音频并按
  audio_asr 评分。选定 relay 协议并冻结。
  记录：14 条 expert 对/relay 错是 always 的记录且含 judge 波动，
  不作为可修复上限承诺。

### V3 gate 档位（在 V1/V2 冻结之后；GPU ~$80–120）
- validation split 上以 v2 协议 + 冻结 relay 跑 never/always 两臂
  （299 × 2 sessions），据此做 analytical remix 的
  accuracy–call-rate–latency 曲线（matched-rate random 参考 =
  同版本逐题混合，声明非新跑 arm）。
- 用已核实 no-leak OOF 网格导出各语言 nominal 10–50% 的分数阈值；
  选三档 + 理由，冻结 fit/threshold/serving/judge。15/25/40 只是
  候选，8bz 未确认。
- Exp-3 候选（净收益头 E[correct_relay−correct_local|frozen feats]）：
  先盘点校准来源已有的成对 local/expert-relay outcomes 是否够训练；
  够则 $0 拟合 + validation OOF 对比 failure probe（同特征消融），
  不够先报缺口要预算。仅在 matched-rate 上稳定优于 baseline 才进
  正式档位；否则保留原 probe(结果照报)。晚读取点候选仅在此步失败后
  评估，等待时间计入成本。

### F 正式单次评估（时间允许时逐池推进）
- Internal 240 × {local, 3 档 gate, always} × 1 run = 1,200 sessions
  （新 run_id 族、attempt 字段记录基础设施重试、arm 交错）；随后
  striviaqa 250、sdqa 200（各 ×5 arms），WebQ/Llama/zh-Reasoning 随
  时间补齐；AlpacaEval 独立 1–5 分。
- 汇总：分 run n/升级率/文本 acc/audio_asr acc/completion-timeout-
  non_commit 率；onset 与 completion latency mean/P50/P95（服务器
  PCM-ready 命名，不称客户端可听）；stall onset 单列；配对 bootstrap。
- 费用重估（非三 repeats 的旧数）：以 8cx 实测 ~$0.5–0.6/escalated
  session（GPU+expert+judge，judge 缓存后更低）与 local-only
  ~$0.2–0.3/session 估：internal 1,200 ≈ $450–600；三池全做 3,450 ≈
  $1.3–1.9k。正式启动前在 STATUS.md 按当刻价格复核后再批。
- internal 已用于错误诊断 → 泛化审计 holdout 按任务单 P2 流程另行
  返回候选清单（含 overlap audit），不在本文件承诺。

## 3. 纪律（沿用任务单）
不按 query ID/来源标签写路由特例；不删题；歧义判分统一规则复核并保留
旧标签与修改依据；全部行保留（timeout/non-commit/empty 计入完成率）；
负结果照报；输入/relay/gate 的收益分开归因，不全记给 gate。
