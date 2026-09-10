# 真实 TTFA 补跑 — 最终报告（2026-09-09，run_id=ttfa1，全部完成）

## 一句话结论

**5 池 × 5 臂 = 5,950 条正式 session 全部真实运行完成**（协议 ttfa-v3，逐条实测、非阶段拼接），0 条时间戳顺序违规、0 条重复、失败 44 条（0.7%，全部保留原因）。主结果：TTFA_server（P50）local 0.6-1.7s → conservative 0.8-1.8s → balanced 0.9-2.2s → aggressive 1.7-5.1s → always 4.3-6.7s；always mean 5.1-8.0s。终表见 `summary_ttfa1_final.md/json`。

## 计时定义（精确起止事件）

- **时钟**：单 worker 内 `time.monotonic`，同源直减；缺失 null，不填 0。
- **input_end** = feed_start + 输入音频真实时长（最后一个真实样本按实时回放的到达时刻；实际喂入完成另记 `last_real_feed_done`）。chunk i 仅在 feed_start+min((i+1)·1s, 音频时长) 后可被消费——严格因果，输入按原始采样率实时推进，绝无加速。
- **first_answer_pcm**：
  - 未升级：duplex（`generate_audio=True`，部署 demo 同配置）第一个非 listen、含非空 PCM 的 chunk 调用**返回时刻**。接口为 chunk 级阻塞流式，此为该接口下首块可交付时刻（已如实标注实现粒度）。
  - 升级：teacher-forced talker 合成按生成器逐块迭代，**首块非空 PCM 被 yield 的瞬间**打点（实测首块比整段合成结束平均早约 8s，证明确非整段后补）。
- **TTFA_server = first_answer_pcm − input_end**，带符号。early-response（负值）保留不截断：frozen 各臂 7-17 条/240。
- **不计入回答 TTFA、单独记录**：stall 音频（`stall_first_pcm`）；onset chunk 候选音频（`candidate_first_pcm`，部署释放规则：gate 在 listen→speak commit 读取时该 chunk ~1s 音频已发出——demo_duplex 8cp 注释）。两者进 `first_any_pcm`。
- **均为服务端 PCM-ready 时间**；无客户端播放回调，不得表述为用户听到的时延。
- Expert 只见触发时**已到达**的真实音频前缀（样本数+SHA256 逐条记录，随机抽验通过）；`ask_expert_web` 无缓存，逐条真实调用（gpt-transcribe ASR + gpt-5.5 web low）。
- 等待策略：EOT 后 50ms 粒度轮询 expert 完成（v1 为 1s 量化，此处更贴近部署即时转发）；MAX_WAIT 150s。

## 冻结配置（freeze.json）

commit `44bdf2f`；gate `/data/gate_native.json` SHA256 `0e6494c2eeac…`（volume=本地副本）；act gate `f2e4e642…`；en 阈值 conservative .9057 / balanced .7973 / aggressive .5534（名义 15/30/50）；读取点 L22、k=8（listen→speak commit）；top_k=20、force_listen=3、官方 system prompt；relay=teacher-forced tts + clean_expert_v2；模型 config.json SHA `b0c03cf9…`、index SHA `e578de05…`；5 池 query 文件 SHA + 逐文件音频 manifest（`input_manifest.jsonl`）。测试期间未重拟合 gate、未调阈值、未筛选结果。硬件：Modal H100 80GB；每 pool×arm 8 容器（抽样臂 3），容器内串行 + 固定 warm-up（6 chunk + 1 次合成，不入结果）。每 query 一次 attempt，失败保留；断点续跑按已记录 ID 跳过。

## 最终结果（全量实测，mean / P50 / P95 秒；esc%=实际触发率）

| pool | local | conservative | balanced | aggressive | always |
|---|---|---|---|---|---|
| frozen (240) | -0.61 / 0.70 / 2.62 | 1.53 / 0.89 / 10.85 (9.7%) | 2.36 / 1.61 / 14.41 (25.9%) | 4.59 / 2.87 / 18.14 (51.7%) | 8.01 / 6.67 / 25.86 |
| striviaqa (250) | 1.14 / 0.90 / 1.92 | 1.55 / 0.98 / 2.63 (3.6%) | 2.45 / 1.46 / 8.46 (18.4%) | 3.62 / 3.55 / 7.92 (56.4%) | 5.71 / 4.89 / 10.30 |
| swebq (250) | 3.69† / 1.71 / 3.60 | 5.14† / 1.81 / 7.71 (5.7%) | 3.62 / 2.18 / 9.88 (21.7%) | 7.40 / 5.07 / 13.25 (60.3%) | 7.49 / 6.13 / 16.71 |
| sllama (250) | 1.61 / 1.52 / 2.55 | 1.63 / 1.54 / 2.55 (0%) | 2.10 / 1.53 / 3.82 (4.0%) | 2.61 / 1.68 / 8.49 (19.5%) | 5.13 / 4.32 / 9.45 |
| sdqa (200) | 0.93 / 0.62 / 1.67 | 1.50 / 0.75 / 7.18 (6.5%) | 2.60 / 0.87 / 10.81 (24.5%) | 4.66 / 4.20 / 10.17 (65.0%) | 5.85 / 5.10 / 10.64 |

† swebq local/conservative 的 mean 被 5/500 条"迟迟不 commit"离群 session 拉高（输入仅 ~1.9s，head 在 listen 停留 114-306 个静音 chunk 后才开口，TTFA 25-305s；真实模型行为，逐条可查）；该两格引用 P50/P95。

其他交付字段（JSON 内）：P99、`ttfa_first_any`、gate 臂内 local/escalated 两路分布（`ttfa_local_path`/`ttfa_escalated_path`）、early-response 计数、状态分布、共同有效 ID 配对表（`paired`）、逐行 `chunks` 配速轨迹（实时 lag：输入期间 0ms，fire 后管线阻塞 ~2.5-3s 如实记录）。

## v1 重构口径（附录/验证用，非实测）

`recon_ttfa1_final.{json,md}`：用 v1 主实验日志按名义 chunk 时钟重构（head 为 chunk 同步；expert_latency_s 为真实墙钟），校准常数 d_proc=0.60s、c_relay=0.72s（来自 5,906 条实测行）。分布级验证：striviaqa/sllama/sdqa 各臂 MAE 0.2-2s；frozen 配对 MAE 大（长输入下 onset 逐次随机性所致，非时钟误差）。**已有全量实测，重构仅作方法学验证，不用于任何主数字。**

## 失败与边界（全部保留）

- 44/5,950 失败：non_commit（onset 从未出现）、expert_error/timeout、error（会话异常），逐条含原因；不重试、不剔除。
- frozen/local 的负 mean（-0.61s）由 17 条 early-response 驱动（长输入未结束模型已开口），P50 0.70s 为典型值。
- smoke 行（.smoke 后缀）与正式 shard 分离，未混入。
- v2 val1、299-validation、旧 turn-based TTS 实验均未混入本批。

## 文件与可复现

`interactive_paper/modal_ttfa_bench.py`（权威实现）+ `ttfa_real/`：`run_lane.sh`/`run_sample_*.sh`（准确运行命令）、`local_ttfa_bench.py`（无 Modal 同协议本地 runner）、`fetch_inputs.sh`/`fetch_results.sh`、`summarize_ttfa.py`/`verify_rows.py`/`recon_ttfa.py`、`freeze.json`、`sample_ids.json`（时长分层系统抽样，池内 5 臂配对同 ID）、各池 `*_manifest.jsonl`、`summary_ttfa1_final.{json,md}`、`recon_ttfa1_final.{json,md}`、`results/ttfa1/`（全部逐条日志）。输出音频（每 session 的 local/candidate/stall/relay wav + SHA256）在 volume `gate-data:/data/ttfa_real/ttfa1/{pool}/audio/`。

**未改动**：论文、图、主表、v1/v2 历史数据、gate 与阈值。
