"""Phone-friendly gallery for the figure deliverable, with a written
interpretation under every figure (what it shows / why we win or lose /
whether the comparison is fair). Deploy: modal deploy gallery_app.py
PNGs are bundled at deploy time — redeploy to refresh."""
import os

import modal

TOKEN = "62dc5cd9"

# (file, title, verdict-class, interpretation-html)
FIGS = [
    ("native_regimechain", "图N1 · 原生全双工 — 读点 regime 排序 + 校准 scaling（2026-09-01）", "win", """
<b>这是什么</b>：系统换到了 MiniCPM 的<b>原生 full-duplex head</b> 上（mic 音频逐秒 prefill 进生成中的 context，
模型自己决定 listen/speak，探针在它"决定开口"的那个 chunk 读）——之前所有图的结果都来自 turn-based/harness loop。
<b>数字</b>：in-regime AUC 排序 turn-based .877/.771 &gt; <b>原生 .830/.709</b> &gt; concurrent .76/.689（internal/external mean），六个池无一例外。
<b>为什么重要</b>：§8bb 测到的"上线代价"大半是 harness 拼接读点的 artifact；部署的原生读点几乎免费，而且不需要任何外部 EOT 检测——<b>duplex head 的开口决策本身就是读点</b>。
<b>右图</b>：2310 行 in-regime 校准仍未饱和（internal .717→.830），scaling 形状与 concurrent regime 一致。
<b>不变的坏消息</b>：阈值三个 regime 都不可搬（balanced .386→.645），必须 in-regime 重定标。
"""),
    ("native_validity", "图N2 · 原生全双工 — gated accuracy vs matched-random（六池）", "win", """
<b>看什么</b>：蓝线=native 探针选择的升级曲线，灰虚线=同升级率随机选择；* 是置换 p&lt;.05。
<b>数字</b>：5/6 池显著——our pool .371→.596（44% 升级）、TriviaQA .612→.872、SD-QA .440→.830（70%）。
<b>为什么我们好</b>：native regime 的本地 floor 更低（同 240 题 .483→.371，⅓是解码温度、~8 分是真 regime 代价），
探针判别力却没掉——所以 gate 在部署 regime 里的<b>边际收益比 turn-based 还大</b>。
<b>为什么有一格不好</b>：Reasoning-zh 的 gate 完全不触发——全英文校准把中文分数压在所有全局阈值之下。
这是老问题（语言轴）换了个表达形式：从 AUC 崩塌变成 fire-rate 失灵。如实报告。
<b>第七池 AlpacaEval（图外，见论文表）</b>：turn-based 时代的纯 honest-negative，native 下 aggressive 档
出现弱显著信号（VB 4.12 vs 随机 4.03，p=.019，但 fire 只有 24%）——开放式生成仍是方法边界，只是边界挪了一点。
<b>公平吗</b>：expert 结果复用 always 臂缓存（8ad remix 算术），local 结果是 native 实测重判；正在跑的 live 验证臂会给出端到端对照。
"""),
    ("native_floor", "图N3 · 原生全双工 — floor control：零 harness 的打断/附和行为", "mixed", """
<b>这是什么</b>：旧 demo 用能量 VAD + ASR 词表实现"软打断"；现在全部拆掉，这张图测的是 <b>head 自己</b>的话轮行为（108 组 overlap 试验）。
<b>好的一半</b>：backchannel（"嗯""okay"）几乎从不打断它（误停 3/24），<b>没有任何 ASR/词表参与</b>——旧 harness 要靠 hosted ASR 才做到的事，模型原生就会。
<b>差的一半</b>：短命令"Stop!"只有 25% 在 6 秒内让位——原生 head 读的是<b>持续语音</b>不是命令；旧的能量 VAD 反而切得更快。这是自然度换可靠性的真实 trade，不藏。
<b>右图的发现</b>：escalation 等待期来一个<b>新问题</b>，70% 的 pending relay 会被冲掉（head 自己转场了）——这是"用户说话时要 abort 后台 thinker"的最强实证，已列为部署项。
<b>后续修复（8bh，已部署）</b>：用户发现 stop words 本身会触发 escalation 判断（"停"被送给 gpt-5.5）。
根因：full-duplex 里很多 commit 是话轮管理不是回答问题，失败 probe 在那里是分布外的（aggressive 档对 stop 命令误触发 45%）。
修复：同一 L22 读点再加一个"信息型 vs 话轮管理"线性头（OOF AUC 1.000，完全线性可分），双条件才升级——损失 0.5% 真升级，误触发归零。
<b>公平吗</b>：脚本化注入、固定偏移、in-process 与部署同一条 chunk 循环；stim 音频与 8ba 完全同一批。
"""),
    ("native_validity_official", "图N4 · 部署版 gate（8bq）— 官方 serving config + native 标签 + 分语言阈值，六池 validity（2026-09-02）", "win", """
<b>这是什么</b>：图N2 的升级版，也是<b>现在线上真正跑的那份 gate</b>：5228 行校准（官方 serving config 下重采：top_k=20、force_listen_count=3、官方 system prompt），
标签改为"模型在部署配置下自己的回答被 judge 判错"（不再是旧 turn-based 回答的标签），阈值按语言各自定标（en / zh 两套）。
<b>数字</b>：六池全部在 balanced 和 aggressive 两档显著优于同率随机（* 标）——our pool .431→.565（27% 升级）、TriviaQA .568→.852（54%）、SD-QA .455→.775（61%）。
<b>N2 里那格坏的修好了</b>：Reasoning-zh 之前"gate 完全不触发"，现在用 zh 阈值 conservative/balanced 档触发 17%/34%，两档都显著（.525→.609→.658）——这不是排序变好，是操作点终于落到中文分数的分布里。
<b>没变好的</b>：标签源换成 native 对<b>排序</b>是平局（外部五池均值 .755 vs .743，无一池置信区间排除 0，合作者独立复核同一结论），所以 8bq 是"来源正确 + 校准正确"的修正，不是性能升级；En-4 均值 .771 与 turn-based 参考 .771 持平。
<b>公平吗</b>：与 N2 同一套 remix 算术；expert 结果复用 always 臂缓存；local 结果是官方配置下 native 实测重判。数据与产物已公开在 HF dataset <code>dyyfk/minicpm-o45-native-gate-data</code>。
"""),
    ("shadow_compare", "图N5 · 合作者的两个 shadow 候选 vs 线上 gate：排序 +.02，级联只涨零点几（2026-09-02）", "mixed", """
<b>背景</b>：issue #8 里合作者在同一份 5228×12288 冻结特征上系统扫了 16 个实验（换标签、换表示、非线性头、PCA、多头路由）——<b>全部失败</b>，结论与我们的 8bq-3 一致：同一份 hidden state 已被一次线性读榨干。
真正有效的是<b>加一个独立信号</b>：让模型多采样两次回答算语义熵，再加 repeat-then-judge 的 p(True)，融合后外部均值 AUC +.033；但要多跑两三次生成，延迟中位 7 秒，不能同步上线。于是蒸馏成单次点积：<b>P9</b>（只用 eot_mean8+user_mean 两块，8192 维，老师=语义熵+RTJ）和 <b>P16</b>（线上分数 + P9 等权集成，折成一个 12288 维向量）。
<b>左图</b>：两候选在五个外部池上排序全部优于线上；P9 幅度大（均值 +.021），P16 更均匀（+.015，WebQ / SD-QA / Reasoning-zh 单池置信区间排除 0）。SD-QA（真人语音）是唯一两者都单独可靠的池——语义熵老师在 ASR 噪声主导的地方最有用。
<b>右图，也是保留的原因</b>：AUC 涨两分，级联准确率在 30% 预算只涨 +0.8 / +0.3 个点，而 oracle 空间有 5–18 个点——排序收益还没变成路由收益。
<b>前瞻验证</b>：合作者又冻结了三批 source-disjoint 集（WinoGrande/SciQ、BoolQ/HellaSwag/QASC、SNLI/SST-2/WiC，文本 benchmark 经 TTS）：P16 在第二批宏平均 +.014（CI 排除 0）但差 .0008 没到 +.015 门槛，第三批 SST-2 显著回退。结论：<b>不上线，进 shadow</b>——demo 里并行打分只记日志，等真实流量说话。
<b>公平吗</b>：三者用同一批行、同一份缓存 expert 结果、按池精确 15/30/50% 预算（候选没有阈值，只能这样比）；我们的复算与合作者贴出的数字逐位一致。
"""),
    ("native_floors", "图N6 · 七池本地 floor：原生 full-duplex vs 已下架的 harness loop（2026-09-02）", "mixed", """
<b>这是什么</b>：每个池 never 臂（不升级）的本地准确率，原生 duplex 会话 vs 旧 harness loop，同一判分器。
<b>数字</b>：原生 floor 普遍低 1–8 分（Reasoning-zh 差最多 .589→.510），our pool 反而高 2.5 分。这是 regime 本身的代价——原生 duplex 下模型在流式 prefill 里边听边答，
不是 harness 时代先听完再答——与 8be/8bq 的离线 remix 观察一致。
<b>公平吗</b>：同题、同判分；原生这边带 bootstrap 95% 区间。旧 harness 的数字只在这张图上出现，作为"为什么整套图都换了"的交代。
"""),
    ("native_striviaqa_dualview", "图1 · Speech TriviaQA（OpenAudioBench，含对比模型） — escalation vs acc，原生 full-duplex（2026-09-02）", "win", """
<b>这是什么</b>：图 3–14 那批曲线的原生 full-duplex 重跑：每题一个新的 duplex 会话，官方 serving config，部署中的 8bq gate 在 talker 决定开口的那个 chunk 读一次，
触发则真 ASR 上行 + 真 gpt-5.5，等待按 1 chunk/秒真实计时，专家答案由 talker 用自己的声音<b>逐字念出</b>（TTS relay，2026-09-02 起线上默认）。五档各自实测（实心点）；数据未到齐时中间档按 never 臂的实测 onset 分数分支（空心点）。
<b>判分</b>：与旧图同一把尺子（OpenAudioBench 官方 gpt-4o / VoiceBench gpt-4o-mini / 我们的 ref-anchored judge）。绿色虚线 = 专家原文直接判分，即 relay 无损时的上界。
<b>橙色虚线（有的池才有）</b>：同一个 gate、换成旧的"prompt 引导 talker 转述"relay 的实测曲线——talker 会截断、自答、99% 需要 nudge，在 TriviaQA 上把 always 臂从 .960 压到 .728。这就是 relay 改成 TTS 直读的原因。
<b>⭐ 仍是最有力的一张，而且现在是原生 full-duplex 的实测。</b> never .628 → balanced .700（18% 升级）→ aggressive <b>.884</b>（56%）→ always .960；专家原文上界 .964，即 relay 已无损。
对比线：MiniCPM 官方离线 .755、Qwen3-Omni-30B .629、Kimi-Audio .419——9B + 路由在实时 duplex 下 aggressive 档比 30B 单体高 25 分。
<b>橙色虚线</b>是同一个 gate 用旧 relay 的实测：always 只有 .728、aggressive .740——换 relay 前，路由收益的一大半都被 talker 转述吃掉了。
<b>没那么好的</b>：conservative 档 .604 低于 never 的 .628，4% 的触发率下这是本地回答重采样的噪声（top_k=20），置信区间覆盖；balanced 档只比随机线高一点。
<b>公平吗</b>：全部分数在 OpenAudioBench 官方判分器（gpt-4o + 官方 prompt）下；对比模型是离线数字，我们是实时流式。
"""),
    ("native_striviaqa_pareto", "图2 · Speech TriviaQA（OpenAudioBench，含对比模型） — latency vs acc，原生 full-duplex", "win", """
<b>横轴</b>：用户说完到回答结束的 P50 时长（秒）。原生 duplex 里 talker 可能在用户说完前就开口，所以本地臂的时长可以很短；升级臂 = stall + 等待专家（真实计时）+ relay。
<b>数字</b>：never 1.8 s → balanced 2.6 s → aggressive 6.4 s → always 10.2 s（含专家答案念完的时长）。+26 分（aggressive）多等 4.6 秒。
"""),
    ("native_swebq_dualview", "图3 · Speech Web Questions（OpenAudioBench，含对比模型） — escalation vs acc，原生 full-duplex（2026-09-02）", "win", """
<b>这是什么</b>：图 3–14 那批曲线的原生 full-duplex 重跑：每题一个新的 duplex 会话，官方 serving config，部署中的 8bq gate 在 talker 决定开口的那个 chunk 读一次，
触发则真 ASR 上行 + 真 gpt-5.5，等待按 1 chunk/秒真实计时，专家答案由 talker 用自己的声音<b>逐字念出</b>（TTS relay，2026-09-02 起线上默认）。五档各自实测（实心点）；数据未到齐时中间档按 never 臂的实测 onset 分数分支（空心点）。
<b>判分</b>：与旧图同一把尺子（OpenAudioBench 官方 gpt-4o / VoiceBench gpt-4o-mini / 我们的 ref-anchored judge）。绿色虚线 = 专家原文直接判分，即 relay 无损时的上界。
<b>橙色虚线（有的池才有）</b>：同一个 gate、换成旧的"prompt 引导 talker 转述"relay 的实测曲线——talker 会截断、自答、99% 需要 nudge，在 TriviaQA 上把 always 臂从 .960 压到 .728。这就是 relay 改成 TTS 直读的原因。
<b>数字</b>：never .528 → balanced .624（22%）→ aggressive <b>.752</b>（60%）→ always .796；专家原文上界 .823。aggressive 档超过 MiniCPM 官方离线 .702，与 Qwen3-Omni-30B 的 .749 持平。
<b>为什么天花板低</b>：WebQ 参考答案是 Freebase 实体列表、判分严格，gpt-5.5 自己也只有 .823；always 与上界的 3 分差是 400 字符念读上限截掉的长列表。
<b>橙色虚线</b>：旧 relay 下 always 只有 .564——比 TriviaQA 丢得还多，因为列表型答案最容易被 talker 截断或自答。
<b>公平吗</b>：同一把官方尺子；旧图里"我们自己的判分器只有 .464"的教训依然成立，跨来源比较必须先对齐判分器。
"""),
    ("native_swebq_pareto", "图4 · Speech Web Questions（OpenAudioBench，含对比模型） — latency vs acc，原生 full-duplex", "win", """
<b>横轴</b>：用户说完到回答结束的 P50 时长（秒）。原生 duplex 里 talker 可能在用户说完前就开口，所以本地臂的时长可以很短；升级臂 = stall + 等待专家（真实计时）+ relay。
<b>数字</b>：never 3.1 s → aggressive 11.3 s → always 20.4 s。WebQ 的专家答案是列表，念读时间长，always 档的延迟一半是念读。
"""),
    ("native_sllama_dualview", "图5 · Llama Questions（OpenAudioBench） — escalation vs acc，原生 full-duplex（2026-09-02）", "mixed", """
<b>这是什么</b>：图 3–14 那批曲线的原生 full-duplex 重跑：每题一个新的 duplex 会话，官方 serving config，部署中的 8bq gate 在 talker 决定开口的那个 chunk 读一次，
触发则真 ASR 上行 + 真 gpt-5.5，等待按 1 chunk/秒真实计时，专家答案由 talker 用自己的声音<b>逐字念出</b>（TTS relay，2026-09-02 起线上默认）。五档各自实测（实心点）；数据未到齐时中间档按 never 臂的实测 onset 分数分支（空心点）。
<b>判分</b>：与旧图同一把尺子（OpenAudioBench 官方 gpt-4o / VoiceBench gpt-4o-mini / 我们的 ref-anchored judge）。绿色虚线 = 专家原文直接判分，即 relay 无损时的上界。
<b>橙色虚线（有的池才有）</b>：同一个 gate、换成旧的"prompt 引导 talker 转述"relay 的实测曲线——talker 会截断、自答、99% 需要 nudge，在 TriviaQA 上把 always 臂从 .960 压到 .728。这就是 relay 改成 TTS 直读的原因。
<b>数字</b>：never .824 → aggressive .856（19%）→ always .916；上界 .932。本地 floor 本来就高，gate 只在 19% 的题上触发，收益 3 分。
<b>和旧图不同的地方</b>：旧 harness 图上这个池是"选择性升级 > 全部升级"（.948 vs .928）的明星；原生 regime 下 always .916 更高，这个反超没有复现——原生本地 floor 低了 1.2 分、专家 relay 没有损失，两头都对 always 有利。
<b>噪声提醒</b>：conservative 档 0% 触发却是 .788，比 never 低 3.6 分——同一批题重跑一遍本地回答就有这么大的波动，读所有 5 分以内的差别都要带着这个前提。
<b>公平吗</b>：OpenAudioBench 官方判分器；这个池没有官方对比线。
"""),
    ("native_sllama_pareto", "图6 · Llama Questions（OpenAudioBench） — latency vs acc，原生 full-duplex", "win", """
<b>横轴</b>：用户说完到回答结束的 P50 时长（秒）。原生 duplex 里 talker 可能在用户说完前就开口，所以本地臂的时长可以很短；升级臂 = stall + 等待专家（真实计时）+ relay。
<b>数字</b>：never 2.8 s → aggressive 3.0 s → always 9.0 s。触发率低，中间档几乎不付延迟代价。
"""),
    ("native_sreason_dualview", "图7 · Reasoning QA（中文） — escalation vs acc，原生 full-duplex（2026-09-02）", "win", """
<b>这是什么</b>：图 3–14 那批曲线的原生 full-duplex 重跑：每题一个新的 duplex 会话，官方 serving config，部署中的 8bq gate 在 talker 决定开口的那个 chunk 读一次，
触发则真 ASR 上行 + 真 gpt-5.5，等待按 1 chunk/秒真实计时，专家答案由 talker 用自己的声音<b>逐字念出</b>（TTS relay，2026-09-02 起线上默认）。五档各自实测（实心点）；数据未到齐时中间档按 never 臂的实测 onset 分数分支（空心点）。
<b>判分</b>：与旧图同一把尺子（OpenAudioBench 官方 gpt-4o / VoiceBench gpt-4o-mini / 我们的 ref-anchored judge）。绿色虚线 = 专家原文直接判分，即 relay 无损时的上界。
<b>橙色虚线（有的池才有）</b>：同一个 gate、换成旧的"prompt 引导 talker 转述"relay 的实测曲线——talker 会截断、自答、99% 需要 nudge，在 TriviaQA 上把 always 臂从 .960 压到 .728。这就是 relay 改成 TTS 直读的原因。
<b>数字</b>：never .510 → conservative .624（20%）→ balanced .663（33%）→ aggressive .728（50%）→ always .837；上界 .861。
<b>为什么这张图存在</b>：图N2 时代这个池"gate 完全不触发"（全英文阈值压住中文分数）。8bq 的分语言阈值后三档按预期触发 20/33/50%，每档都在随机线之上。
<b>为什么不是最好的池</b>：中文推理题的本地 floor 最低、专家延迟最长（always 档 P50 27 秒，见下图），推理型失败仍是 gate 最难判的一类。
<b>公平吗</b>：我们的 ref-anchored judge（gpt-5.4-mini），与 N4 同口径；无官方对比线。
"""),
    ("native_sreason_pareto", "图8 · Reasoning QA（中文） — latency vs acc，原生 full-duplex", "win", """
<b>横轴</b>：用户说完到回答结束的 P50 时长（秒）。原生 duplex 里 talker 可能在用户说完前就开口，所以本地臂的时长可以很短；升级臂 = stall + 等待专家（真实计时）+ relay。
<b>数字</b>：never 5.1 s → balanced 6.2 s → aggressive 11.9 s → always 26.9 s。中文推理答案长、专家慢，always 档最贵。
"""),
    ("native_sdqa_dualview", "图9 · SD-QA 真人语音（VoiceBench） — escalation vs acc，原生 full-duplex（2026-09-02）", "win", """
<b>这是什么</b>：图 3–14 那批曲线的原生 full-duplex 重跑：每题一个新的 duplex 会话，官方 serving config，部署中的 8bq gate 在 talker 决定开口的那个 chunk 读一次，
触发则真 ASR 上行 + 真 gpt-5.5，等待按 1 chunk/秒真实计时，专家答案由 talker 用自己的声音<b>逐字念出</b>（TTS relay，2026-09-02 起线上默认）。五档各自实测（实心点）；数据未到齐时中间档按 never 臂的实测 onset 分数分支（空心点）。
<b>判分</b>：与旧图同一把尺子（OpenAudioBench 官方 gpt-4o / VoiceBench gpt-4o-mini / 我们的 ref-anchored judge）。绿色虚线 = 专家原文直接判分，即 relay 无损时的上界。
<b>橙色虚线（有的池才有）</b>：同一个 gate、换成旧的"prompt 引导 talker 转述"relay 的实测曲线——talker 会截断、自答、99% 需要 nudge，在 TriviaQA 上把 always 臂从 .960 压到 .728。这就是 relay 改成 TTS 直读的原因。
<b>数字</b>：never .480 → balanced .650（27%）→ aggressive <b>.825</b>（66%）→ always .870；上界 .885。真人语音上路由收益最大：aggressive 比 never 高 34 分。
<b>为什么</b>：真人录音里 ASR 噪声主导本地失败，而 gate 在 L22 读到的正是这种"没听清"的状态；专家上行用的是原始音频 + gpt-transcribe，不吃小模型的转写。
<b>公平吗</b>：我们的 judge；VoiceBench 官方没有 SD-QA 的可比离线数字，所以没画对比线。
"""),
    ("native_sdqa_pareto", "图10 · SD-QA 真人语音（VoiceBench） — latency vs acc，原生 full-duplex", "win", """
<b>横轴</b>：用户说完到回答结束的 P50 时长（秒）。原生 duplex 里 talker 可能在用户说完前就开口，所以本地臂的时长可以很短；升级臂 = stall + 等待专家（真实计时）+ relay。
<b>数字</b>：never 2.4 s → balanced 3.3 s → aggressive 12.9 s → always 18.7 s。+34 分（aggressive）多等 10.5 秒，balanced 档是性价比拐点。
"""),
    ("native_frozen_dualview", "图11 · our pool（内部 test 240） — escalation vs acc，原生 full-duplex（2026-09-02）", "win", """
<b>这是什么</b>：图 3–14 那批曲线的原生 full-duplex 重跑：每题一个新的 duplex 会话，官方 serving config，部署中的 8bq gate 在 talker 决定开口的那个 chunk 读一次，
触发则真 ASR 上行 + 真 gpt-5.5，等待按 1 chunk/秒真实计时，专家答案由 talker 用自己的声音<b>逐字念出</b>（TTS relay，2026-09-02 起线上默认）。五档各自实测（实心点）；数据未到齐时中间档按 never 臂的实测 onset 分数分支（空心点）。
<b>判分</b>：与旧图同一把尺子（OpenAudioBench 官方 gpt-4o / VoiceBench gpt-4o-mini / 我们的 ref-anchored judge）。绿色虚线 = 专家原文直接判分，即 relay 无损时的上界。
<b>橙色虚线（有的池才有）</b>：同一个 gate、换成旧的"prompt 引导 talker 转述"relay 的实测曲线——talker 会截断、自答、99% 需要 nudge，在 TriviaQA 上把 always 臂从 .960 压到 .728。这就是 relay 改成 TTS 直读的原因。
<b>数字</b>：never .408 → balanced .542（25%）→ aggressive .629（52%）→ always .704；上界 .765。
<b>怎么读</b>：这是我们自己造的内部池（对话式 speakable 子集），floor 最低，always 与上界的 6 分差是念读上限截掉的长答案。
<b>公平吗</b>：内部完全配对，但它不是公开集，只作为"六个公开池之外的一致性检查"。
"""),
    ("native_frozen_pareto", "图12 · our pool（内部 test 240） — latency vs acc，原生 full-duplex", "win", """
<b>横轴</b>：用户说完到回答结束的 P50 时长（秒）。原生 duplex 里 talker 可能在用户说完前就开口，所以本地臂的时长可以很短；升级臂 = stall + 等待专家（真实计时）+ relay。
<b>数字</b>：never 3.6 s → balanced 5.6 s → aggressive 10.9 s → always 20.8 s。
"""),
    ("native_valpaca_dualview", "图13 · VoiceBench AlpacaEval（开放式生成，judge 1–5 分） — escalation vs acc，原生 full-duplex（2026-09-02）", "loss", """
<b>这是什么</b>：图 3–14 那批曲线的原生 full-duplex 重跑：每题一个新的 duplex 会话，官方 serving config，部署中的 8bq gate 在 talker 决定开口的那个 chunk 读一次，
触发则真 ASR 上行 + 真 gpt-5.5，等待按 1 chunk/秒真实计时，专家答案由 talker 用自己的声音<b>逐字念出</b>（TTS relay，2026-09-02 起线上默认）。五档各自实测（实心点）；数据未到齐时中间档按 never 臂的实测 onset 分数分支（空心点）。
<b>判分</b>：与旧图同一把尺子（OpenAudioBench 官方 gpt-4o / VoiceBench gpt-4o-mini / 我们的 ref-anchored judge）。绿色虚线 = 专家原文直接判分，即 relay 无损时的上界。
<b>橙色虚线（有的池才有）</b>：同一个 gate、换成旧的"prompt 引导 talker 转述"relay 的实测曲线——talker 会截断、自答、99% 需要 nudge，在 TriviaQA 上把 always 臂从 .960 压到 .728。这就是 relay 改成 TTS 直读的原因。
<b>数字</b>（VoiceBench 1–5 分）：never 3.68 → balanced 3.77（5%）→ aggressive 4.01（44%）→ always 4.20；专家原文 4.96。
<b>为什么标"不利"</b>：开放式生成任务上 gate 触发率低、收益小，而且 always 与专家原文差 0.76 分——AlpacaEval 奖励完整长回答，语音场景 400 字符念读上限直接截掉了一半内容。这是方法边界 + 语音介质的双重限制，如实报告。
<b>公平吗</b>：VoiceBench 官方 gpt-4o-mini 判分器；没有官方离线对比线可画。
"""),
    ("native_valpaca_pareto", "图14 · VoiceBench AlpacaEval（开放式生成，judge 1–5 分） — latency vs acc，原生 full-duplex", "win", """
<b>横轴</b>：用户说完到回答结束的 P50 时长（秒）。原生 duplex 里 talker 可能在用户说完前就开口，所以本地臂的时长可以很短；升级臂 = stall + 等待专家（真实计时）+ relay。
<b>数字</b>：never 6.5 s → aggressive 13.8 s → always 39.4 s。开放式长答案念读时间最长，是这个池不适合语音路由的另一半原因。
"""),
    ("nvda_layer_sweep", "图15 · NVDA VoiceChat-11B — 探针层扫（第二个全双工家族）", "nvda", """
<b>看什么</b>：把我们的探针配方原样搬到 NVIDIA NemotronLabs-VoiceChat-11B（Nemotron Nano v2 9B 主干，56 层 Mamba2/attention 混合架构——和 MiniCPM 完全不同的架构家族）。
<b>数字</b>：中段 L30-34 最强（OOF AUC .714），两端弱（L2 .693 / L54 .682）。
<b>为什么重要</b>：这是论文 §9 预注册的"第二家族"测试。"中层语义带最可读"的结构在一个 Mamba 混合主干上复现了——探针读的不是 MiniCPM 的私有特征，是全双工语音模型的共性结构。
<b>公平吗</b>：校准只用了我们冻结的 600 题池（vs MiniCPM v3 的 2310），判分器同一把（gpt-5.4-mini ref-anchored）；离线重放口径，非实时 loop。
"""),
    ("nvda_transfer", "图16 · NVDA VoiceChat-11B — 冻结方法论迁移（AUC 对比 MiniCPM）", "nvda", """
<b>看什么</b>：同样三个读数（eot_last / +窗口均值 / +user-audio 均值）、同样的 C=1e-4 逻辑回归，在 NVDA 模型上从零校准，然后直接测 4 个外部公开池。
<b>数字</b>：OOF .790；striviaqa .781 / swebq .793 / sdqa .754 / sllama .701——绿线是 MiniCPM v3（.79-.81），600 条校准就摸到了同一水平带；特征叠加的增益模式也和 MiniCPM 完全一致（.714→.761→.790）。
<b>边界（如实报）</b>：sreason（中文）在此模型上 fail rate = 1.000——NVDA VoiceChat 是英文单语模型，听中文音频直接幻觉英文答案，跨语言迁移在它身上没有对应物，该池无 AUC 可算。另外它的知识 floor 明显低于 MiniCPM（striviaqa 本地正确率 .32 vs .62，同判分器）——底座弱不妨碍探针读失败信号，反而 base-fail 高信号更足。
<b>公平吗</b>：判分器与标签定义完全同口径；尚未跑实时 4 臂曲线（需要把 streaming loop 移植到 NeMo，是下一步的花钱决定）。
"""),
    ("nvda_remix", "图24 · NVDA 五池重混：MiniCPM 上 flat 的三个池在第二家族全部跑赢随机", "nvda", """
<b>看什么</b>：把 tab:transfer 的格子在 NVDA 上补满——按 NVDA 探针分数 top-r 换成实测 gpt-5.5 结果、其余保留 NVDA 本地答案的离线重混（8ad 验证过的算术），逐池用官方判分器（OAB 三池 = 官方 gpt-4o judge、SD-QA = 我们的、AlpacaEval = VoiceBench 1-5）。
<b>数字</b>：五个池全部跑赢 matched-rate random（置换检验 p≤.0003）。50% 档：striviaqa .356→.741（随机 .656）、swebq .376→.696（.602）、sllama .705→.876（.818）、sdqa .310→.675（.600）、AlpacaEval 3.55→4.46（4.23）。
<b>为什么重要</b>：论文正文的两个 honest negative（WebQ/SD-QA flat、AlpacaEval 不赢随机）在第二家族<b>都不复现</b>——选择性没变（官方标签下 AUC .72-.78），变的是 headroom（NVDA floor .31-.38 vs MiniCPM .51-.66）和失败物种（NVDA 开放题是硬失败：被选中半边 VB 3.01 vs 留下 4.08）。negative 是"池×底座"的属性，不是信号的属性。
<b>公平吗</b>：离线重混非实时 loop；expert 听的是 MiniCPM 的 heard transcript（NVDA 离线没有自己的 ASR relay），A 口径不含转述税——含实测转述结果的 B 口径档位点差 ≤.032。striviaqa 3 题、sllama 16 题因官方 judge 输出不可解析剔除。
"""),
    ("nvda_probe_reads", "图25 · NVDA 读点体检：commit 后 8 帧读的是回答开头不是“偷看”，因果读只亏 .03 AUC / 1 个准确率点（PR #7，2026-09-02）", "nvda", """
<b>这是什么</b>：合作者对 NVDA 部署探针（v2：校准 600→2,481 行、读点从"用户音频结束"改到模型自己的 commit-to-speak 后 8 帧）做的第三次独立 replay 复盘。回答审稿人必问的一句：读点落在模型开口后约 640 ms，这算不算已经看到了答案？
<b>左图</b>：六种读法的冷外部四池均值 AUC（点=单池，Δ=与部署读点的配对 bootstrap 差）。<b>严格因果</b>读（只读 commit 帧之前：commit 帧 ‖ 前 8 帧均值 ‖ 到 commit 为止的运行均值）比部署读点低 .025–.030，每一层都一致，置信区间排除 0；而 commit 之前的窗口并不比老的"音频结束"窗口好（.781 vs .782）——所以多出来的信号就是模型开口的前几个 token 本身，不是 commit 前听到了什么。
<b>右图</b>：换成固定升级预算下的准确率（官方判分器，四池均值），部署 probe 与 v2 probe 完全重合，都在 15/30/50% 三档显著跑赢同率随机（.552/.650/.756 vs .507/.577/.670）；严格因果读只低 0.5–1 个点（.546/.640/.749）。
<b>为什么重要</b>：(1) 部署读点不是 pre-answer，但把"偷看"去掉代价很小，论文可以两种口径都报；(2) 两位独立 reviewer 提的 ~85 个配方变体（多层平均、LDA、PLS、重加权、逐帧特征……）外部 AUC 最多 +.013，且<b>没有一个</b>改变固定预算准确率——线性 probe 在这组特征上到顶了，以后不必再在配方上花时间；(3) 在线状态机的运行均值（不知道音频何时结束）和离线全句均值可互换（−.004 [−.009, +.001]），streaming 移植不用重新校准。
<b>没修好的</b>：固定校准阈值在 Llama-Q 只触发 1/4/12%、SD-QA 触发 7/27/53%（名义 15/30/50）——和 MiniCPM 上见过四次的问题一样，NVDA 侧的 per-pool 分位数阈值还没移植，是下一个 PR。
<b>公平吗</b>：校准、外部池、判分器与图16/图24 同口径；表内 AUC 来自 2,481 行分析拟合，导出的部署/v2/因果 artifact 是剔除 240 行冻结 test 后的 2,258 行独立重拟合，两者不混用。离线重放，非实时 loop。
"""),
    ("kink_case_study", "图21 · 案例走查:一道题的两个世界(拐弯怎么出现)", "aux", """
<b>用途</b>:用错题簿里的一道典型题(sllama0164「锡克教有几位祖师」)的<b>完整实测 trace</b> 讲拐弯:世界A(探针关)本地绕 487 字符、3.10 秒、答错;世界B(探针开)话音落下 21ms 读出 eot=0.631≥0.513 → 升级,gpt-5.5 1.68s + 转述 0.62s = 2.32 秒,答对——<b>同一道题快 0.78 秒且从错变对</b>。下半用三根条讲池效应:38 道这种慢题(P50 2.38s)离开本地队列后,留下 212 题 P50 掉到 0.94s,整臂中位 1.52→1.17s = 图8 的左折。含 Addendum 4 的边界修正(探针挑"会错"不挑"会绕")。
"""),
    ("entropy_traj", "图22 · token 级机制:熵轨迹 + 停止意愿(hedging 的显微镜)", "aux", """
<b>看什么</b>:93 道 striviaqa 题按四种行为分组重放,逐 token 记录全词表熵和终止符概率。<b>左</b>:答对(蓝)全程低熵;hedged 错(红)开头 ×1.8 高熵、全程游走;自信错(黄)又短又低熵——<b>它的输出分布也被错误事实骗了</b>,这就是熵信号(AUC .70)到不了探针(.80)的微观原因。<b>右</b>:句号后一步终止符的概率,答对题 .083,hedged 错 <b>.0024(低 35 倍)</b>——"不肯停"在 token 层面直接可见。彩蛋:绕对组开头低熵但游走最久——早期熵=检索状态,后期熵=文风,两者在此分离。工程脚注:MiniCPM 真终止符 id=151704 不在 generation_config 里。
"""),
    ("nvda_fold_test", "图23 · 预测验证:极简风格的模型没有拐弯(NVDA vs MiniCPM)", "nvda", """
<b>看什么</b>:8ab 预注册的预测——"回答风格极简的模型不会有延迟左折"——现在是观测。同一套 top-r 重混算术:MiniCPM(蓝)在 sllama 上折出 −0.05s;NVDA(绿)两个池全程<b>严格单调</b>。口径:NVDA 是语音原生双工,部署延迟由帧钟决定(1 token = 80ms),本地延迟 = 回答 token 数 × 0.08s,免疫批量计时污染;专家路径 = 同题实测 gpt-5.5 RTT。机制:NVDA 本地 P90 仅 2.0s,连最慢的本地回答都快过专家(~4s),升级永远净加时。<b>这也反向确认了拐弯的机制论:拐弯需要"会错的题恰好慢",把慢(绕)拿掉,拐弯就消失。</b>
"""),

]

VERDICT = {"win": ("✓ 有利", "#1e9e50"), "mixed": ("~ 有保留", "#b8860b"),
           "pending": ("… 跑批中", "#5a6270"), "aux": ("机制分析（MiniCPM，离线重放）", "#6b4e9e"),
           "nvda": ("第二家族 · NVDA VoiceChat-11B 原生 duplex — 合作者在该模型上自训探针（离线重放）", "#0f6e8c"),
           "loss": ("✗ 不利（如实报告）", "#b00")}

app = modal.App("figures-gallery")
HERE = os.path.dirname(os.path.abspath(__file__))
image = modal.Image.debian_slim().pip_install("fastapi[standard]")
for name, _, _, _ in FIGS:
    _p = os.path.join(HERE, "figures", f"{name}.png")
    if os.path.exists(_p):
        image = image.add_local_file(_p, f"/root/figs/{name}.png")


@app.function(image=image, timeout=60 * 5, min_containers=0)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response

    api = FastAPI()
    blocks = []
    for name, title, verdict, text in FIGS:
        label, color = VERDICT[verdict]
        img = (f'<img src="/{TOKEN}/f/{name}.png" loading=lazy>'
               if os.path.exists(f"/root/figs/{name}.png") else
               '<div class=pend>图还在跑批中 — 数据到齐后自动补上</div>')
        blocks.append(
            f'<div class=fig><h3>{title}</h3>'
            f'<span class=badge style="background:{color}">{label}</span>'
            + img +
            f'<div class=interp>{text}</div></div>')
    page = f"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Figures + interpretation</title><style>
body{{font-family:-apple-system,system-ui,sans-serif;max-width:900px;
margin:0 auto;padding:1rem;background:#fafafa;color:#111;line-height:1.55}}
.fig{{background:#fff;border:1px solid #ddd;border-radius:10px;
padding:.8rem;margin:1.2rem 0}}
h1{{font-size:1.2rem}}h3{{font-size:.98rem;margin:.2rem 0 .4rem}}
img{{width:100%;height:auto;border-radius:4px;margin:.5rem 0}}
.badge{{color:#fff;font-size:.72rem;padding:.12rem .5rem;border-radius:99px}}
.interp{{font-size:.86rem;background:#fbfbfa;border-left:3px solid #ccc;
padding:.6rem .7rem;border-radius:0 6px 6px 0}}
.interp b{{color:#000}}
.pend{{background:#eef1f4;color:#5a6270;padding:2rem;text-align:center;border-radius:4px;margin:.5rem 0;font-size:.9rem}}
.note{{background:#fff;border-left:4px solid #333;padding:.7rem;
font-size:.86rem}}</style></head><body>
<h1>原生 full-duplex 结果集 + 逐图解读</h1>
<div class=note><b>2026-09-02 完成</b>：整套结果图已全部换成<b>原生 full-duplex 实测</b>——七个池 × 五档，每题一个真实 duplex 会话、官方 serving config、
线上 8bq gate、真 ASR 上行 + 真 gpt-5.5 + TTS relay，五档各自实测（不再有分支估计）。此前 turn-based/harness loop 的图 1–14、17–20 已下架，只在图N6 保留 floor 对照。<br><br>
<b>relay 已换</b>：旧 relay 让 talker 自己转述专家答案，TriviaQA always 臂专家原文 .960、用户听到只剩 .728（丢掉 27% 的正确答案）。
2026-09-02 起 relay 改为 talker 用自己的声音逐字念专家文本（TTS 直读），同题 A/B：.733 → .933；线上 demo 已同步切换。旧 relay 的曲线在 TriviaQA / WebQ 以橙色虚线保留作对照。<br><br>
<b>一句话总结</b>：原生 regime 下本地 floor 比 harness 低 1–8 分，但路由收益更大——六个 QA 池 aggressive 档比 never 高 3 到 34 分，真人语音（SD-QA）收益最大；
开放式生成（AlpacaEval）仍是边界，且语音念读上限又砍掉一半专家内容。<br><br>
<b>怎么读这套图</b>：每张图下面写三件事——<b>为什么占优</b>、<b>为什么吃亏</b>、<b>这个比较公不公平</b>。
外部集用<b>官方判分器</b>（OpenAudioBench 的 gpt-4o / VoiceBench 的 gpt-4o-mini，逐字复制官方 prompt）；对比模型的数字来自官方表，全部是<b>离线 chat 模式</b>，我们的曲线是<b>实时流式</b>的。
本地回答是随机解码（top_k=20），同一批题重跑一遍 floor 会动 ±3 分，5 分以内的差别请带着这个前提读。</div>
{''.join(blocks)}</body></html>"""

    @api.get(f"/{TOKEN}", response_class=HTMLResponse)
    def index():
        return page

    @api.get(f"/{TOKEN}/f/{{name}}.png")
    def fig(name: str):
        path = f"/root/figs/{name}.png"
        if not os.path.exists(path) or "/" in name:
            raise HTTPException(404)
        return Response(open(path, "rb").read(), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=600"})

    return api
