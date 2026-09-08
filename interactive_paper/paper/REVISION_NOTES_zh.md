# 论文修订说明（2026-09-08）

这版是在删除 human-eval 后的版本上做的文字与图表修订，未启动新的 GPU / API 实验。原始内部测试集、实验回答与评测标签均未改动；图 3 左上角采用下述明确标注的事后重配，Table 1 仍保留原分布。

## 2026-09-08：已确认的图 3 样式

- 保留 3×2 布局；采用 Times 衬线字体、细线、空心标记和共享图例，移除增幅标语。图内脚注保留事后重配说明。
- Internal 准确率和调用率使用全部 240 题：knowledge、math 每题权重各为 0.25，其余题型为 1，再归一化。各 arm 和 random-mixture reference 使用同一套权重；不删除单题或修改其结果。这是检查测试结果后选定的分布敏感性视图，不是新的独立测试或模型改进。
- Internal 计时仍用原分布，另外两个 pool 不变；图内明确说明该差别。Table 1、正文数值和原图注依作者要求保留，因此图 3 左上角不应当作 Table 1 的同分布复现。
- 原 teaser 与 Table 1 整块保留；`main.pdf` 是作者确认的本地编译版本。绘图输入、所选权重和逐条曲线数据一并存入 `revision_data/`。

## 2026-09-08：3ade619 交付后的更新

- 更正主文的 expert/relay 5.5 pp 差距：其中包含内容变化和 judge 波动，不能全归因于 formatter。
- 附录加入后续诊断：已完成的历史 r1/r2 与主表 r0 区分；不要求任何新 repeats，不更换 Table 1 的原数值。
- 从已交付的 val1 逐题文件核验 299 题、排除 validation 后 4,922 个 fit ID 的哈希、实际 gate SHA，以及七档 accuracy remix；补入文本/实际输出音频转写分数。它们属于 v2 validation，不混进原 internal 主结果。
- 收紧 validation 计时结论：scripts/53 仍把 chunk 数与时长项相加，不能因保存了真实音频就把该 sweep 当作实测 audio-ready latency。
- 写入 v2/v3 离线 formatter 的可复算收益和负结果：空输出 1→0、internal 语法标记 22→8；SD-QA 2→3。明确 token 保留率不是 accuracy 或 TTS 可懂度。
- 更新 P0 可复现性状态：所缺脚本、输入 manifest 和 clean-checkout 记录现已交付；区分该记录与 Codex 本地独立完成的检查。
- 原 Table 1、原 teaser 继续整块保持；human-eval 仍删除，BayLing-Duplex related-work 引用仍保留。
- 给 Claude 的补充单与此前缺失的本地 JSON 已放到工作区 `output/claude_handoff/`。trunc1 新结果的逐题来源尚需补齐；没有把摘要中的局部提升投影到全 internal。

## 已完成

- 保持删除 human-eval section；重写摘要、Introduction、Related Work、方法、Setup、Results 和 Discussion。
- 主文采用同一组 native benchmark 数据，清理旧 turn-based 数字混入新版结论的问题。历史实验保留在附录，并说明各自适用的运行方式。
- 按作者要求完整恢复原 Table 1（原布局、数值、表注，以及 MiniCPM / NVDA 两个区块）。正文按原表解释五个/四个外部 pool 的 avg、Delta、各模型的实验协议；逐 pool 实际升级率另列在表注所指的附录。主表不由绘图脚本改写。
- Figure 3 改成 3×2：Internal、Speech TriviaQA、SD-QA；左侧准确率，右侧 mean / P50 / P95 耗时。always 参考线通过真实 always 点。图来自现有逐题数据，均值不是多次重跑均值。
- 按作者要求恢复并保留原 Figure 1 / teaser_v2 原图和原图注；后续图表生成脚本不重画 teaser。
- 区分文本内容准确率、音频实际播放质量，以及日志重建耗时。现有实验并未统一测量两条路径的可听完成时间。
- 明确披露已知 calibration overlap、去重候选未部署、native benchmark 向 expert 传入预载入音频等证据边界。
- 为新图、附录升级率表与指标汇总增加离线生成脚本，记录输入 SHA-256、样本数和指标定义，便于核对；原主表保持不变。

## P0 数据已落实到论文（dd33f7c 审阅后）

- Setup 与附录写清 4,986 core + 242 fresh 的 native fit；阈值只取 core 的五折 OOF 分位点。补入已核验的两版 OOF AUC 和 12 个阈值匹配结果，保留“候选尚未用于主实验”的事实。
- 将 expert “收到完整问题”修正为“收到完整预载入 wav 的尾 30 秒”，同时说明 early-onset 看到未来输入、长问题丢失开头这两种情况。
- 主文结果新增分题型解读，附录加入全体 240 题的诊断表。43/60 道知识题超过输入窗口，其中 always 实际调用 expert 的 42 道被截去开头、1 道未触发请求；未把截断表述为已证实的错误原因或预测修复收益。所有题目及权重保持不变。
- 摘要、方法与 Figure 3 表注统一使用 reconstructed timing diagnostic，避免把本地文本计时和 relay 音频合成/时长之和称为统一 audio-ready 或 completion latency。
- 附录更新 artifact 可用状态，明确旧 sweep 与当前 OOF 阈值并不相同，完整脚本复现仍需补正；未将 P1 写成已完成。

## 值得注意的数字（主表保持原值）

Internal aggressive：40.8% → 62.9%，实际升级率 51.7%，恢复 74.6% 的 local→always 准确率差距。always 实际升级 238/240；原表采用 always-policy endpoint 的 random mixture，aggressive reference 为 56.1%、优势 6.8 pp。正文保留原表分布；Figure 3 使用同一 random-mixture 公式，但左上角换为上述事后加权分布。按实际 always 调用率规范化后的原分布结果为 56.2% / 6.7 pp，单独保留为附录敏感性分析。五个外部 QA pool 的 aggressive macro accuracy 仍为 59.4% → 80.9%，random reference 75.2%。

Internal aggressive 的重建耗时 mean 27.3 s、P95 103.2 s，不能包装成已经解决 real-time latency。原日志 40% 档的 19.6 s 是 cached remix；只能同口径与 remix 50% 的 23.3 s 比较，约下降 15.9%，不能直接同另一轮 native 的 27.3 s 比。

## 下一轮实验建议（以最新补充单为准）

1. 先 CPU 修正文档中的 v1/v2 对账和 latency 统计，补齐 val1、formatter 与 trunc1 的来源。已完成的两路音频、因果输入实现和 validation 不重复列为未开始。
2. 优先复用 validation endpoint；在 validation 文本上评估 v2/v3。structured expert 与 web baseline 的工具能力不同，须控制或明确为整套配置比较。候选未带来收益时不扩大实验。
3. 每个 query/arm 正式运行一次，不新增 repeats。Internal 五个 arms 为 1,200 sessions，不含补充 smoke/validation；已有 r1/r2 仅作历史诊断。新正式结果必须与原表协议区分。
4. 扩展 MiniCPM 已测 benchmark 前先检查开发集重叠；Speech CMMLU 已参与过本项目开发实验，不能直接当作全新 untouched test。

## 复现本版图表

仅复现当前已确认的图 3，在 paper 目录下使用已提交的 JSON 汇总：

```bash
python build_academic_revision_figure.py
tectonic main.tex
```

若需从原始 parquet 重建原分布汇总和附录表格，最后再运行当前图 3 的绘图脚本，以免被旧样式覆盖：

```bash
python build_revision_figures.py --data-dir ../data/native_bench
python build_internal_diagnostic.py --data-dir ../data/native_bench --queries ../data/queries.jsonl
python build_academic_revision_figure.py
tectonic main.tex
```

图表脚本只读现有数据，不调用模型。当前图 3 依赖 numpy、matplotlib；从 parquet 重建另需 pandas、pyarrow。`revision_data/native_summary.json` 记录来源 commit、逐文件哈希与计算结果；`joint_reweighting_ideas.json` 保留候选权重，`academic_figure_values.json` 保留当前所选曲线的数值。待重跑的实验没有写成已完成结果。
