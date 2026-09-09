# MiniMCP 引用核验与修订记录

核验日期：2026-09-08。对象：当前字体修复版 `main.pdf` 及与其对应的 LaTeX/BibTeX，Git 基线 `5a5dbc371671ec39714f6457cd96d8028805de5f`。PDF 共 42 页；Overleaf 因引擎排版产生的页码可能不同，下文同时提供源文件行号。

**结论：实际引用的 45 项均有真实的一手来源，没有发现虚构文献；但不能判定为“全部引用准确、全部数据已验证”。最需要改的是附录三处超出原文证据的概括。**

范围：从新编译的 `.aux` 取得实际引用集合，再核对正文/附录所有引用上下文。`refs.bib` 共 90 项，另 45 项没有进入这份 PDF，未把它们算作已核验引用。取得 41 份文献 PDF，另四项读取期刊页面或作者官方仓库。题名、作者表、年份及已标明的出版信息与一手页面对照；预印本年份与会议年份不同不自动判错。Lin et al. 的 TMLR 详情页被 OpenReview 验证页面阻挡，其内容、作者与 arXiv 年份已核实，但直接出版核验留有一项限制。

## 修订状态

本次源码已应用 C1–C3 和 B1–B3：限制 Cheang 的检测结论、明确 David 最早读点为 4 个 token 且未测试生成前对照、将 Cartography/AUM 的机制归因降调；修正 RouterBench 题名和 LLM Router 姓名字段；Freeze-Omni 固定为 v5 并使用该版 8 人作者表。

同时明确 RouterBench 的 36,497 为本实验过滤后的子集，FDB 数字是本论文单次运行所得、比较对象限于原论文所列历史基线且 ASR 配置不同；为官方 AlpacaEval 4.81 补充 Table 4 引用。实验结果未因引用修订而重新估计或替换。

下面保留**修订前版本**的逐条核验记录，位置和“需修改”等状态均指原审计基线。上述修订已经完成；D1/D2 原始记录的复算限制仍存在。

## 优先处理的引用表述

### C1：不能把检测效果下降写成“无法检测”

位置：`sections/appendix.tex:1913`，PDF 第 40 页，Cheang et al. (2025)。

当前文字说关联驱动的自信错误是 `undetectable`。原文 §5 的 **AH-only 白盒探针 AUROC 约 0.58–0.69**，相对 UH-only 的 0.91–0.93 明显降低，但不是所有探针都没有信号；原文 Limitations 还明确限制到事实补全任务。应表述为“关联型幻觉与正确事实的表示重叠，使所测试探针的检测效果显著下降”，不要泛化为所有内部状态探针均无法发现这类错误。[原文](https://arxiv.org/abs/2510.09033)

### C2：没有生成前对照，不能得出“只有生成后才可读出”

位置：`sections/appendix.tex:1916`，PDF 第 40 页，David (2025)。

原文 §3.3 使用前缀长度 **t ∈ {4,8,16,32,64,128,192,256,384,512}**，最早为 4 个 token，没有 t=0 对照。它支持“早期生成状态已含可预测最终正确性的信号”，不支持正文的 `only as generation proceeds` 排他性命题，更不能直接证明语音模型回答前不存在可读信号。建议删掉 only，并说明是数学推理任务上的实验发现。[原文](https://arxiv.org/abs/2511.14773)

### C3：训练动态文献不能直接证明当前冻结表示中的机制

位置：`sections/appendix.tex:1909`，PDF 第 39–40 页，Swayamdipta et al. (2020)、Pleiss et al. (2020)。

“上调难例权重无益，所以是特征不可区分的正例而非代表性不足的正例”是本论文的机制推断。Dataset Cartography 研究训练置信度及其变化，AUM 研究错标样本识别；二者没有验证本论文冻结语音特征的线性可分性。原段前文已限定为 timestamp-limited predictability、并否认原则上的不可约性；这些限定值得保留，但后续机制归因仍应降调。Cartography 中的 ambiguous 样本还能改善域外泛化，不能把所有难/模糊样本统一解释为有害。应将结论限定为“在所测试的特征、权重和正则化设置下未获收益”，并将机制解释写成假设。[Dataset Cartography](https://aclanthology.org/2020.emnlp-main.746/)、[AUM](https://proceedings.neurips.cc/paper_files/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html)

## 书目信息与版本

- **B1 — RouterBench 题名**（`refs.bib:102`）：当前 `Routing Systems` 应为 `Routing System`，arXiv v1/v2 都是单数。[原始条目](https://arxiv.org/abs/2403.12031)
- **B2 — LLM Router 姓名字段**（`refs.bib:664`）：第四作者 arXiv 元数据为 `Krishnan, Gomathy Venkata`；当前 `Venkata Krishnan, Gomathy` 的姓/名边界不一致。建议按官方元数据统一。这是字段规范问题，不是作者身份虚构。[原始条目](https://arxiv.org/abs/2603.20895)
- **B3 — Freeze-Omni 版本**（`refs.bib:169`）：当前 7 人作者表与 [v1](https://arxiv.org/abs/2411.00774v1) 完全一致；[v5](https://arxiv.org/abs/2411.00774v5) 增加了第四作者 Yunhang Shen。不能直接把旧版表判作漏署名。建议明确固定 v1，或将条目更新到 v5 的 8 人表。
- **可改善可追溯性**：Dolly、Stanford Alpaca 目前没有直接 URL；中文 Alpaca 只有 note 中的仓库名。建议补发布者 URL、派生数据集 URL 和实际版本。预印本更新成会议条目属于可选规范化，不等于当前预印本引用虚假。
- **不应误报的姓名差异**：SimpleQA 原始 PDF 的第二作者确写作 Nguyen Karina；Lightman 预印本作者使用 Yura Burda、Harri Edwards；AUM 正式 NeurIPS 页面使用 Ethan Elenberg。未根据其他数据库的变体强行改名。

## 数值、量表与数据来源

| 检查项 | 结果与边界 |
|---|---|
| BayLing 的四个状态 token、GLM-4-Voice、时序 DPO | §2、Fig.1 与正文全部对应。 |
| Full-Duplex-Bench 727 样本 | 原文 Table II：216 + 119 + 55 + 200 + 137 = 727，正确。 |
| VoiceBench AlpacaEval 的 1–5 分 | VoiceBench §3 的明确量表，正确；不要换成通用 AlpacaEval 胜率。 |
| MiniCPM 9B、Qwen3-8B backbone | MiniCPM-o 4.5 原文确认。 |
| 附录的官方 AlpacaEval 约 4.8 | 原文 Table 4 是 4.81，按一位小数写 4.8 合理；建议直接标 Table 4。 |
| MATH-500 来源 | Lightman Appendix C 明确从 MATH test 留出 500 题，与引用关系一致。 |
| 内部 600 题、360/240 split | 已读取实际 `queries.jsonl` 核对来源、数量、split，见机器可读复核记录。 |
| 主表 MiniCPM 上半部 | 已从 Git 读取 35 份逐题 parquet，对文件 SHA、各臂 ID 集合、计分样本数、正确率、升级率、专家文本分数复算；再对主表七行及均值/增量核对，全部一致（35/35 哈希一致、35/35 ID 检查通过、数值差异 0）。 |
| 主表 NVDA 下半部 | 外部四列与 `remix_eval3.json` 的三层 v2 分支及 AlpacaEval 汇总对照；六行及均值/增量全部一致。Internal 列使用已有 pass-3 replay 的同次回答重新调用 gpt-5.4-mini 判分，并用排除全部 test ID 的 2,258-row probe fit 复算；240 题全部进入 accuracy 分母，17 题无 answer-onset read、保持 local。此次没有重新做 GPU inference。 |
| FreshQA fast-/never-changing | 原文分类存在；本论文 a priori 升级标签是额外策略标签，不能当成原数据的模型失败标注。 |

### D1：FDB 的模型成绩有运行报告，但尚未逐样本复算

PDF 第 28 页的 0.125/0.117、0.915、约 0.90s，均与 Git 中 `fdb/RESULTS.md` 的本论文 MiniCPM 实测报告对应（时延原值 0.904s）。它们不来自 2025 年 FDB 原论文。报告说明原始输出与 eval logs 在 Modal `fdb-data` volume，当前 Git 只包含报告和推理代码。报告也注明 ASR 版本与原论文历史实验可能不同。因此应将 best-in-class 限定为“相对于所列原论文基线”，并保留单次运行/配置可比性的限制。[FDB 原文](https://arxiv.org/abs/2503.04721)

[本仓库 FDB 运行报告](https://github.com/dyyfk/minimcp_test/blob/5a5dbc371671ec39714f6457cd96d8028805de5f/fdb/RESULTS.md)。

### D2：RouterBench 的 36,497 与 AUC 尚未由原始输入重算

PDF 第 21 页的 36,497、AUC 0.710/0.440 等来自本论文 `RESULTS.md` 中的路由实验，不是 RouterBench 原文表格。代码 `modal_stream.py:973` 读取官方 `routerbench_0shot.pkl` 后，取 prompt、eval_name、Mixtral、GPT-4 对应列并 `dropna()`；这解释了“筛选后问题数”与原文 405,467 次模型推断记录采用不同口径。当前 Git 未包含运行输出 `router_bench.json` 或此次实验的冻结输入副本/哈希，故这些精确结果不能仅靠引用和报告判定已复现。需要补充运行 artifact、输入版本、过滤前后计数和 out-of-fold 预测。[RouterBench](https://arxiv.org/abs/2403.12031)、[官方数据发布](https://huggingface.co/datasets/withmartian/routerbench)

**数据验证边界：**缓存标签的均值复算能够确认“论文数字与保存的结果一致”，不能证明模型输出一定被正确判分，也不能代替重新运行。所有引用相关的外部量表/计数已检查；没有把附录全部实验、所有置信区间/p 值都宣称为独立复现。

## 45 项逐条记录

“通过”表示所核对的书目身份及论文对该文献的概括没有发现不一致，不表示为其全部科学结论提供保证。

| 引用键 / 原始来源 | 当前书目年份 | 核验结果 | 对应检查 |
|---|---:|---|---|
| [aggarwal2024automix](https://arxiv.org/abs/2310.12963) | 2024 | 通过 | 原文用小模型草稿的自验证结果决定是否升级到大模型；NeurIPS 2024 与 arXiv 2023 是不同版本日期。 |
| [azaria2023internal](https://arxiv.org/abs/2304.13734) | 2023 | 通过 | 隐藏层分类器估计陈述真实性；并不保证任意模型或域都适用。 |
| [bayling2026duplex](https://arxiv.org/abs/2606.14528) | 2026 | 通过 | §2/Fig.1 确认三路交织和四个状态 token；§2.2 确认 GLM-4-Voice、SFT 和针对时序的 DPO。 |
| [burns2023discovering](https://arxiv.org/abs/2212.03827) | 2023 | 通过 | 无监督 CCS 从隐藏激活恢复真假知识，支持隐藏表示可读出知识；ICLR 2023。 |
| [cheang2025recall](https://arxiv.org/abs/2510.09033) | 2025 | 需修改概括 | 附录把关联型幻觉写成 undetectable，超出 §5 的检测效果下降及特定实验范围，见 C1。 |
| [chen2023frugalgpt](https://arxiv.org/abs/2305.05176) | 2023 | 通过 | 学习模型级联以平衡准确率和成本；正文未转引其 98% 节省等数值。 |
| [chen2024voicebench](https://arxiv.org/abs/2410.17196) | 2024 | 通过 | §2/§3 确认 SD-QA 使用真实语音，以及 AlpacaEval 使用 1–5 分；不应混同 AlpacaEval 原始胜率。 |
| [chien2026moshirag](https://arxiv.org/abs/2604.12928) | 2026 | 通过 | 异步检索与全双工交互的概括符合原文；arXiv 作者备注确认 ICML 2026 接收。 |
| [chrono2026thinking](https://arxiv.org/abs/2510.05150) | 2026 | 通过 | 因果地边听边推理，符合正文概括；arXiv 始于 2025，作者备注确认 SIGDIAL 2026 接收。 |
| [cobbe2021gsm8k](https://arxiv.org/abs/2110.14168) | 2021 | 通过 | GSM8K 来源正确；本论文抽样 100 条，不是将原始数据集总量写成 100。 |
| [conover2023dolly](https://github.com/databrickslabs/dolly) | 2023 | 通过，建议补链接 | Databricks 官方仓库的推荐 BibTeX 与作者、标题、年份一致；本地 manifest 确认使用 Dolly 75 条。 |
| [david2025temporal](https://arxiv.org/abs/2511.14773) | 2025 | 需修改概括 | 研究最早读取 t=4；没有 t=0 的消融，不能支持只有开始生成后才可读出，见 C2。 |
| [defossez2024moshi](https://arxiv.org/abs/2410.00037) | 2024 | 通过 | 并行用户/助手语音流和 Inner Monologue 支持同时听说的概括。 |
| [ding2024hybrid](https://arxiv.org/abs/2404.14618) | 2024 | 通过 | 基于难度和质量约束在小/大模型间路由；作者备注确认 ICLR 2024。 |
| [faisal2021sdqa](https://arxiv.org/abs/2109.12072) | 2021 | 通过 | SD-QA 为真实方言语音 QA；原始数据集与 VoiceBench 的选用版本关系正确。 |
| [farquhar2024detecting](https://www.nature.com/articles/s41586-024-07421-0) | 2024 | 通过 | Nature 630, 625–630 (2024)，作者及 DOI 一致；采样答案的语义不确定性概括正确。 |
| [hendrycks2021math](https://arxiv.org/abs/2103.03874) | 2021 | 通过 | MATH 来源、NeurIPS 2021 Datasets and Benchmarks 正确；MATH-500 的子集来源另有 Lightman 引用。 |
| [hu2024routerbench](https://arxiv.org/abs/2403.12031) | 2024 | 标题需改；数值追溯未闭合 | 正式标题是 Routing System（单数）；36,497 是本论文筛选后的子集，不能由原论文 405,467 次推断直接推出，见 B1/D2。 |
| [huang2026duplexomni](https://arxiv.org/abs/2606.09186) | 2026 | 通过 | 交互层与可插拔思考层异步协作，支持正文对语音与推理/工具使用的概括。 |
| [joshi2017triviaqa](https://aclanthology.org/P17-1147/) | 2017 | 通过 | ACL 2017 原文、作者和数据集身份正确；当前 manifest 确认抽取 100 条。 |
| [kadavath2022language](https://arxiv.org/abs/2207.05221) | 2022 | 通过 | 原文区分回答后的 P(True) 与回答前 P(IK)，支持正文自评能力概括。 |
| [kossen2024semantic](https://arxiv.org/abs/2406.15927) | 2024 | 通过 | SEPs 用一次生成的隐藏状态近似 semantic entropy；正文没有把它说成无需任何生成即可运行。 |
| [kuhn2023semantic](https://arxiv.org/abs/2302.09664) | 2023 | 通过 | 按语义等价合并采样答案来量化不确定性；ICLR 2023 与 arXiv 年份一致。 |
| [lightman2024lets](https://arxiv.org/abs/2305.20050) | 2024 | 通过 | 原文 Appendix C 明确 500 个 held-out MATH test 问题；ICLR 官网确认 2024 接收。Yura/Harri 与其他目录的名字写法差异不判错。 |
| [lin2022teaching](https://arxiv.org/abs/2205.14334) | 2022 | 内容通过；出版字段留一项核验限制 | arXiv 原文证实内容、作者和 2022；TMLR 官方 OpenReview 详情页受验证页面阻挡，未完成其出版字段的直接核验。 |
| [lin2025fdb](https://arxiv.org/abs/2503.04721) | 2025 | 样本数通过；模型成绩属本论文测量 | Table II 的 216+119+55+200+137=727；MiniCPM 0.125/0.117/0.915/0.904 来自本仓库 FDB 运行报告，不是 2025 原论文成绩。 |
| [lugoloobi2026failures](https://arxiv.org/abs/2602.09924) | 2026 | 通过 | 冻结模型的生成前激活线性探针预测数学/代码成功，并用于模型路由。 |
| [mahaut2024factual](https://arxiv.org/abs/2406.13415) | 2024 | 通过 | §4.2.1 使用第 24 层，§5.1 显示探针优于输出概率；因此中间层可包含输出未体现的信息有原文依据。 |
| [marks2024geometry](https://arxiv.org/abs/2310.06824) | 2024 | 通过 | 真假事实的线性表示、迁移和干预实验支持相关工作概括；COLM 2024 已在作者备注确认。 |
| [ong2024routellm](https://arxiv.org/abs/2406.18665) | 2024 | 通过 | 偏好数据训练的路由器真实存在；本论文的 TF-IDF 基线及 0.523/0.533 是本论文实验，不能当作 RouteLLM 原论文结果。 |
| [openaudiobench2025](https://arxiv.org/abs/2502.17239) | 2025 | 通过 | Baichuan-Audio §5.5.1 引入 OpenAudioBench，列出 Reasoning QA、Llama/Web Questions、TriviaQA；数据来源对应。 |
| [openbmb2026minicpmo](https://arxiv.org/abs/2604.27393) | 2026 | 通过 | 原文确认 9B/Qwen3-8B backbone、全双工；Table 4 的 AlpacaEval 4.81 与附录约写 4.8 一致。 |
| [orgad2024llms](https://arxiv.org/abs/2410.02707) | 2025 | 通过 | §5 及摘要的内部知识/外部输出不一致支持正文；正式 PDF 标注 ICLR 2025。 |
| [pleiss2020aum](https://arxiv.org/abs/2001.10528) | 2020 | 文献正确；应用解释需降调 | 原文用训练动态识别错标/模糊样本，不证明当前冻结语音特征中正例不可区分，见 C3。 |
| [stivers2009universals](https://pmc.ncbi.nlm.nih.gov/articles/PMC2705608/) | 2009 | 通过 | PNAS 年卷期页与作者顺序一致；跨语言对话倾向减少间隙和重叠支持响应时限动机。 |
| [swayamdipta2020cartography](https://arxiv.org/abs/2009.10795) | 2020 | 文献正确；应用解释需降调 | 原文区分 easy/ambiguous/hard-to-learn；模糊样本甚至有助域外泛化，不能直接推出本论文的机制诊断，见 C3。 |
| [taori2023alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 2023 | 通过，建议补直接数据链接 | Stanford 官方仓库推荐条目一致；当前 manifest 确认中文样本来自 shibing624/alpaca-zh 75 条，应保留其派生数据集来源。 |
| [varshney2026llmrouter](https://arxiv.org/abs/2603.20895) | 2026 | 姓名字段需统一 | 论文存在且 prefill 路由概括正确；第四作者官方元数据为 Krishnan, Gomathy Venkata，当前姓/名边界不同，见 B2。 |
| [veluri2024syncllm](https://arxiv.org/abs/2409.15594) | 2024 | 通过 | SyncLLM 将时间信息引入生成，同步处理全双工语音；EMNLP 2024 与作者备注一致。 |
| [vu2023freshllms](https://arxiv.org/abs/2310.03214) | 2023 | 类别通过；结果属本论文实验 | FreshQA 确有 fast-/never-changing 类别；直接将 fast-changing 标为 escalate=1 是本论文新增政策标签，不是 FreshQA 的模型失败标签。 |
| [wang2024freezeomni](https://arxiv.org/abs/2411.00774) | 2024 | 版本需标清，不是虚构 | v1 确为当前 BibTeX 的 7 位作者；v5 增加 Yunhang Shen。应固定 v1 或更新到 v5 的作者表，见 B3。 |
| [wang2024fsm](https://arxiv.org/abs/2405.19487) | 2024 | 通过 | 神经 FSM 和控制 token 支持相关工作概括；作者备注确认 NeurIPS 2024。 |
| [wang2024mmlupro](https://arxiv.org/abs/2406.01574) | 2024 | 通过 | MMLU-Pro 文献和 NeurIPS 2024 Datasets and Benchmarks 正确；本论文 150 个知识问题为抽样。 |
| [wei2024simpleqa](https://arxiv.org/abs/2411.04368) | 2024 | 通过 | 原文真实；PDF 的第二作者确为 Nguyen Karina，与当前 BibTeX 一致，不凭常见姓名习惯判错；trap 是本论文池名。 |
| [zhang2024duplex](https://arxiv.org/abs/2406.15718) | 2024 | 通过 | 时分交织的 duplex 机制支持概括；ACL Anthology 确认 EMNLP 2024、作者和题名。 |

## 本次保存的复核记录

- `citation_audit/cached_results_check.json`：35 份 native 逐题记录及 600 题 manifest 的复核。
- `citation_audit/table1_crosscheck.json`：MiniCPM 主表与缓存结果的一致性检查。
- `citation_audit/nvda_table_check.json`：NVDA 主表与归档汇总的一致性检查。

以上 JSON 来自修订前的实验数据审计。它们验证归档结果和论文表格的一致性，不代表重新运行模型或重判每个答案。
