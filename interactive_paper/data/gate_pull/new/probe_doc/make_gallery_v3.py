"""Gallery page for the review round + causal re-capture, in the same card
format as nvda_v2_figures.html / nvda_gallery.html (self-contained HTML,
figures embedded as base64 PNG).

    uv run --with numpy --with matplotlib python make_gallery_v3.py
    -> gallery_v3.html (tracked) and ../nvda_probe_v3_figures.html (untracked copy)
"""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figs"
BLUE, GREEN, ORANGE, MINI = "#2a78d6", "#1baf7a", "#f28e2b", "#eb6834"
INK, MUT, GRID, BASE = "#0b0b0b", "#52514e", "#e6e4de", "#898781"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUT})
POOLS = ["striviaqa", "swebq", "sllama", "sdqa"]
PNAME = {"striviaqa": "Speech TriviaQA (OAB judge)", "swebq": "Speech Web Questions (OAB judge)", "sllama": "Llama Questions (OAB judge)", "sdqa": "SD-QA (our judge, real speech)"}
RATES = ["0.15", "0.3", "0.5"]

R3 = json.load(open(HERE / "remix_eval3.json"))
R4 = {r["name"]: r for r in json.load(open(HERE / "results_round4.json"))["results"]}
B1 = {r["name"]: r for r in json.load(open(HERE / "results_batch1.json"))}
DN = json.load(open(HERE / "doc_numbers.json"))
K_DEP = "deployed read (pass-3): onset_last|onset_mean8|user_mean @L30"
K_V2 = "v2: layer-avg x3, commit|onset_last|onset_mean8|run_mean"
K_CAU = "strictly causal: commit|pre_mean8|run_mean @L34"


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", name)


# ---- G1: per-pool accuracy vs escalation ----------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, pool in zip(axes, POOLS):
    x = np.array([0, 15, 30, 50, 100])
    for key, c, lab, mk in ((K_DEP, BLUE, "deployed (commit + 8 frames)", "o"), (K_V2, GREEN, "v2 (3-layer avg + commit frame)", "s"), (K_CAU, ORANGE, "strictly causal (commit frame, L34)", "^")):
        d = R3[key]["loc_official"][pool]
        ys = [d["local"]] + [d[f"gate@{r}"] for r in RATES] + [d["expert"]]
        ax.plot(x, ys, "-" + mk, color=c, ms=4, lw=1.5, label=lab)
    d = R3[K_DEP]["loc_official"][pool]
    ax.plot([0, 100], [d["local"], d["expert"]], color=BASE, ls="--", lw=1.1, label="matched-rate random")
    for xi, r in zip(x[1:4], RATES):
        p = R3[K_CAU]["loc_official"][pool][f"p@{r}"]
        ax.text(xi, R3[K_CAU]["loc_official"][pool][f"gate@{r}"] - .04, ("p<.001" if p < .001 else f"p={p:.3f}"), ha="center", fontsize=6.5, color=ORANGE)
    ax.set_title(f"{PNAME[pool]}  n={d['n']}", fontsize=8.5); ax.set_xticks([0, 15, 30, 50, 100]); ax.set_xlabel("escalation rate (%)")
    ax.grid(color=GRID, lw=.6)
axes[0].set_ylabel("accuracy (local kept, top-r escalated to gpt-5.5)"); axes[0].legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "g1_remix_pools_pass3.png")

# ---- G3: layer sweep, causal vs deployed --------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.6))
Ls = [22, 26, 30, 34, 38]
dep_p2 = {22: B1["onset@L22 stack"], 26: B1["onset@L26 stack"], 30: B1["baseline onset@L30 C=1e-4"], 34: B1["onset@L34 stack"], 38: B1["onset@L38 stack"]}
ax.plot(Ls, [dep_p2[L]["ext"]["mean"] for L in Ls], "-o", color=BLUE, label="deployed read (commit + 8 frames), pass-2 capture")
ax.plot(Ls, [R4[f"S2 @L{L}"]["ext"]["mean"] if L != 30 else R4["S2 at-commit: commit|pre_mean8|run_mean @L30"]["ext"]["mean"] for L in Ls], "-^", color=ORANGE, label="strictly causal (commit | pre_mean8 | run_mean), pass-3 capture")
ax.plot(Ls, [R4[f"S1 @L{L}"]["ext"]["mean"] if L != 30 else R4["S1 strictly pre-commit: pre_last|pre_mean8|run_mean @L30"]["ext"]["mean"] for L in Ls], ":v", color=ORANGE, alpha=.7, label="strictly pre-commit (pre_last | pre_mean8 | run_mean), pass-3")
ax.scatter([30], [R4[K_DEP.replace("deployed read (pass-3)", "A0' deployed read (pass-3)")]["ext"]["mean"]], marker="D", s=40, color=BLUE, zorder=5, label="deployed read, pass-3 replication")
ax.axhline(R4["E0 old eot read (pass-3): eot_last|eot_mean8|user_mean @L34"]["ext"]["mean"], color=BASE, ls="--", lw=1, label="old end-of-audio read @L34, pass-3")
ax.set_xticks(Ls); ax.set_xlabel("layer"); ax.set_ylabel("cold external AUC, 4-pool mean"); ax.set_ylim(.74, .84)
ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.grid(color=GRID, lw=.6)
save(fig, "g3_layer_causal_vs_deployed.png")

# ---- G4: fire rates at calibration thresholds --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, key, title in ((axes[0], K_DEP, "deployed read"), (axes[1], K_CAU, "strictly causal @L34")):
    w = .25
    for ti, (r, nominal, c) in enumerate((("0.15", 15, "#9ecae1"), ("0.3", 30, "#4292c6"), ("0.5", 50, "#08519c"))):
        vals = [100 * R3[key]["loc_official"][p]["fire_at_calib_thr"][r] for p in POOLS]
        xs = np.arange(4) + (ti - 1) * w
        ax.bar(xs, vals, width=w * .9, color=c, label=f"tier calibrated to {nominal} %")
        ax.hlines([nominal] * len(xs), xs - w / 2, xs + w / 2, color=INK, lw=1.2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(POOLS); ax.set_ylabel("% of pool escalated"); ax.set_ylim(0, 80)
    ax.set_title(f"{title}: realised fire rate at the calibration thresholds (black tick = nominal)", fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=.6)
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
save(fig, "g4_fire_rates.png")


# ---- HTML -----------------------------------------------------------------------------------
def img(name):
    p = FIG / name if (FIG / name).exists() else HERE.parents[3] / "figures" / name
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def cell(x, digits=3):
    return f"{x:.{digits}f}"


m = lambda key, lab, r: np.mean([R3[key][lab][p][f"gate@{r}"] for p in POOLS])
rnd = lambda r: np.mean([R3[K_DEP]["loc_official"][p][f"random@{r}"] for p in POOLS])
rows_remix = "".join(
    f"<tr><td>{lab}</td>" + "".join(f"<td>{cell(m(key, 'loc_official', r))}</td>" for r in RATES) + "</tr>"
    for key, lab in ((K_DEP, "deployed（commit + 8 帧，L30）"), (K_V2, "v2（三层平均 + commit 帧）"), (K_CAU, "strictly causal（commit 帧，L34）")))
rows_remix = f"<tr><td>matched-rate random</td>" + "".join(f"<td>{cell(rnd(r))}</td>" for r in RATES) + "</tr>" + rows_remix

reads = [("end of user audio @L34（旧读点）", "E0 old eot read (pass-3): eot_last|eot_mean8|user_mean @L34"),
         ("strictly causal @L34", "S2 @L34"), ("strictly causal @L30", "S2 at-commit: commit|pre_mean8|run_mean @L30"),
         ("<b>deployed</b>：commit + 8 帧 @L30", "A0' deployed read (pass-3): onset_last|onset_mean8|user_mean @L30"),
         ("deployed，causal user_mean（H_run）", "F1 deployed read, causal user_mean: onset_last|onset_mean8|run_mean @L30"),
         ("v2：三层平均 + commit 帧", "V2c layer-avg ×3 (26,30,34): commit|onset_last|onset_mean8|run_mean")]
rows_reads = ""
for lab, key in reads:
    r = R4[key]; d = r.get("delta_ext"); e = r["ext"]
    dtxt = "—" if not d else f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
    rows_reads += f"<tr><td>{lab}</td><td>{cell(r['oof'])}</td><td>{cell(r['lopo'])}</td><td>{cell(r['lopo_macro'])}</td><td>{cell(e['striviaqa'])}</td><td>{cell(e['swebq'])}</td><td>{cell(e['sllama'])}</td><td>{cell(e['sdqa'])}</td><td><b>{cell(e['mean'])}</b></td><td>{dtxt}</td></tr>"

drift = json.load(open(HERE / "results_round4.json"))["drift"]
fr = lambda pool: "/".join(f"{100*R3[K_DEP]['loc_official'][pool]['fire_at_calib_thr'][r]:.0f}" for r in RATES) + "%"
wp = sorted(DN["within_pool"], key=lambda r: -r["auc"])
rows_wp = "".join(f"<tr><td>{r['pool']}</td><td>{r['n']}</td><td>{r['fail']:.2f}</td><td>{r['auc']:.3f}</td></tr>" for r in wp if r["pool"] != "trap")

html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>NVDA Probe v3 — review round + causal re-capture</title>
<style>
:root {{ --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --hairline:rgba(11,11,11,.10); --nvda:#2a78d6; --aqua:#1baf7a; --warnbg:#fff8ec; --warnbd:#e8d9b5; --good:#006300; --bad:#b00020; }}
body {{ background:var(--page); color:var(--ink); font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; margin:0; padding:40px 20px 72px; }}
.wrap {{ max-width:980px; margin:0 auto; }}
header h1 {{ font-size:26px; font-weight:650; margin:0 0 6px; }}
.meta {{ color:var(--ink-2); font-size:13px; margin-bottom:14px; }}
.badges {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 30px; }}
.badge {{ font-size:12px; padding:3px 10px; border-radius:999px; border:1px solid var(--hairline); background:var(--surface); color:var(--ink-2); }}
.badge b {{ color:var(--ink); font-weight:600; }} .badge.warn {{ background:var(--warnbg); border-color:var(--warnbd); }}
.fig {{ background:var(--surface); border:1px solid var(--hairline); border-radius:10px; padding:22px 24px 18px; margin-bottom:26px; }}
.eyebrow {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin-bottom:2px; }}
.fig h2 {{ font-size:16px; font-weight:650; margin:0 0 2px; }} .fig .sub {{ color:var(--ink-2); font-size:12.5px; margin:0 0 14px; }}
.fig img {{ display:block; width:100%; height:auto; border-radius:4px; }}
.note {{ border-top:1px solid var(--hairline); margin-top:12px; padding-top:10px; color:var(--ink-2); font-size:13px; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; font-variant-numeric:tabular-nums; }}
th,td {{ text-align:left; padding:5px 8px; border-bottom:1px solid var(--hairline); }} th {{ color:var(--muted); font-weight:600; font-size:11.5px; }}
.overflow {{ overflow-x:auto; }} .up {{ color:var(--good); }} .down {{ color:var(--bad); }}
ul {{ margin:6px 0 0 18px; padding:0; }} li {{ margin:3px 0; }} code {{ font-size:12px; background:rgba(11,11,11,.05); padding:1px 4px; border-radius:3px; }}
</style></head><body><div class="wrap">
<header>
  <h1>NVDA Probe v3 — review round + causal re-capture（2026-09-01 夜）</h1>
  <div class="meta">NVIDIA NemotronLabs-VoiceChat-11B · calib 2,481（frozen + expansion + expansion2，committed rows）· pass-2 labels · 第三次独立 replay（pass 3，4×H100）· 四个外部池 cold（n=249/250/249/200）</div>
  <div class="badges">
    <span class="badge"><b>模型冻结</b> — 只训线性 probe</span>
    <span class="badge"><b>原生 duplex 协议</b> — 读点 = 模型自己的 commit-to-speak（首个 ≥3 帧非 PAD run）</span>
    <span class="badge"><b>externals cold</b> — 选择只用 calibration 指标（LOPO），外部池只报告</span>
    <span class="badge"><b>pass 3 复现</b> — deployed 读点 .809 vs pass 2 .808</span>
    <span class="badge warn"><b>offline replay</b> — NeMo 官方 offline 路径，greedy，bf16；非 live streaming</span>
    <span class="badge warn"><b>labels = pass 2</b> — pass 3 答案文本漂移 {drift['answer_text_changed']:.0%}，commit 帧移动 {drift['commit_frame_changed']:.1%}</span>
  </div>
</header>

<div class="fig">
  <div class="eyebrow">图 A · 三个 probe 在四个外部池上的 gate 收益</div>
  <h2>固定升级预算下的准确率：deployed ≈ v2；strictly causal 只低 0.5–1 个点</h2>
  <p class="sub">x = 升级比例（按 probe 分数取 top-r 送 gpt-5.5，其余保留本地答案）· y = 准确率（OAB 官方 judge；SD-QA 用我们的 judge）· 虚线 = 同预算随机升级 · 橙色 p 值 = causal probe vs random 置换检验</p>
  <img src="{img('g1_remix_pools_pass3.png')}" alt="per-pool remix">
  <div class="note"><b>解读：</b>三个 probe 在每个池、每个预算上都赢随机（最弱一格 SD-QA@15%，p=.010–.063）。四池均值 @15/30/50%：deployed .552/.650/.756，v2 .550/.650/.757，<b>strictly causal .546/.640/.749</b>（随机 .507/.577/.670）。AUC 上 causal 低 .03，但换成"固定预算下的准确率"只低 0.5–1 个点——因为差距集中在分数排序的中段，而不在被升级的头部。</div>
</div>

<div class="fig">
  <div class="eyebrow">图 B · 读点：causal 的代价</div>
  <h2>只读 commit 帧之前的状态，外部 AUC 掉 ≈ .03；多出来的信号来自模型自己开口的前 640 ms</h2>
  <p class="sub">左：同一次 replay（pass 3）上各读点的 cold external AUC（柱 = 四池均值，点 = 各池，Δ = 与 deployed 的 paired bootstrap 差及 95% CI）· 右：固定预算准确率（四池均值）</p>
  <img src="{img('nvda_probe_reads.png')}" alt="read points">
  <div class="note"><b>解读：</b>deployed 读点 = commit 帧起 8 帧（模型已经在说前 ~7 个 token）。strictly causal（commit 帧本身 + 之前 8 帧均值 + 到 commit 为止的用户音频均值）在 L22–L38 每一层都低 .025–.046。commit 之前的 8 帧窗口（H_pre）并不比旧的"用户音频末尾 8 帧"（H_eot，.782）更好——所以额外信号是答案起始本身，而不是 commit 之前听到的东西。这是论文里应写在"读点落在 commit 后 640 ms"旁边的数字。</div>
</div>

<div class="fig">
  <div class="eyebrow">图 C · 逐层</div>
  <h2>causal 读点的层曲线：L34 最好（.781），deployed 读点在 L26–L34 平坦</h2>
  <p class="sub">x = 层号 · y = cold external AUC 四池均值 · 蓝 = deployed 读点（pass 2 逐层）· 橙 = causal 两种窗口（pass 3）· 灰虚线 = 旧 eot 读点</p>
  <img src="{img('g3_layer_causal_vs_deployed.png')}" alt="layer sweep causal">
  <div class="note"><b>解读：</b>两条 causal 曲线几乎重合，说明 commit 帧本身和 commit 前最后一帧携带同样的信息。deployed 读点的层选择在 .005 内平坦（L26 .818 / L30 .808 / L34 .808），L26 略高但在 LOPO 宏平均上不占优，属噪声。</div>
</div>

<div class="fig">
  <div class="eyebrow">图 D · 运行点</div>
  <h2>固定 calibration 阈值给不出名义预算：同一阈值在 Llama-Q 只升级 {fr('sllama')}，在 SD-QA 升级 {fr('sdqa')}</h2>
  <p class="sub">柱 = 用 calibration OOF 分位数定的三档阈值（名义 15/30/50%）在各外部池上实际触发的升级比例 · 黑刻度 = 名义值</p>
  <img src="{img('g4_fire_rates.png')}" alt="fire rates">
  <div class="note"><b>解读：</b>probe 的排序（AUC）在每个池内都成立，但分数的绝对水平随池难度整体平移：容易的池（Llama-Q，本地正确率 .71）几乎不触发，难的池（SD-QA，.31）过度触发（本页数字为 pass 3；论文/doc 里 pass 2 的对应值是 1/4/12% 与 7/27/53%）。MiniCPM 实验也出现过相同问题，8bn/8bp 的解法是 per-pool / windowed 分位数阈值——图 A 的固定预算行就是 per-pool 分位数阈值会给出的结果。NVDA 侧还没移植（<code>scripts/26_pool_thresholds.py</code>）。</div>
</div>

<div class="fig">
  <div class="eyebrow">图 E · 配方天花板</div>
  <h2>≈85 个变体（两位独立 reviewer 的提案）：外部 AUC 最多 +.013，没有一个改变固定预算下的准确率</h2>
  <p class="sub">四个面板：pooled OOF · pooled LOPO · LOPO 宏平均（held-out 池内 AUC 均值，去掉"池身份"捷径）· cold external 均值；右侧黑线 = 与 deployed 的 paired bootstrap 95% CI</p>
  <img src="{img('fig10_review_round.png')}" alt="review round variants">
  <div class="note"><b>解读：</b>PCA/PLS/白化、shrinkage LDA、robust scaling、池/类别重加权、逐帧特征、双读点拼接、多层拼接——全部在 ±.01 内或更差。唯一一致的小幅收益是"三层 logit 平均 + commit 帧"（v2：OOF +.003，pooled LOPO +.023，external +.007 [−.000, +.015]，在 pass 3 上复现为 +.004）。<b>结论：线性 probe 在这组特征上已到天花板；能动的杠杆是数据（expansion3）和阈值，不是配方。</b></div>
</div>

<div class="fig">
  <div class="eyebrow">图 F · 诚实体检</div>
  <h2>pooled OOF .820 里有多少是"池身份"？只输出池失败率的 oracle 已有 AUC .752</h2>
  <p class="sub">x = 各 calibration 池的失败率 · y = pooled OOF 分数在该池内部的 AUC · 气泡大小 = n</p>
  <img src="{img('fig8_within_pool_auc.png')}" alt="within pool">
  <div class="note"><b>解读：</b>池内 AUC .60–.88（均值 .713；LOPO 下 .687），所以 .820 中约 +.07 是 probe 真正区分"同类题里哪道会错"的能力，其余来自池的基础难度。这不是 bug（部署时池难度也是真实信号），但审稿人会要求把池内均值和 LOPO 与 pooled 数字并列报告。MCQ 类知识池（arc/mmlu/openbook/commonsense）最弱也最均衡——扩数据应该往这里加。</div>
</div>

<div class="fig">
  <div class="eyebrow">数值表 1</div>
  <h2>读点对照（pass 3，同一次 replay；L2-logistic C=1e-4 + scaler；lopoM = held-out 池内 AUC 均值，13 个合格池）</h2>
  <div class="overflow"><table>
    <tr><th>读点</th><th>OOF</th><th>LOPO</th><th>lopoM</th><th>TriviaQA</th><th>WebQ</th><th>Llama-Q</th><th>SD-QA</th><th>均值</th><th>Δ vs deployed [95% CI]</th></tr>
    {rows_reads}
  </table></div>
</div>

<div class="fig">
  <div class="eyebrow">数值表 2</div>
  <h2>固定预算准确率，四池均值（OAB 官方 judge / SD-QA 我们的 judge）</h2>
  <div class="overflow"><table>
    <tr><th>probe</th><th>@15%</th><th>@30%</th><th>@50%</th></tr>
    {rows_remix}
  </table></div>
</div>

<div class="fig">
  <div class="eyebrow">数值表 3</div>
  <h2>池内 AUC（pooled OOF 分数，deployed probe，pass 2；trap 池只有 2 个负例，不计）</h2>
  <div class="overflow"><table>
    <tr><th>池</th><th>n</th><th>fail rate</th><th>池内 AUC</th></tr>
    {rows_wp}
  </table></div>
  <div class="note"><b>诚实脚注：</b>
    <ul>
      <li><b>"deployed"</b> = 2026-09-01 训出的 probe（onset@L30，calib 2,481，pass-2 labels，导出为 <code>gate_demo_nvda.json</code>）。它在读点对齐到原生 commit-to-speak（8be 的 NVDA 类比）之后训练；与 MiniCPM 8bl serving-config 问题无关——NVDA 的 capture 走 NVIDIA 官方 offline 路径、库默认 greedy 解码，与 streaming pipeline 默认一致。</li>
      <li><b>labels：</b>pass 3 复用 pass-2 labels。两次 replay 之间答案文本漂移 {drift['answer_text_changed']:.0%}（bf16 + batching），但 label 翻转率 pass1→pass2 只有 1.3%，commit 帧 {100-100*drift['commit_frame_changed']:.1f}% 不动。</li>
      <li><b>噪声：</b>单池 AUC SE ≈ ±.03–.04（n=200–250）；只有 paired bootstrap 的 Δ 可以下结论。OOF 的 CV 种子抖动 ≈ .0005，抽样 SE ≈ .009。</li>
      <li><b>选择规则：</b>所有变体按 calibration-only 的 LOPO 指标排序，外部池只报告；但 reviewer 与我在迭代时都看到了外部数字，严格意义上四个外部池现在是 development set——确认需要新数据（expansion3 / SD-QA 其他方言）。</li>
      <li><b>配方差异（跨家族表要注明）：</b>NVDA 用 StandardScaler（+.015–.023），MiniCPM 部署配方不用；C 都是 OOF 网格选，NVDA 1e-4，MiniCPM 3e-4。</li>
      <li>数据与脚本：<code>data/gate_pull/new/probe_doc/</code>（<code>experiments4.py</code>、<code>remix_eval3.py</code>、<code>results_round4.json</code>、<code>remix_eval3.json</code>）；pass-3 全层 shards（12 GB）未纳入 git。</li>
    </ul>
  </div>
</div>
</div></body></html>"""
out = HERE / "gallery_v3.html"; out.write_text(html, encoding="utf-8")
(HERE.parent / "nvda_probe_v3_figures.html").write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/2**20:.1f} MB) and ../nvda_probe_v3_figures.html")
