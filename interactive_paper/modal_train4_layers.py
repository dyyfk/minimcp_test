"""Layer ablation of the deployed three-position probe (review item R3,
2026-09-02): capture EVERY decoder layer in the same streaming replay
that produced the deployed v3 features (modal_train2.py::eoth2_shard,
five layers), then re-fit the identical eot_last + eot_mean8 + user_mean
feature set at every depth and score the identical evaluation pools.

Why: the paper's "mid-network rather than final-layer" claim rests on
single-position text-input LOPO sweeps; the deployed probe was never
re-fit at another depth on the audio path it actually runs on, and the
audio-calibrated single-position sweep (data/layers/
layer_sweep_minicpm-o45-audio.json) shows the final layer is NOT worse
than L22 on any held-out pool. This settles it on the deployed feature
set, the deployed training mix, and the paper's external pools.

Stages
  # 1. GPU capture (~3,900 replays, ~14 H100-h like eoth2; ~10 GB on
  #    the gate-data volume as float16). Externals first so the partial
  #    ablation can run while calibration tags finish.
  modal run modal_train4_layers.py::run_eoth3 --tags striviaqa,swebq,sdqa,sllama,sreason,frozen
  modal run modal_train4_layers.py::run_eoth3 --tags expansion,expansion2
  # 2. CPU ablation on the volume (writes /data/layer_ablation_eoth3.json + .md)
  modal run modal_train4_layers.py::layer_ablation3 --source eoth3
  # 0. (available TODAY, no GPU) same ablation over the five layers the
  #    eoth2 capture already holds: L14/18/22/26/30
  modal run modal_train4_layers.py::layer_ablation3 --source eoth2
  # 3. pull the receipts
  modal volume get gate-data layer_ablation_eoth3.json figures/
  modal volume get gate-data layer_ablation_eoth3.md figures/

Protocol (pre-registered in LAYER_ABLATION.md): feature set, C, training
rows, evaluation pools and labels are all frozen to the v3 recipe; the
only free variable is the layer. Externals are read once, after the
capture, by the ablation function; nothing is selected on them.
"""
import json
import os
import sys

from modal_app import app, GPU_VOL, gate_data, DATA, MODEL_DIR
from modal_train import XLABELS
from modal_train2 import (image_st2, util_st2, AUDIO2, Q2_FILES, EOTH2,
                          EXT_TRACES, K_EOT, YLABELS, ART_V3, _read_q2)

HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN2_PY = os.path.join(HERE, "modal_train2.py")
image_st3 = image_st2.add_local_file(_TRAIN2_PY, "/root/modal_train2.py")
util_st3 = util_st2.add_local_file(_TRAIN2_PY, "/root/modal_train2.py")

EOTH3 = f"{DATA}/eoth3"                  # + _{tag}.shard{i}.npz
FEATS_AUDIO = f"{DATA}/features_minicpm-o45-audio.parquet"


def _parse_layers(spec: str, n_layers: int) -> list:
    if not spec or spec == "all":
        return list(range(n_layers))
    return sorted({int(x) for x in spec.split(",") if x.strip()})


@app.function(image=image_st3, gpu="H100", volumes=GPU_VOL,
              timeout=60 * 60 * 3, max_containers=8)
def eoth3_shard(tag: str, shard: list, shard_id: int,
                layers: str = "all") -> int:
    """One streaming replay per query — byte-for-byte the eoth2 replay
    (same session reset, system prompt, 1 s chunks, assistant prefill,
    rolling K_EOT tail across forwards, user-audio running mean) — with
    hooks on EVERY decoder layer instead of five.

    Writes {EOTH3}_{tag}.shard{shard_id}.npz with
      H_eot  (n, nL, K_EOT, d) float16, H_mean (n, nL, d) float16,
      eot_len (n,) int16, layers (nL,) int, ids (n,) str."""
    import glob as _glob
    import inspect
    import shutil
    import numpy as np
    import librosa
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache = os.path.expanduser("~/.cache/huggingface/modules/"
                               "transformers_modules/"
                               + os.path.basename(MODEL_DIR))
    os.makedirs(cache, exist_ok=True)
    for f in _glob.glob(f"{MODEL_DIR}/*.py"):
        shutil.copy(f, cache)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=False,
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    def call_def(fn, /, **kw):
        params = set(inspect.signature(fn).parameters)
        return fn(**{k: v for k, v in kw.items() if k in params})

    all_layers = model.llm.model.layers
    LAY = _parse_layers(layers, len(all_layers))
    print(f">>> {tag} shard {shard_id}: hooks on {len(LAY)}/{len(all_layers)} "
          f"layers", flush=True)

    # rolling last-K_EOT-token window ACROSS forwards (the streaming
    # assistant prefill runs 1-token forwards; see eoth2_shard)
    state = {"accum": False, "tail": {}, "sum": {}, "cnt": 0}

    def mk_hook(L, count_here):
        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            h = hs[0].detach().float()               # (T, d)
            t = h[-K_EOT:].cpu()
            prev = state["tail"].get(L)
            state["tail"][L] = (t if prev is None
                                else torch.cat([prev, t])[-K_EOT:])
            if state["accum"]:
                s = h.sum(0).cpu()
                prev = state["sum"].get(L)
                state["sum"][L] = s if prev is None else prev + s
                if count_here:
                    state["cnt"] += h.shape[0]
        return hook

    handles = [all_layers[L].register_forward_hook(mk_hook(L, L == LAY[0]))
               for L in LAY]

    adir = AUDIO2[tag]
    ids, E, M, ELEN = [], [], [], []
    try:
        for k, q in enumerate(shard):
            wav = f"{adir}/{q['id']}.wav"
            if not os.path.exists(wav):
                continue
            au, _ = librosa.load(wav, sr=16000, mono=True)
            chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
            model.reset_session()
            state.update(accum=False, tail={}, sum={}, cnt=0)
            sys_msg = call_def(model.get_sys_prompt, mode="omni",
                               language="en")
            call_def(model.streaming_prefill, session_id="s1",
                     msgs=[sys_msg], tokenizer=tok)
            state["accum"] = True
            for i, ch in enumerate(chunks):
                if len(ch) < 16000:
                    ch = np.pad(ch, (0, 16000 - len(ch)))
                call_def(model.streaming_prefill, session_id="s1",
                         msgs=[{"role": "user",
                                "content": [ch.astype(np.float32)]}],
                         tokenizer=tok,
                         is_last_chunk=(i == len(chunks) - 1))
            state["accum"] = False
            got = None
            for content in ("", " "):
                try:
                    call_def(model.streaming_prefill, session_id="s1",
                             msgs=[{"role": "assistant",
                                    "content": [content]}],
                             tokenizer=tok, is_last_chunk=True)
                    got = {L: state["tail"][L] for L in LAY}
                    break
                except Exception:
                    continue
            if got is None or state["cnt"] == 0:
                continue
            d = got[LAY[0]].shape[1]
            eot = np.zeros((len(LAY), K_EOT, d), dtype=np.float16)
            mean = np.zeros((len(LAY), d), dtype=np.float16)
            elen = got[LAY[0]].shape[0]
            for j, L in enumerate(LAY):
                t = got[L].numpy()
                eot[j, K_EOT - t.shape[0]:] = t.astype(np.float16)
                mean[j] = (state["sum"][L].numpy()
                           / state["cnt"]).astype(np.float16)
            ids.append(q["id"])
            E.append(eot)
            M.append(mean)
            ELEN.append(elen)
            if k < 3 or k % 50 == 0:
                print(f"  [{k}] {q['id']}", flush=True)
    finally:
        for h in handles:
            h.remove()

    if not ids:
        print(f">>> {tag} shard {shard_id}: nothing captured", flush=True)
        return 0
    np.savez_compressed(f"{EOTH3}_{tag}.shard{shard_id}.npz",
                        ids=np.array(ids), H_eot=np.stack(E),
                        H_mean=np.stack(M),
                        eot_len=np.array(ELEN, dtype=np.int16),
                        layers=np.array(LAY))
    gate_data.commit()
    print(f">>> wrote eoth3_{tag} shard {shard_id} ({len(ids)})", flush=True)
    return len(ids)


@app.local_entrypoint()
def run_eoth3(tags: str = "striviaqa,swebq,sdqa,sllama,sreason,frozen,"
                          "expansion,expansion2",
              workers: int = 3, limit: int = 0, layers: str = "all"):
    """Flat starmap across tags (max_containers bounds H100 use). With
    --limit N only the first N queries of each tag are captured into
    shard 99 (smoke test)."""
    items = []
    for tag in [t.strip() for t in tags.split(",") if t.strip()]:
        qs = _read_q2.remote(tag)
        if limit:
            qs = qs[:limit]
        w = 1 if limit else (workers if len(qs) > 400 else 2)
        shards = [qs[i::w] for i in range(w)]
        items += [(tag, shards[i], i if not limit else 99, layers)
                  for i in range(w)]
        print(f">>> {tag}: {len(qs)} queries / {w} shards")
    total = sum(eoth3_shard.starmap(items))
    print(f">>> captured {total}")


def _labels_train():
    """Training labels exactly as modal_train2._train_xy: frozen calib
    (features parquet, split == calib) + expansion + expansion2."""
    import pandas as pd
    feats = pd.read_parquet(FEATS_AUDIO)[["id", "split", "escalate_label"]]
    cal = feats[(feats["split"] == "calib") & feats["escalate_label"].notna()]
    lab = dict(zip(cal["id"], cal["escalate_label"].astype(int)))
    for path in (XLABELS, YLABELS):
        df = pd.read_parquet(path)
        df = df[df["escalate_label"].notna()]
        lab.update(dict(zip(df["id"], df["escalate_label"].astype(int))))
    return lab


def _labels_pools():
    """Evaluation labels exactly as modal_train2.eval_transfer3: external
    never-arm local failure per pool + frozen test split."""
    import pandas as pd
    pools = {}
    for bench, tpath in EXT_TRACES:
        try:
            tr = pd.read_parquet(f"{DATA}/{tpath}")
        except FileNotFoundError:
            print(f">>> {bench}: no traces — skipped", flush=True)
            continue
        nev = tr[(tr["tier"] == "never") & tr["heard_ok"].notna()]
        pools[bench] = dict(zip(nev["id"], 1 - nev["heard_ok"].astype(int)))
    feats = pd.read_parquet(FEATS_AUDIO)[["id", "split", "escalate_label"]]
    tst = feats[(feats["split"] == "test") & feats["escalate_label"].notna()]
    pools["frozen-test"] = dict(zip(tst["id"], tst["escalate_label"].astype(int)))
    return pools


@app.function(image=util_st3, volumes={DATA: gate_data}, timeout=60 * 180,
              cpu=16, memory=65536)
def layer_ablation3(source: str = "eoth3", layers: str = "", C: float = 1e-4,
                    B: int = 2000, with_oof: bool = False,
                    out: str = "") -> str:
    """Re-fit the deployed feature set at every captured layer and score
    the paper's evaluation pools. `source` = eoth3 (all layers, new
    capture) or eoth2 (the five layers already on the volume)."""
    import numpy as np
    sys.path.insert(0, "/workspace/gate")
    import layer_ablation as la

    prefix = {"eoth3": EOTH3, "eoth2": EOTH2}[source]

    def load(tag):
        return la.load_shards(f"{prefix}_{tag}.shard*.npz")

    lab = _labels_train()
    parts = []
    for tag in ("frozen", "expansion", "expansion2"):
        sh = load(tag)
        keep = [j for j, i in enumerate(sh.ids) if i in lab]
        parts.append(sh.subset(keep))
        print(f">>> train {tag}: {len(keep)}/{sh.n} labeled", flush=True)
    train = la.Shards(sum((p.ids for p in parts), []),
                      np.concatenate([p.E for p in parts]),
                      np.concatenate([p.M for p in parts]),
                      np.concatenate([p.elen for p in parts]),
                      parts[0].layers)
    y = np.array([lab[i] for i in train.ids])

    pool_labels = _labels_pools()
    pools = []
    for name, lb in pool_labels.items():
        tag = "frozen" if name == "frozen-test" else name
        try:
            sh = load(tag)
        except FileNotFoundError:
            print(f">>> {name}: no {source} shards — skipped", flush=True)
            continue
        keep = [j for j, i in enumerate(sh.ids) if i in lb]
        if len(keep) < 30:
            print(f">>> {name}: only {len(keep)} labeled — skipped", flush=True)
            continue
        yy = np.array([lb[sh.ids[j]] for j in keep])
        if yy.min() == yy.max():
            continue
        pools.append(la.Pool(name, sh.subset(keep), yy,
                             external=(name != "frozen-test")))
        print(f">>> pool {name}: n={len(keep)} fail={yy.mean():.2f}",
              flush=True)

    anchor = None
    if os.path.exists(ART_V3):
        anchor = json.load(open(ART_V3))
    lay = ([int(x) for x in layers.split(",") if x.strip()]
           if layers else None)
    print(f">>> train n={len(y)} fail={y.mean():.2f}; layers="
          f"{lay or list(train.layers)}; C={C}", flush=True)
    res = la.run_ablation(train, y, pools, layers=lay, C=C, anchor=anchor,
                          B=B, with_oof=with_oof)
    res.notes.append(f"source={source}; labels: train = frozen calib + "
                     "expansion + expansion2 (v3 recipe); pools = never-arm "
                     "local failure (EXT_TRACES) + frozen test split")
    out = out or f"{DATA}/layer_ablation_{source}.json"
    la.save_json(res, out)
    md = la.markdown_table(res)
    with open(out.replace(".json", ".md"), "w") as fh:
        fh.write(md + "\n")
    gate_data.commit()
    print("\n" + md, flush=True)
    print(f">>> wrote {out} (+ .md)", flush=True)
    return md
