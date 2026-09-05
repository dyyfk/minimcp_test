"""Native-duplex regime feature dump (8be) — the §8bb recipe moved onto
MiniCPMODuplex.

Per query: fresh duplex session (prepare) -> the TTS question wav streams
in 1 s units -> the head itself decides when to speak. The L22 read
happens EXACTLY where the deployed demo reads it (demo_duplex.py): after
the generate call of the first listen->speak chunk, tail-8 + user_mean
features, accum wrapping only the audio prefills. The spoken answer text
is collected to end_of_turn for in-regime judging later.

No realtime pacing (chunks feed back-to-back), generate_audio=False —
the TTS/vocoder path never touches the LLM cache, so features are
byte-equivalent to the deployed demo at ~2x the speed.

Outputs (gate-data volume, §8bb naming):
  /data/frozen_native_{tag}_feats.shard{i}.npz   (ids, X)
  /data/frozen_native_{tag}_traces.jsonl.shard{i}
    {id, pool, onset_chunk, n_q_chunks, no_speak, onset_score,
     answer_text, eot_seen, n_ans_chunks}

Run (cwd=interactive_paper, PYTHONUTF8=1):
  modal run modal_native_dump.py::run_native --pool frozen --split calib \
      --tag calib --limit 10          # smoke
  modal run modal_native_dump.py::run_native --pool frozen --split calib \
      --tag calib --workers 4         # calib 360
  modal run modal_native_dump.py::run_native --pool expansion --tag exp \
      --workers 8                     # 800
  modal run modal_native_dump.py::run_native --pool expansion2 --tag exp2 \
      --workers 8                     # 1150
  modal run modal_native_dump.py::run_native --pool frozen --split test \
      --tag test --workers 4          # test 240 (validity)
  modal run modal_native_dump.py::run_native --pool expansion3 \
      --tag exp3 --workers 8          # ~2300 (modal_train3.py)
  modal run modal_native_dump.py::run_native --pool expansion3zh \
      --tag exp3zh --workers 2        # ~355 zh (modal_train3.py)
"""
import json
import os
import sys
import time

import modal

app = modal.App("native-dump")
gate_data = modal.Volume.from_name("gate-data")
weights = modal.Volume.from_name("minicpm-o45-weights")
DATA = "/data"
MODEL_DIR = "/workspace/models/MiniCPM-o-4_5"
PROMPT_WAV = f"{MODEL_DIR}/assets/system_ref_audio.wav"
LAYER = 22
K3 = 8
OUT = f"{DATA}/frozen_native"
MAX_WAIT = 12      # silence chunks before giving up on speak-onset
MAX_ANS = 60       # answer chunks before truncating (smoke: 2/10 GSM
                   # answers overran 40; truncation biases the judge)

# VoiceBench official open-ended judge (verbatim copy of
# modal_bench.VB_META_PROMPT_OPEN — inlined because importing
# modal_bench drags modal_app's image definitions into the container)
VB_JUDGE_MODEL = "gpt-4o-mini"
VB_META_PROMPT_OPEN = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user’s instruction and models’ response:
### [Instruction]: {prompt}
### [Response]: {response}

After evaluating, please output the score only without anything else.
You don’t need to provide any explanations.
"""

from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")

util_img = (modal.Image.debian_slim(python_version="3.11")
            .pip_install("pandas", "pyarrow")
            .add_local_file(_APP_PY, "/root/modal_app.py"))
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libsndfile1")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install(
        "minicpmo-utils[all]",
        "transformers==4.51.0",
        "accelerate==1.12.0",
        "setuptools<81",
        "pydantic>=2.11",
        "PyYAML",
        "soundfile",
        "opencv-python-headless",
        "huggingface_hub[hf_transfer]",
        "scikit-learn",
        "pandas",
        "pyarrow",
        "openai",
        "sentencepiece",
        "fastapi[standard]",   # unused here — keeps the layer hash equal
                               # to demo_app/demo_duplex so the big pip
                               # layer is a cache hit, not a 30-min build
    )
    .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
    .add_local_file(_APP_PY, "/root/modal_app.py"))

FEAT_POOLS = {
    "frozen":     (f"{DATA}/queries.jsonl",            f"{DATA}/audio_pool"),
    "expansion":  (f"{DATA}/queries_expansion.jsonl",  f"{DATA}/audio_expansion"),
    "expansion2": (f"{DATA}/queries_expansion2.jsonl", f"{DATA}/audio_expansion2"),
    "expansion3": (f"{DATA}/queries_expansion3.jsonl", f"{DATA}/audio_expansion3"),
    "expansion3zh": (f"{DATA}/queries_expansion3zh.jsonl", f"{DATA}/audio_expansion3zh"),
    "expansion4zh": (f"{DATA}/queries_expansion4zh.jsonl", f"{DATA}/audio_expansion4zh"),
    "expansion5rs": (f"{DATA}/queries_expansion5rs.jsonl", f"{DATA}/audio_expansion5rs"),
    "striviaqa":  (f"{DATA}/queries_striviaqa.jsonl",  f"{DATA}/bench_audio"),
    "swebq":      (f"{DATA}/queries_swebq.jsonl",      f"{DATA}/bench_audio"),
    "sllama":     (f"{DATA}/queries_sllama.jsonl",     f"{DATA}/bench_audio"),
    "sdqa":       (f"{DATA}/queries_sdqa.jsonl",       f"{DATA}/sdqa_audio"),
    "sreason":    (f"{DATA}/queries_sreason.jsonl",    f"{DATA}/bench_audio"),
    "valpaca":    (f"{DATA}/queries_valpaca.jsonl",    f"{DATA}/bench_audio"),
    "flooract":   (f"{DATA}/queries_flooract.jsonl",   f"{DATA}/flooract_audio"),
    "reqq":       (f"{DATA}/queries_reqq.jsonl",       f"{DATA}/reqq_audio"),
    "stopq":      (f"{DATA}/queries_stopq.jsonl",      f"{DATA}/stopq_audio"),
    "fresh":      (f"{DATA}/queries_fresh.jsonl",      f"{DATA}/audio_fresh"),
}

# 8ch: post-cut context simulation (see native_shard postcut flag) —
# the stall + partial-relay turns teacher-forced before the stim,
# mirroring what the 8cf serving path leaves behind after a barge cut
# (the model's own onset words arrive naturally at the carrier commit;
# the stall line continues that open turn, exactly like the live fire).
POSTCUT_STALL = " Hmm, let me double-check that — one moment."
POSTCUT_RELAY = (" Nvidia (NVDA) is currently trading at $230.36 USD,"
                 " up $1.77")


@app.function(image=gpu_image, gpu="H100",
              volumes={"/workspace/models": weights, DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 60 * 4)
def native_shard(shard: list, shard_id: int = -1, tag: str = "",
                 audio_dir: str = f"{DATA}/audio_pool",
                 temperature: float = 0.0,
                 carrier: str = "",
                 official_cfg: int = 0,
                 postcut: int = 0) -> list:
    """carrier: path to a question wav. If set, every query becomes the
    SECOND turn of a session: carrier question -> model answers to
    end_of_turn (capped) -> per-deployment sum/cnt reset -> target
    utterance -> features at ITS onset. This matches how live floor
    turns and follow-ups actually arrive (8bj: standalone-calibrated
    act scores shift once conversational context enters the tail).

    postcut (8ch, requires carrier): instead of letting the model
    answer the carrier, reproduce the 8cf post-barge-cut context — at
    the carrier commit teacher-force the stall line onto the open turn
    and close it, hold listen a few chunks (thinker window), teacher-
    force a PARTIAL relay turn + turn_eos (the cut), reset the force-
    listen counter like the live cut path, then feed the target
    utterance. The act read at its onset is then in-distribution for
    barge-in follow-ups."""
    import glob as _glob
    import shutil

    import librosa
    import numpy as np
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
        init_vision=False, init_audio=True, init_tts=True,
    ).eval().cuda()
    _ = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    duplex = model.as_duplex(generate_audio=False)
    if official_cfg:      # 8bl: official serving config
        duplex.force_listen_count = 3
    SYS = ("You are a friendly assistant." if official_cfg
           else "Streaming Omni Conversation.")
    GKW = {"top_k": 20} if official_cfg else {}
    ref, _sr = librosa.load(PROMPT_WAV, sr=16000, mono=True)
    print(f">>> native-dump shard{shard_id}: {len(shard)} queries",
          flush=True)

    st3 = {"accum": False, "tail": None, "sum": None, "cnt": 0}

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[0].detach().float()
        t = h[-K3:].cpu()
        st3["tail"] = (t if st3["tail"] is None
                       else torch.cat([st3["tail"], t])[-K3:])
        if st3["accum"]:
            sm = h.sum(0).cpu()
            st3["sum"] = sm if st3["sum"] is None else st3["sum"] + sm
            st3["cnt"] += h.shape[0]
    h = model.llm.model.layers[LAYER].register_forward_hook(hook)

    def feat_now():
        parts = [st3["tail"][-1], st3["tail"].mean(0),
                 st3["sum"] / max(1, st3["cnt"])]
        return torch.cat(parts).numpy().astype(np.float32)

    rng = np.random.default_rng(3)

    def sil():
        return rng.normal(0, 0.003, 16000).astype(np.float32)

    car_chunks = None
    if carrier:
        cau, _cs = librosa.load(carrier, sr=16000, mono=True)
        car_chunks = [cau[i:i + 16000] for i in range(0, len(cau), 16000)]
        car_chunks = [np.pad(c, (0, 16000 - len(c)))
                      if len(c) < 16000 else c for c in car_chunks]

    def run_carrier():
        """First turn: carrier question, model answers to eot (capped),
        then the deployment-mirroring sum/cnt reset."""
        onset, spoke, ended = None, 0, False
        feed = list(car_chunks) + [None] * 30
        for ci, ch in enumerate(feed):
            st3["accum"] = True
            ok = duplex.streaming_prefill(
                audio_waveform=(sil() if ch is None
                                else ch.astype(np.float32)))
            st3["accum"] = False
            if not ok.get("success"):
                continue
            rr = (duplex.streaming_generate(temperature=temperature,
                                             **GKW)
                  if temperature else duplex.streaming_generate(**GKW))
            if not rr["is_listen"]:
                onset = onset if onset is not None else ci
                spoke += 1
            if rr.get("end_of_turn"):
                ended = True
                break
            if spoke >= 25:
                break
        st3.update(sum=None, cnt=0)      # demo resets at end_of_turn
        return ended

    def speak_forced(text, end_turn=True):
        """8ch: inline teacher-forcing (checkpoint modeling sources have
        no speak_forced) — mirrors demo_duplex._speak_forced: normal
        speak-unit framing, no TTS (generate_audio=False here anyway)."""
        d = duplex
        ids = d.tokenizer.encode(text, add_special_tokens=False)
        d.pending_logits = None
        unit_end = d.tokenizer.convert_tokens_to_ids("</unit>")

        def _feed(tid):
            d.decoder.feed(d.decoder.embed_token(tid))
            d.total_ids.append(tid)

        groups = [ids[i:i + 18] for i in range(0, len(ids), 18)] or [[]]
        for gi, g in enumerate(groups):
            last = gi == len(groups) - 1
            d.decoder.register_unit_start()
            gen = []
            _feed(d.unit_token_id)
            _feed(d.speak_token_id)
            d.current_turn_ended = False
            for tid in g:
                _feed(tid)
                gen.append(tid)
                d.res_ids.append(tid)
                d.speak_count += 1
            if last and end_turn:
                _feed(d.turn_eos_token_id)
                gen.append(d.turn_eos_token_id)
                d.current_turn_ended = True
                _feed(d.chunk_tts_eos_token_id)
            else:
                _feed(d.chunk_eos_token_id)
            _feed(unit_end)
            d.decoder.register_unit_end(
                input_type="text", generated_tokens=gen, is_listen=False,
                generated_text=d.tokenizer.decode(
                    g, skip_special_tokens=True))
            d.total_hidden.append([])
        if end_turn:
            d.tts_text_start_pos = 0
            d.tts_past_key_values = None
            d.tts_current_turn_start_time = None

    def run_postcut():
        """8ch: carrier question -> model COMMITS (first non-listen
        chunk voices its onset words) -> stall teacher-forced onto the
        open turn + closed -> ~4 forced-listen chunks (thinker window)
        -> partial relay turn teacher-forced + eos (the cut) -> live
        cut-path resets. Mirrors demo_duplex 8cf exactly."""
        committed = False
        feed = list(car_chunks) + [None] * 15
        for ci, ch in enumerate(feed):
            st3["accum"] = True
            ok = duplex.streaming_prefill(
                audio_waveform=(sil() if ch is None
                                else ch.astype(np.float32)))
            st3["accum"] = False
            if not ok.get("success"):
                continue
            rr = (duplex.streaming_generate(temperature=temperature,
                                             **GKW)
                  if temperature else duplex.streaming_generate(**GKW))
            if not rr["is_listen"]:
                committed = True
                break
        speak_forced(POSTCUT_STALL, end_turn=True)   # fire
        for _ in range(4):                           # thinker window
            duplex._streaming_generate_count = 0
            duplex.force_listen_count = 1
            st3["accum"] = True
            ok = duplex.streaming_prefill(audio_waveform=sil())
            st3["accum"] = False
            if ok.get("success"):
                (duplex.streaming_generate(temperature=temperature,
                                           **GKW)
                 if temperature else duplex.streaming_generate(**GKW))
        speak_forced(POSTCUT_RELAY, end_turn=True)   # partial relay+cut
        duplex._streaming_generate_count = 0         # live cut path
        duplex.force_listen_count = 3
        st3.update(sum=None, cnt=0)
        return committed

    traces, feat_ids, feat_X, feat_pre = [], [], [], []
    feat_post = {1: [], 2: [], 3: []}   # 8bw: read again after k answer chunks
    feat_npost = []
    try:
        for qi, q in enumerate(shard):
            wav_p = f"{audio_dir}/{q['id']}.wav"
            if not os.path.exists(wav_p):
                continue
            au, _sr = librosa.load(wav_p, sr=16000, mono=True)
            chunks = [au[i:i + 16000] for i in range(0, len(au), 16000)]
            chunks = [np.pad(c, (0, 16000 - len(c)))
                      if len(c) < 16000 else c for c in chunks]

            duplex.prepare(
                prefix_system_prompt=SYS,
                ref_audio=ref, prompt_wav_path=None)
            st3.update(tail=None, sum=None, cnt=0, accum=False)
            if postcut:
                car_ended = run_postcut()
            else:
                car_ended = run_carrier() if car_chunks else None

            onset_chunk, onset_vec, onset_score = None, None, None
            pre_vec, onset_pre = None, None   # 8br: pre-generate read
            post_vecs = []                    # 8bw: L22 after each answer chunk
            texts, eot_seen, n_ans = [], False, 0
            feed = list(chunks) + [None] * (MAX_WAIT + MAX_ANS)
            for ci, ch in enumerate(feed):
                st3["accum"] = True
                ok = duplex.streaming_prefill(
                    audio_waveform=(sil() if ch is None
                                    else ch.astype(np.float32)))
                st3["accum"] = False
                if not ok.get("success"):
                    continue
                if onset_chunk is None:
                    # 8br causal diagnostic: the same L22 read taken
                    # BEFORE this chunk's generate (no answer tokens)
                    pre_vec = feat_now()
                r = (duplex.streaming_generate(temperature=temperature,
                                                **GKW)
                     if temperature else duplex.streaming_generate(**GKW))
                if r["is_listen"]:
                    if onset_chunk is not None and eot_seen:
                        break
                    if (onset_chunk is None
                            and ci >= len(chunks) + MAX_WAIT - 1):
                        break               # never spoke
                    continue
                if onset_chunk is None:
                    # deployed read point: after the onset chunk's
                    # generate (demo_duplex.py semantics, verbatim)
                    onset_chunk = ci
                    onset_vec = feat_now()
                    onset_pre = pre_vec
                elif len(post_vecs) < 3:
                    # 8bw later read points: the same 12,288-d read
                    # after the 1st/2nd/3rd answer chunk following
                    # onset (tail now holds answer tokens; user_mean
                    # is unchanged since accum is off during generate)
                    post_vecs.append(feat_now())
                if r.get("text"):
                    texts.append(r["text"])
                n_ans += 1
                if r.get("end_of_turn"):
                    eot_seen = True
                    break
                if n_ans >= MAX_ANS:
                    break

            no_speak = onset_chunk is None
            if not no_speak:
                feat_ids.append(q["id"])
                feat_X.append(onset_vec)
                feat_pre.append(onset_pre)
                feat_npost.append(len(post_vecs))
                for k in (1, 2, 3):
                    src = post_vecs[k - 1] if len(post_vecs) >= k else (
                        post_vecs[-1] if post_vecs else onset_vec)
                    feat_post[k].append(src)
            traces.append({
                "id": q["id"], "pool": q.get("pool", tag or "?"),
                "n_q_chunks": len(chunks),
                "onset_chunk": onset_chunk, "no_speak": no_speak,
                "answer_text": "".join(texts).strip(),
                "eot_seen": eot_seen, "n_ans_chunks": n_ans,
                "carrier_ended": car_ended})
            print(f"  [{qi}] {q['id']} onset={onset_chunk}"
                  f"/{len(chunks)}q no_speak={no_speak} eot={eot_seen} "
                  f"ans={''.join(texts).strip()[:60]!r}", flush=True)
    finally:
        h.remove()

    sfx = "smoke" if shard_id < 0 else f"shard{max(shard_id, 0)}"
    with open(f"{OUT}_{tag}_traces.jsonl.{sfx}", "a",
              encoding="utf-8") as fh:
        for r in traces:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if feat_ids:
        np.savez_compressed(f"{OUT}_{tag}_feats.{sfx}.npz",
                            ids=np.array(feat_ids), X=np.stack(feat_X),
                            X_pre=np.stack(feat_pre),
                            X_k1=np.stack(feat_post[1]),
                            X_k2=np.stack(feat_post[2]),
                            X_k3=np.stack(feat_post[3]),
                            n_post=np.array(feat_npost))
    gate_data.commit()
    n_ns = sum(t["no_speak"] for t in traces)
    print(f">>> shard{shard_id}: {len(traces)} traces, "
          f"{n_ns} no_speak, {len(feat_ids)} feats", flush=True)
    return [t["id"] for t in traces]


judge_img = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("openai", "pandas", "pyarrow")
             .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
             .add_local_file(_APP_PY, "/root/modal_app.py"))


@app.function(image=judge_img, volumes={DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 60)
def judge_native(tag: str, pool: str = "frozen"):
    """gpt-5.4-mini judge over the native answers of one dump tag ->
    /data/frozen_native_{tag}_judged.parquet (id, answer, adequate,
    escalate_label). Only judges rows missing from the parquet
    (8au lesson: never re-judge existing rows)."""
    import asyncio
    import glob as _glob

    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    qfile, _ = FEAT_POOLS[pool]
    qs = {q["id"]: q for q in
          (json.loads(x) for x in open(qfile, encoding="utf-8")
           if x.strip())}
    rows = {}
    for p in sorted(_glob.glob(f"{OUT}_{tag}_traces.jsonl.shard*")):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                rows[r["id"]] = r
    out_p = f"{OUT}_{tag}_judged.parquet"
    old = (pd.read_parquet(out_p)
           if os.path.exists(out_p) else pd.DataFrame(columns=["id"]))
    # rows whose earlier judge call ERRORed (429 etc.) are retried; a
    # row with a real verdict is never re-judged
    if "adequate" in old:
        old = old[old["adequate"].notna()]
    have = set(old["id"])
    todo = []
    for r in rows.values():
        if r["id"] in have or r["id"] not in qs:
            continue
        todo.append({"id": r["id"], "query": qs[r["id"]]["query"],
                     "reference_answer": qs[r["id"]].get(
                         "reference_answer"),
                     "answer": r.get("answer_text") or "",
                     "no_speak": r.get("no_speak"),
                     "eot_seen": r.get("eot_seen")})
    print(f">>> judge_native[{tag}]: {len(rows)} traces, "
          f"{len(have)} already judged, {len(todo)} to judge")
    if todo:
        import time as _time
        judged = []
        pending = todo
        for attempt in range(6):          # 429 backoff passes
            got = asyncio.run(escalate.judge_many(pending, concurrency=4))
            judged += [r for r in got if r["adequate"] is not None]
            pending = [r for r in got if r["adequate"] is None]
            if not pending:
                break
            print(f"    pass {attempt}: {len(pending)} errored — "
                  f"sleeping 60s then retrying", flush=True)
            _time.sleep(60)
        judged += pending                  # keep any final errors
        new = pd.concat([old, pd.DataFrame(judged)], ignore_index=True)
        new.to_parquet(out_p)
        gate_data.commit()
        ok = [r for r in judged if r["adequate"] is not None]
        print(f">>> judged {len(judged)} ({len(ok)} ok-parse); "
              f"native local-correct rate "
              f"{sum(r['adequate'] for r in ok) / max(1, len(ok)):.3f}")
    return len(todo)


@app.local_entrypoint()
def judge_all(tags: str = "caliboff:frozen,expoff:expansion,"
              "exp2off:expansion2,freshoff:fresh,exp3off:expansion3"):
    """Sequential judge over several tags in ONE app (run with
    --detach so a dying local client cannot kill the remote pass;
    parallel judge apps trip the org 429 TPM/RPM limits)."""
    for pair in tags.split(","):
        tag, pool = pair.split(":")
        n = judge_native.remote(tag, pool)
        print(f">>> judge_all: {tag} done ({n} judged)", flush=True)


@app.function(image=judge_img, volumes={DATA: gate_data},
              secrets=[OPENAI], timeout=60 * 60)
def vb_judge_native(tag: str = "valpaca", pool: str = "valpaca"):
    """VoiceBench 1-5 judge (official gpt-4o-mini prompt, ported from
    modal_bench._score_many) over native answers ->
    /data/frozen_native_{tag}_vb.parquet."""
    import asyncio
    import glob as _glob
    import re as _re

    import pandas as pd
    sys.path.insert(0, "/workspace/gate")
    import escalate

    qfile, _ = FEAT_POOLS[pool]
    qs = {q["id"]: q for q in
          (json.loads(x) for x in open(qfile, encoding="utf-8")
           if x.strip())}
    rows = {}
    for p in sorted(_glob.glob(f"{OUT}_{tag}_traces.jsonl.shard*")):
        for ln in open(p, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                rows[r["id"]] = r
    out_p = f"{OUT}_{tag}_vb.parquet"
    old = (pd.read_parquet(out_p)
           if os.path.exists(out_p) else pd.DataFrame(columns=["id"]))
    have = set(old["id"])
    todo = [{"id": r["id"], "query": qs[r["id"]]["query"],
             "answer": r.get("answer_text") or ""}
            for r in rows.values()
            if r["id"] not in have and r["id"] in qs]
    print(f">>> vb_judge[{tag}]: {len(todo)} to score")
    if not todo:
        return 0
    client = escalate._async_client()
    sem = asyncio.Semaphore(3)

    async def one(r):
        prompt = (VB_META_PROMPT_OPEN
                  .replace("{prompt}", str(r["query"]))
                  .replace("{response}", str(r["answer"])))
        r["score"] = None
        for _ in range(6):
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=VB_JUDGE_MODEL, max_tokens=1024,
                        frequency_penalty=0, presence_penalty=0,
                        messages=[
                            {"role": "system", "content":
                             "You are a helpful assistant who tries to "
                             "help answer the user's question."},
                            {"role": "user", "content": prompt}],
                        user=escalate.USER_ID)
                    txt = (resp.choices[0].message.content or "").strip()
                    m = _re.search(r"\d+", txt)
                    v = int(m.group()) if m else None
                    if v is not None and 1 <= v <= 5:
                        r["score"] = v
                        return r
                except Exception:
                    await asyncio.sleep(3)
        return r

    async def run():
        return await asyncio.gather(*(one(r) for r in todo))

    judged = asyncio.run(run())
    new = pd.concat([old, pd.DataFrame(judged)], ignore_index=True)
    new.to_parquet(out_p)
    gate_data.commit()
    ok = [r["score"] for r in judged if r["score"]]
    print(f">>> scored {len(ok)}/{len(judged)}; native local VB mean "
          f"{sum(ok) / max(1, len(ok)):.2f}")
    return len(judged)


@app.function(image=util_img, volumes={DATA: gate_data}, timeout=60 * 5)
def _read_qfile(qfile: str, split: str = "") -> list:
    qs = [json.loads(x) for x in open(qfile, encoding="utf-8")
          if x.strip()]
    return [q for q in qs if q.get("split") == split] if split else qs


@app.local_entrypoint()
def run_native(pool: str = "frozen", workers: int = 4, limit: int = 0,
               split: str = "", tag: str = "", temp: float = 0.0,
               carrier: str = "", official: int = 0, postcut: int = 0):
    assert tag, "pass --tag (calib/exp/exp2/test/<pool>)"
    assert not postcut or carrier, "postcut needs --carrier"
    qfile, audio_dir = FEAT_POOLS[pool]
    qs = _read_qfile.remote(qfile, split)
    if limit:
        qs = qs[:limit]
        workers = 1
    shards = [qs[i::workers] for i in range(workers)]
    print(f">>> native dump [{pool}/{split or 'all'}] tag={tag} "
          f"temp={temp or 'default'} carrier={carrier or '-'}"
          f"{' postcut' if postcut else ''}: "
          f"{len(qs)} queries, {workers} workers")
    done = list(native_shard.starmap(
        [(shards[i], i if not limit else -1, tag, audio_dir, temp,
          carrier, official, postcut) for i in range(workers)]))
    print(f">>> complete: {sum(len(d) for d in done)} traces")
