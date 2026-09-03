import glob
import os
from pathlib import Path
import sys

import torch

ROOT = os.environ.get(
    "NVDA_ROOT",
    str(Path(__file__).resolve().parents[2] / "demo"),
)
DATA_DIR = os.environ.get("NVDA_DATA", f"{ROOT}/data")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.environ.get("NVDA_NEMO_ROOT", f"{ROOT}/nemo-speech"))
from nvda_replay import _load_model, SYS_PROMPT
from nemo.collections.speechlm2.inference.utils.offline_voicechat import encode_system_prompt, run_offline_inference
import librosa
m = _load_model()
wav = sorted(glob.glob(f"{DATA_DIR}/audio_expansion/*.wav"))[0]
au, _ = librosa.load(wav, sr=16000, mono=True)
sig = torch.zeros(1, len(au) + 12*16000); sig[0, :len(au)] = torch.tensor(au)
sig = sig.cuda(); lens = torch.tensor([sig.shape[1]], device="cuda")
pt, pl = encode_system_prompt(m, SYS_PROMPT, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    q = torch.zeros(1, len(au), device="cuda"); q[0] = torch.tensor(au, device="cuda")
    nf = int(m.stt_model.perception(input_signal=q, input_signal_length=torch.tensor([len(au)], device="cuda"))[1][0])
    r = run_offline_inference(m, input_signal=sig, input_signal_lens=lens, prompt_tokens=pt, prompt_token_lens=pl)
print("keys:", sorted(r.keys()))
print("prompt_len:", int(pl[0]), "| query_frames:", nf, "| t_end:", int(pl[0])+nf)
tt = r.get("tokens_text")
if tt is not None:
    tt = tt[0].tolist()
    pad = m.stt_model.text_pad_id if hasattr(m.stt_model, "text_pad_id") else None
    print("tokens_text len:", len(tt), "| text_pad_id:", pad)
    import collections
    print("most common token:", collections.Counter(tt).most_common(3))
    nz = [i for i, t in enumerate(tt) if t != tt[0]]
    print("first 30 tokens:", tt[:30])
    onset = next((i for i, t in enumerate(tt) if t not in (pad, 0)), None) if pad is not None else None
    print("first non-pad frame:", onset)
    win = [i for i, t in enumerate(tt) if t not in (tt[0],)][:12]
    print("first 12 non-background frames:", win)
print("text:", r["text"][0][:120])
