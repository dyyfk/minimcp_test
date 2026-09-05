"""Floor-act calibration stims (8bh): the stop-word false-fire fix.

Problem (user, 2026-09-01): floor-control utterances ("stop", "停",
"别说了", backchannels) hit the same listen->speak commit as real
questions, the failure probe scores them out-of-distribution, and the
gate escalates "stop" to gpt-5.5.

Fix: a second linear head on the SAME L22 read — info-seeking vs
floor-management act — trained on the existing 2310 native question
features (positives) + this TTS'd floor-utterance inventory
(negatives). Escalation condition becomes act=info AND P(fail)>=thr.
The failure probe and its validated calibration stay untouched.

Inventory: hand-inventoried floor-management utterances (stop/hold
commands, backchannels, acknowledgments, fillers; en+zh), following
the 8ba stim protocol (behavioral stims, not an eval pool). Two TTS
voices for variety.

Steps:
  modal run modal_flooract.py::make_stims          # jsonl + wavs -> volume
  modal run modal_native_dump.py::run_native --pool flooract \
      --tag flooract --workers 2                   # features
  (then scripts/24_act_probe.py locally)
"""
import json
import os

import modal

app = modal.App("flooract-stims")
gate_data = modal.Volume.from_name("gate-data")
DATA = "/data"

from modal_app import OPENAI  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_PY = os.path.join(_HERE, "modal_app.py")

img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("openai", "librosa", "soundfile", "numpy")
       .add_local_dir(os.path.join(_HERE, "src"), "/workspace/gate")
       .add_local_file(_APP_PY, "/root/modal_app.py"))

# stop / hold commands
STOPS = ["Stop.", "Stop!", "Wait, stop.", "Okay stop.", "Please stop.",
         "Stop talking.", "Hold on.", "Hold on a second.", "Wait.",
         "Wait a moment.", "Hang on.", "One second.", "Just a second.",
         "Shush.", "Quiet please.", "That's enough.", "Enough.",
         "Never mind.", "Forget it.", "Cancel that.",
         "停。", "停下。", "别说了。", "别讲了。", "先别说。", "等等。",
         "等一下。", "稍等。", "打住。", "够了。", "算了。", "行了行了。",
         "先停一下。", "你先别说话。", "安静。", "别念了。"]
# backchannels / continuers
BCS = ["Mm-hm.", "Uh-huh.", "Yeah.", "Yeah yeah.", "Okay.", "Oh okay.",
       "Right.", "Sure.", "I see.", "Got it.", "Go on.", "Keep going.",
       "Interesting.", "Oh wow.", "Really?", "Makes sense.", "Mm.",
       "嗯。", "嗯嗯。", "哦。", "好的。", "对。", "对对对。", "是的。",
       "继续。", "接着说。", "有道理。", "原来如此。", "哦这样啊。", "行。"]
# acknowledgments / closers / social
ACKS = ["Thanks.", "Thank you.", "Thanks a lot.", "Great, thanks.",
        "That's all.", "That's all I needed.", "Perfect.", "Sounds good.",
        "Alright then.", "Okay bye.", "Goodbye.", "See you.",
        "谢谢。", "多谢。", "好的谢谢。", "没事了。", "就这样吧。",
        "可以了。", "明白了。", "再见。", "拜拜。", "辛苦了。"]
# fillers / hesitations (speaker holds THEIR floor — not a query)
FILLS = ["Um...", "Uh...", "Hmm, let me think.", "Well...",
         "How do I put this...", "Let me see...",
         "呃……", "嗯……让我想想。", "怎么说呢……", "我想想啊。"]

CATS = [("stopcmd", STOPS), ("backch", BCS), ("ack", ACKS),
        ("filler", FILLS)]
VOICES = ["alloy", "echo"]

# 8ch: STOP-PREFIXED question compounds — a live barge-in follow-up
# usually arrives as "Stop, stop. <new question>" in one breath, and
# the stop prefix (plus the post-cut context) drags the act read below
# threshold: the gate ruled the user's real Apple question a floor
# turn (P(info)=.336) and the escalation never launched. These are
# POSITIVES (behavioral stims, 8ba protocol; question halves follow
# the REQQ hand-inventory style, not an eval pool).
STOPQ = ["Stop, stop. What is the stock price of Apple today?",
         "Stop. What's the weather in Tokyo tomorrow?",
         "Wait, stop. How tall is Mount Everest?",
         "Okay stop. Who wrote The Old Man and the Sea?",
         "Stop talking. What is the capital of Mongolia?",
         "Hold on. What's 15 percent of 260?",
         "Wait wait. When was the Eiffel Tower built?",
         "Stop, stop. Can you check Nvidia's stock price right now?",
         "Hang on. What is the population of Brazil?",
         "That's enough. What's the boiling point of ethanol?",
         "Never mind that. Who won the World Cup in 2018?",
         "Okay okay stop. What is the largest moon of Saturn?",
         "停停。今天苹果的股价是多少？",
         "别说了。明天上海的天气怎么样？",
         "等等。珠穆朗玛峰有多高？",
         "先别说。英伟达现在的股价是多少？",
         "打住。巴西有多少人口？",
         "停一下。二〇一八年世界杯是谁夺冠的？",
         "行了行了。乙醇的沸点是多少？",
         "别讲了。《老人与海》是谁写的？"]


@app.function(image=img, volumes={DATA: gate_data}, secrets=[OPENAI],
              timeout=60 * 40)
def make_stopq():
    import sys
    import librosa
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, "/workspace/gate")
    import escalate

    os.makedirs(f"{DATA}/stopq_audio", exist_ok=True)
    rows, n = [], 0
    cli = escalate._client()
    for txt in STOPQ:
        for voice in VOICES:
            qid = f"sq{len(rows):04d}"
            wav_p = f"{DATA}/stopq_audio/{qid}.wav"
            if not os.path.exists(wav_p):
                r = cli.audio.speech.create(
                    model="tts-1", voice=voice, input=txt,
                    response_format="wav")
                open("/tmp/sq.wav", "wb").write(r.content)
                au, _ = librosa.load("/tmp/sq.wav", sr=16000, mono=True)
                sf.write(wav_p, au.astype(np.float32), 16000)
                n += 1
            rows.append({"id": qid, "pool": "stopq", "query": txt,
                         "reference_answer": None, "split": ""})
    with open(f"{DATA}/queries_stopq.jsonl", "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> {len(rows)} stop-prefixed question stims ({n} new)")
    return len(rows)

# 8bj: REQUEST-phrased questions — live speech phrases queries as
# requests ("can you check…") and the benchmark-question-trained act
# probe scores them lower; these are POSITIVES for the act refit.
REQQ = ["Can you check for me what Nvidia is trading at right now?",
        "Could you look up the capital of Mongolia?",
        "Can you tell me how tall Mount Everest is?",
        "I want to know who wrote The Old Man and the Sea.",
        "Help me figure out what 15 percent of 260 is.",
        "Would you mind checking when the Eiffel Tower was built?",
        "Look up the boiling point of ethanol for me.",
        "I need to know the population of Brazil.",
        "Tell me the speed of light, please.",
        "Do you happen to know who won the World Cup in 2018?",
        "Can you find out what the largest moon of Saturn is?",
        "Please check how many ounces are in a kilogram.",
        "帮我查一下英伟达现在的股价。",
        "帮我看看明天上海的天气怎么样。",
        "你能告诉我珠穆朗玛峰有多高吗？",
        "帮我算算二百六的百分之十五是多少。",
        "麻烦查一下巴西有多少人口。",
        "你知道二〇一八年世界杯是谁夺冠吗？",
        "帮我查查乙醇的沸点。",
        "我想知道《老人与海》是谁写的。"]


@app.function(image=img, volumes={DATA: gate_data}, secrets=[OPENAI],
              timeout=60 * 40)
def make_reqq():
    import sys
    import librosa
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, "/workspace/gate")
    import escalate

    os.makedirs(f"{DATA}/reqq_audio", exist_ok=True)
    rows, n = [], 0
    cli = escalate._client()
    for txt in REQQ:
        for voice in VOICES:
            qid = f"rq{len(rows):04d}"
            wav_p = f"{DATA}/reqq_audio/{qid}.wav"
            if not os.path.exists(wav_p):
                r = cli.audio.speech.create(
                    model="tts-1", voice=voice, input=txt,
                    response_format="wav")
                open("/tmp/rq.wav", "wb").write(r.content)
                au, _ = librosa.load("/tmp/rq.wav", sr=16000, mono=True)
                sf.write(wav_p, au.astype(np.float32), 16000)
                n += 1
            rows.append({"id": qid, "pool": "reqq", "query": txt,
                         "reference_answer": None, "split": ""})
    with open(f"{DATA}/queries_reqq.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> {len(rows)} request-phrased stims ({n} new)")
    return len(rows)


@app.function(image=img, volumes={DATA: gate_data}, secrets=[OPENAI],
              timeout=60 * 40)
def make_stims():
    import sys
    import librosa
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, "/workspace/gate")
    import escalate

    os.makedirs(f"{DATA}/flooract_audio", exist_ok=True)
    rows = []
    n = 0
    cli = escalate._client()
    for cat, texts in CATS:
        for ti, txt in enumerate(texts):
            for voice in VOICES:
                qid = f"fa{len(rows):04d}"
                wav_p = f"{DATA}/flooract_audio/{qid}.wav"
                if not os.path.exists(wav_p):
                    r = cli.audio.speech.create(
                        model="tts-1", voice=voice, input=txt,
                        response_format="wav")
                    open("/tmp/fa.wav", "wb").write(r.content)
                    au, _ = librosa.load("/tmp/fa.wav", sr=16000,
                                         mono=True)
                    sf.write(wav_p, au.astype(np.float32), 16000)
                    n += 1
                rows.append({"id": qid, "pool": f"flooract-{cat}",
                             "query": txt, "reference_answer": None,
                             "split": ""})
    with open(f"{DATA}/queries_flooract.jsonl", "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gate_data.commit()
    print(f">>> {len(rows)} stims ({n} newly synthesized), "
          f"cats: " + ", ".join(f"{c}:{len(t) * len(VOICES)}"
                                for c, t in CATS))
    return len(rows)
