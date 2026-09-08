"""Relay formatters for the expert->talker handoff (Exp-2, 2026-09-08).

clean_expert_v2 is the v2 formatter moved verbatim out of
modal_native_bench2.live_shard so the bench and offline diagnostics
share one source (bench2 imports this module from /workspace/gate).

format_spoken_v3 is the conservative candidate (plan Exp-2 candidate
2): keep the explicit conclusion and the required steps / quantities /
list items, strip only display-only syntax (markdown emphasis,
headings, links, code fences, LaTeX delimiters, table pipes) while
NEVER deleting code content, comparison operators, underscores, or
answer-bearing table cells. Length stays bounded (max_chars + 280
worst case, same grace as v2); when over budget it drops rationale
from the FRONT so the conclusion survives. Chinese text without
sentence spacing falls back to tail-keep truncation (v2 kept the head
and could drop the conclusion).
"""
import re

_SENT_SPLIT = re.compile(
    r"(?<!\b[A-Z])(?<!\b[A-Z][a-z])(?<!\bU\.S)(?<!\bDr)(?<!\bMr)(?<!\bMrs)"
    r"(?<!\bSt)(?<!\bNo)(?<=[.!?])\s+(?=[A-Z0-9\u4e00-\u9fff])")

_ANSWER_CUE = re.compile(
    r"(?i)(final answer|the answer is|answer\s*[::]|"
    r"correct (?:option|answer)|答案|最终答案|所以答案)")


def clean_expert_v2(txt, max_chars=400):
    """Expert markdown -> one spoken paragraph: strip emphasis/links/
    tables, flatten bullets into a comma list, keep whole sentences
    (abbreviation-aware) up to max_chars.
    v2 relay-loss fixes: fenced code blocks are dropped (code read
    aloud is garbage on the delivered channel), and the FINAL
    sentence is always kept (reasoning answers put the conclusion
    last; the v1 head-only cap cut it off)."""
    t = str(txt)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)            # code blocks
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)          # [text](url)
    t = re.sub(r"\(\s*https?://[^)]*\)", "", t)              # bare (url)
    t = re.sub(r"^\s*\|.*\|\s*$", " ", t, flags=re.M)         # table rows
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)            # headings
    t = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", ", ", t, flags=re.M)   # bullets
    t = re.sub(r"[*_`>]+", "", t)
    t = re.sub(r"\s*\n+\s*", " ", t)
    t = re.sub(r"\s*,\s*,+", ", ", t)
    t = re.sub(r":\s*,\s*", ": ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,")
    sents = _SENT_SPLIT.split(t)
    out = ""
    for se in sents:
        if out and len(out) + 1 + len(se) > max_chars:
            break
        out = (out + " " + se).strip()
    last = sents[-1].strip() if sents else ""
    if last and not out.endswith(last) \
            and len(out) + len(last) < max_chars + 200:
        out = (out + " " + last).strip()      # keep the conclusion
    if len(out) > max_chars + 280:
        out = out[:max_chars].rsplit(" ", 1)[0] + "."
    return out or t[:max_chars]


def _table_row_cells(m):
    cells = [c.strip() for c in m.group(0).strip().strip("|").split("|")]
    kept = ", ".join(c for c in cells if c)
    return (kept + ".") if kept else ""


def _strip_display_syntax(t):
    t = str(t)
    # code fences: drop the fence lines (incl. language tag), KEEP content
    t = re.sub(r"^\s*```[\w+-]*\s*$", "", t, flags=re.M)
    t = t.replace("```", " ")
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)           # [text](url)
    t = re.sub(r"\(\s*https?://[^)]*\)", "", t)              # bare (url)
    # LaTeX display syntax: delimiters and common display macros only;
    # the mathematical content (numbers, = < >) is kept
    t = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", t)
    t = (t.replace("\\times", "×").replace("\\cdot", "×")
          .replace("\\div", "÷"))
    t = re.sub(r"\\\[|\\\]|\\\(|\\\)", " ", t)
    # tables: separator rows die, data rows read out their cells
    t = re.sub(r"^\s*\|[\s:|-]+\|\s*$", "", t, flags=re.M)
    t = re.sub(r"^\s*\|.*\|\s*$", _table_row_cells, t, flags=re.M)
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)          # headings
    t = re.sub(r"^> ", "", t, flags=re.M)                     # blockquotes
    # paired emphasis markers only — a stray * _ ` survives
    t = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", t)
    t = re.sub(r"__([^_\n]+)__", r"\1", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\1", t)
    t = re.sub(r"`([^`\n]+)`", r"\1", t)
    # bullet / numbered-list markers: marker dies, item text stays
    t = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"\s*\n+\s*", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def format_spoken_v3(txt, max_chars=400):
    t = _strip_display_syntax(txt)
    sents = [s for s in _SENT_SPLIT.split(t) if s.strip()]
    if not sents:
        return t[:max_chars]
    keep = [False] * len(sents)
    keep[-1] = True                        # conclusion-last answers
    for i, s in enumerate(sents):
        if _ANSWER_CUE.search(s):
            keep[i] = True
    budget = sum(len(s) + 1 for i, s in enumerate(sents) if keep[i])
    for i, s in enumerate(sents):          # rationale, in order
        if not keep[i] and budget + len(s) + 1 <= max_chars:
            keep[i] = True
            budget += len(s) + 1
    kept = [s for i, s in enumerate(sents) if keep[i]]
    while len(kept) > 1 and len(" ".join(kept)) > max_chars + 280:
        kept.pop(0)                        # rationale dies, answer stays
    out = " ".join(kept).strip()
    if len(out) > max_chars + 280:
        out = out[-(max_chars + 280):].lstrip()
    return out or t[:max_chars]
