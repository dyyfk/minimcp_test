"""Recompute nonnegative post-input server waiting time from ttfa-v3 logs.

Usage: python3 recompute_nonnegative_ttfa.py --results PATH --out-dir PATH
Raw logs and original signed summaries are never changed. This is a new
summary of an existing experiment, not a new serving run or correctness eval.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

POOLS = {
    "frozen": ("Internal", 240),
    "striviaqa": ("Speech TriviaQA", 250),
    "swebq": ("Speech Web Questions", 250),
    "sllama": ("Speech Llama Questions", 250),
    "sdqa": ("SD-QA", 200),
}
ARMS = ["local", "conservative", "balanced", "aggressive", "always"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    xs = sorted(values)
    if not xs:
        return None

    def quantile(p):
        k = (len(xs) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {"n": len(xs), "mean": sum(xs) / len(xs),
            "p50": quantile(.50), "p95": quantile(.95),
            "p99": quantile(.99), "min": xs[0], "max": xs[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--write-rows", action="store_true", help="Also write the per-attempt audit JSONL")
    args = ap.parse_args()
    root = args.results.resolve()
    out = args.out_dir.resolve()
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "Nonnegative post-input server TTFA (waiting time)",
        "formula": "max(0, ts.first_answer_pcm - ts.input_end)",
        "input_end": "Scheduled arrival of the final real input sample",
        "endpoint": "First answer PCM ready on the server: local if not fired, relay if fired",
        "validity": "status == completed with both required timestamps present",
        "early_response_policy": "Retain completed early-response sessions with waiting time 0; count separately",
        "failure_policy": "Retain failed attempts with null waiting time; report separately",
        "limitations": [
            "This changes the summary metric, not the original timestamps or experiment.",
            "Zero denotes answer audio was ready before input end; it does not establish a valid or correct answer.",
            "Candidate and stall audio are excluded; client playback was not measured.",
            "Timing-session answers are unjudged and cannot be paired with the paper's separate accuracy results.",
            "Logged timestamps have millisecond precision; direct subtraction can differ from the recorded signed value by 0.001 s.",
        ],
        "pools": {}, "source_files": [], "audit": {},
    }
    output_rows, all_source_paths, source_hashes = [], [], {}
    max_delta, endpoint_checks = 0.0, 0
    old_path = root.parent.parent / "summary_ttfa1_final.json"
    old = json.loads(old_path.read_text())
    old_hash = digest(old_path)

    for pool, (name, expected) in POOLS.items():
        po = {"name": name, "arms": {}, "paired_completed": {}}
        per_arm = {}
        pool_ids = None
        for arm in ARMS:
            paths = sorted((root / pool).glob(f"{arm}.jsonl.shard*"))
            assert paths, (pool, arm, "missing shards")
            records = []
            seen = set()
            for path in paths:
                file_hash = digest(path)
                source_hashes[path] = file_hash
                all_source_paths.append(path)
                summary["source_files"].append({"path": str(path.relative_to(root)), "sha256": file_hash})
                for line_no, line in enumerate(path.read_text().splitlines(), 1):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    assert (r["pool"], r["arm"], r["run_id"], r["protocol"], r["attempt"]) == (pool, arm, "ttfa1", "ttfa-v3", 1)
                    assert r["id"] not in seen, (pool, arm, r["id"], "duplicate")
                    seen.add(r["id"])
                    ts = r["ts"]
                    wait = signed = None
                    valid = r["status"] == "completed"
                    if valid:
                        first, end = ts["first_answer_pcm"], ts["input_end"]
                        assert first is not None and end is not None
                        assert math.isfinite(first) and math.isfinite(end)
                        selected = ts["relay_first_pcm"] if r["fired"] else ts["local_first_pcm"]
                        assert first == selected, (pool, arm, r["id"], "wrong audio endpoint")
                        assert first >= ts["feed_start"]
                        assert first <= ts["session_end"] + .0011
                        signed = round(first - end, 6)
                        assert r["ttfa_answer_s"] is not None
                        delta = abs(signed - r["ttfa_answer_s"])
                        assert delta <= .001001, (pool, arm, r["id"], "signed value mismatch")
                        max_delta = max(max_delta, delta)
                        wait = max(0.0, signed)
                        endpoint_checks += 1
                    result = {
                        "pool": pool, "arm": arm, "id": r["id"], "attempt": 1,
                        "status": r["status"], "error": r.get("error"), "fired": r["fired"],
                        "source_file": str(path.relative_to(root)), "source_line": line_no,
                        "input_end_s": ts.get("input_end"),
                        "first_answer_pcm_s": ts.get("first_answer_pcm"),
                        "recorded_signed_ttfa_s": r.get("ttfa_answer_s"),
                        "recomputed_signed_ttfa_s": signed,
                        "early_response": valid and signed < 0,
                        "nonnegative_ttfa_s": wait,
                    }
                    records.append(result)
            assert len(records) == expected, (pool, arm, len(records), expected)
            if pool_ids is None:
                pool_ids = seen
            else:
                assert seen == pool_ids, (pool, arm, "query IDs differ across arms")
            valid_rows = [r for r in records if r["nonnegative_ttfa_s"] is not None]
            signed_values = [r["recomputed_signed_ttfa_s"] for r in valid_rows]
            waiting = [r["nonnegative_ttfa_s"] for r in valid_rows]
            previous = old["pools"][pool]["arms"][arm]
            counts = dict(Counter(r["status"] for r in records))
            assert counts == previous["status_counts"]
            assert sum(r["early_response"] for r in records) == previous["n_early_response"]
            assert abs(stats(signed_values)["mean"] - previous["ttfa_answer"]["mean"]) <= .0011
            assert all(x >= 0 for x in waiting)
            po["arms"][arm] = {
                "n_attempts": len(records), "n_completed": len(valid_rows),
                "n_failed": len(records) - len(valid_rows), "status_counts": counts,
                "n_early_response": sum(r["early_response"] for r in records),
                "n_zero_wait": sum(x == 0 for x in waiting),
                "nonnegative_ttfa_s": stats(waiting),
                "signed_timestamp_difference_s": stats(signed_values),
            }
            per_arm[arm] = {r["id"]: r["nonnegative_ttfa_s"] for r in valid_rows}
            output_rows.extend(records)
        common = set.intersection(*(set(per_arm[a]) for a in ARMS))
        po["paired_completed"] = {
            "n_common_ids": len(common),
            "arms": {a: stats([per_arm[a][qid] for qid in sorted(common)]) for a in ARMS},
        }
        summary["pools"][pool] = po

    assert len(output_rows) == 5950
    assert all(digest(p) == source_hashes[p] for p in all_source_paths)
    assert digest(old_path) == old_hash
    summary["audit"] = {
        "n_attempts": len(output_rows), "n_completed": endpoint_checks,
        "n_failed": len(output_rows) - endpoint_checks,
        "n_early_response": sum(r["early_response"] for r in output_rows),
        "n_source_files": len(all_source_paths), "n_pool_arm_combinations": len(POOLS) * len(ARMS),
        "duplicate_attempts": 0, "missing_attempts": 0,
        "max_difference_from_recorded_signed_s": max_delta,
        "original_summary_path": old_path.name, "original_summary_sha256": old_hash,
        "raw_files_unchanged": True, "original_summary_unchanged": True,
        "endpoint_selection_checks_passed": True,
        "failed_rows_retain_null": all(r["nonnegative_ttfa_s"] is None for r in output_rows if r["status"] != "completed"),
        "nonnegative_values_only": True,
    }
    md = [
        "# TTFA 非负等待时间重算", "",
        "按现有 ttfa-v3 原始时间戳重算，不是重新运行模型。单位均为秒。", "",
        "计算：`max(0, 首个答案 PCM 就绪时间 − 输入预定结束时间)`。",
        "输入结束前已经输出答案音频的完成样本记 0 秒，并单独报告提前回答数。失败保留为空值，不计作 0 秒。",
        "均值和分位数包含所有完成样本，包括提前回答；这里没有删除负值对应的样本。", "",
        "这是输入结束后等待服务端答案音频就绪的统计，不是客户端听到音频的时间；提前回答不代表正确或适当，回答仍未评分。", "",
        "| 测试池 | 档位 | 尝试 | 完成 | 失败 | 提前回答 | 平均 TTFA | P50 | P95 | P99 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pool, po in summary["pools"].items():
        for arm, a in po["arms"].items():
            d = a["nonnegative_ttfa_s"]
            md.append(f"| {po['name']} | {arm} | {a['n_attempts']} | {a['n_completed']} | {a['n_failed']} | {a['n_early_response']} | {d['mean']:.3f} | {d['p50']:.3f} | {d['p95']:.3f} | {d['p99']:.3f} |")
    md += ["", "## 校验", "",
           f"共 {len(output_rows):,} 次尝试、{endpoint_checks:,} 次完成、{len(output_rows)-endpoint_checks} 次失败；没有重复或缺失尝试。",
           "逐条核对 local/relay 的答案音频端点；不使用 candidate/stall 音频。原始日志和旧汇总文件哈希保持不变。",
           "JSON 另含旧的带符号时间差统计，以及每池五个档位共同完成 query ID 上的配对统计。",
           "毫秒精度的时间戳相减与原记录最多相差 0.001 秒。", ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if args.write_rows:
        (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output_rows))
    (out / "report.md").write_text("\n".join(md))
    print(json.dumps({"audit": summary["audit"], "internal": summary["pools"]["frozen"]["arms"], "report": str(out / "report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
