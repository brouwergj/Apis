# scan_repos_v.py
#
# Adds --mode filtering to v3 classification output.
#
# Usage:
#   python scan_repos_v4.py <repo_root> [--mode index|core|core+estimator|core-min]
#
# Outputs under <repo_root>/controller_scan/:
#   controller_scan_results_<mode>.{csv,md,json}

import os
import re
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

EXTENSIONS = {".c", ".h", ".cpp", ".hpp"}

INCLUDE_PREFIXES = [
    "src/modules/src",
    "src/modules/interface",
    "src/utils/src/pid.c",
    "src/utils/interface/pid.h",
]

EXCLUDE_PREFIXES = [
    "src/lib",
    "src/drivers",
    "src/hal",
    "src/deck",
    "examples",
    "scripts",
    "test",
]

CATEGORY_RULES: List[Tuple[str, List[str], int]] = [
    ("CORE_CONTROLLER", [
        r"^src/modules/src/stabilizer\.c$",
        r"^src/modules/src/controller/controller\.c$",
        r"^src/modules/src/controller/controller_(pid|mellinger|indi|lee|brescianini)\.c$",
        r"^src/modules/src/controller/(position_controller_pid|attitude_pid_controller|position_controller_indi)\.c$",
        r"^src/modules/interface/stabilizer_types\.h$",
        r"^src/modules/interface/controller/.*\.h$",
        r"^src/utils/src/pid\.c$",
        r"^src/utils/interface/pid\.h$",
    ], 60),

    ("ESTIMATOR", [
        r"^src/modules/src/estimator/.*\.(c|h)$",
        r"^src/modules/interface/estimator/.*\.h$",
        r"^src/modules/src/kalman_core/kalman_core\.c$",
        r"^src/modules/interface/kalman_core/kalman_core\.h$",
        r"^src/modules/interface/kalman_core/kalman_core_params_defaults\.h$",
    ], 50),

    ("MEASUREMENT_MODEL", [
        r"^src/modules/src/kalman_core/mm_.*\.c$",
        r"^src/modules/interface/kalman_core/mm_.*\.h$",
    ], 35),

    ("SETPOINT_INTERFACE", [
        r"^src/modules/src/commander\.c$",
        r"^src/modules/interface/commander\.h$",
        r"^src/modules/src/crtp_commander.*\.c$",
        r"^src/modules/interface/crtp_commander.*\.h$",
        r"^src/modules/src/crtp_commander_generic\.c$",
        r"^src/modules/src/crtp_commander_rpyt\.c$",
    ], 25),

    ("PARAM_LOG_INFRA", [
        r"^src/modules/(src|interface)/param.*\.(c|h)$",
        r"^src/modules/(src|interface)/log\.(c|h)$",
        r"^src/modules/(src|interface)/console\.h$",
    ], 10),

    ("MOTOR_MIXING_OUT_OF_SCOPE", [
        r"^src/modules/src/power_distribution_.*\.c$",
        r"^src/modules/interface/power_distribution\.h$",
    ], 20),

    ("COLLISION_AVOIDANCE_OUT_OF_SCOPE", [
        r"^src/modules/src/collision_avoidance\.c$",
        r"^src/modules/interface/collision_avoidance\.h$",
    ], 15),

    ("LOCALIZATION_PIPELINE_OUT_OF_SCOPE", [
        r"^src/modules/src/lighthouse/.*\.(c|h)$",
        r"^src/modules/interface/lighthouse/.*\.h$",
        r"^src/modules/src/tdoa.*\.(c|h)$",
        r"^src/modules/src/outlierfilter/.*\.(c|h)$",
        r"^src/modules/interface/outlierfilter/.*\.h$",
        r"^src/modules/src/crtp_localization_service\.c$",
        r"^src/modules/interface/crtp_localization_service\.h$",
        r"^src/modules/src/peer_localization\.c$",
        r"^src/modules/interface/peer_localization\.h$",
    ], 10),
]

CONTENT_KEYWORDS = [
    "stabilizer", "controller", "attitude", "position", "velocity", "rate",
    "thrust", "yawrate", "stateEstimate", "kalman", "ekf", "complementary",
    "ukf", "pidInit", "pidUpdate", "PID", "PARAM", "param",
]

STRONG_CONTENT_REGEXES = [
    r"\bstateEstimate\b",
    r"\bcontroller(State|Init|Update)?\b",
    r"\bstabilizer\b",
    r"\bpid(Update|Init)\b",
]

MAX_EVIDENCE_ITEMS = 25


def norm_rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def is_included(path: Path, root: Path) -> bool:
    rel = norm_rel(path, root)
    for ex in EXCLUDE_PREFIXES:
        if rel.startswith(ex):
            return False
    for inc in INCLUDE_PREFIXES:
        if rel.startswith(inc):
            return True
    return False


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def classify_by_path(rel: str) -> Tuple[str, int, List[str]]:
    evidence: List[str] = []
    best_cat = "UNCLASSIFIED"
    best_score = 0

    for cat, regexes, bonus in CATEGORY_RULES:
        for rx in regexes:
            if re.search(rx, rel, flags=re.IGNORECASE):
                score = bonus
                if score > best_score:
                    best_cat = cat
                    best_score = score
                    evidence = [f"path_regex:{rx}"]
                break

    return best_cat, best_score, evidence


def score_by_content(text: str) -> Tuple[int, List[str]]:
    score = 0
    evidence: List[str] = []

    for kw in CONTENT_KEYWORDS:
        if kw in text:
            score += 1
            if len(evidence) < MAX_EVIDENCE_ITEMS:
                evidence.append(f"kw:{kw}")

    for rx in STRONG_CONTENT_REGEXES:
        if re.search(rx, text, flags=re.IGNORECASE):
            score += 6
            if len(evidence) < MAX_EVIDENCE_ITEMS:
                evidence.append(f"rx:{rx}")

    return score, evidence


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def confidence_from_score(score: int) -> str:
    if score >= 85:
        return "HIGH"
    if score >= 55:
        return "MEDIUM"
    return "LOW"


def scan_repo(root: Path) -> List[Dict]:
    results: List[Dict] = []

    for path in root.rglob("*"):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        if not is_included(path, root):
            continue

        rel = norm_rel(path, root)
        text = read_text_safe(path)

        category, base_score, path_ev = classify_by_path(rel)
        content_score, content_ev = score_by_content(text)

        score = base_score + clamp(content_score, 0, 40)

        if category == "UNCLASSIFIED" and any(ev.startswith("rx:") for ev in content_ev):
            category = "REVIEW_CANDIDATE"
            score += 10

        if score < 8:
            continue

        results.append({
            "relative_path": rel,
            "path": str(path),
            "category": category,
            "confidence": confidence_from_score(score),
            "score": score,
            "evidence": "; ".join((path_ev + content_ev)[:MAX_EVIDENCE_ITEMS]),
        })

    results.sort(key=lambda r: (r["category"], -r["score"], r["relative_path"]))
    return results


# -----------------------------
# MODE FILTERING
# -----------------------------

def apply_mode(results: List[Dict], mode: str) -> List[Dict]:
    mode = mode.strip().lower()

    if mode == "index":
        return results

    if mode == "core":
        return [r for r in results if r["category"] == "CORE_CONTROLLER"]

    if mode in {"core+estimator", "core_estimator"}:
        return [r for r in results if r["category"] in {"CORE_CONTROLLER", "ESTIMATOR"}]

    if mode == "core-min":
        # absolute minimum set to start a faithful port:
        # stabilizer + controller dispatch + PID cascade + pid helper + types
        keep = {
            "src/modules/src/stabilizer.c",
            "src/modules/src/controller/controller.c",
            "src/modules/src/controller/controller_pid.c",
            "src/modules/src/controller/position_controller_pid.c",
            "src/modules/src/controller/attitude_pid_controller.c",
            "src/modules/interface/stabilizer_types.h",
            "src/modules/interface/controller/controller.h",
            "src/modules/interface/controller/controller_pid.h",
            "src/modules/interface/controller/position_controller.h",
            "src/modules/interface/controller/attitude_controller.h",
            "src/utils/src/pid.c",
            "src/utils/interface/pid.h",
        }
        return [r for r in results if r["relative_path"] in keep]

    raise SystemExit(f"Unknown --mode '{mode}'. Valid: index, core, core+estimator, core-min")


def write_csv(results: List[Dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "confidence", "score", "relative_path", "path", "evidence"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def write_json(results: List[Dict], out_path: Path, repo_root: Path, mode: str) -> None:
    payload = {
        "repo": str(repo_root),
        "mode": mode,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "total": len(results),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(results: List[Dict], out_path: Path, repo_root: Path, mode: str) -> None:
    grouped: Dict[str, List[Dict]] = {}
    for r in results:
        grouped.setdefault(r["category"], []).append(r)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Crazyflie Control Pipeline Scan (v4))\n\n")
        f.write(f"- Repo: `{repo_root}`\n")
        f.write(f"- Mode: `{mode}`\n")
        f.write(f"- Generated: {datetime.utcnow().isoformat()} UTC\n")
        f.write(f"- Total matches: {len(results)}\n\n")

        f.write("## Summary\n\n")
        f.write("| Category | Count | Top score |\n")
        f.write("|---|---:|---:|\n")
        for cat in sorted(grouped.keys()):
            items = grouped[cat]
            top = max(i["score"] for i in items) if items else 0
            f.write(f"| `{cat}` | {len(items)} | {top} |\n")

        for cat in sorted(grouped.keys()):
            f.write(f"\n## `{cat}`\n\n")
            for r in grouped[cat]:
                f.write(f"- **{Path(r['relative_path']).name}** — {r['confidence']} (score={r['score']})  \n")
                f.write(f"  `{r['relative_path']}`  \n")
                if r["evidence"]:
                    f.write(f"  _{r['evidence']}_\n")
                f.write("\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scan_repos_v4.py <repo_root> [--mode index|core|core+estimator|core-min]")
        sys.exit(1)

    repo_root = Path(sys.argv[1]).resolve()

    mode = "index"
    if "--mode" in sys.argv:
        i = sys.argv.index("--mode")
        if i + 1 >= len(sys.argv):
            raise SystemExit("Missing value after --mode")
        mode = sys.argv[i + 1]

    print(f"\nScanning (classified v4): {repo_root}")
    print(f"Mode: {mode}\n")

    results_all = scan_repo(repo_root)
    results = apply_mode(results_all, mode)

    out_dir = repo_root / "controller_scan"
    out_dir.mkdir(exist_ok=True)

    safe_mode = mode.replace("+", "_").replace("-", "_")
    csv_path = out_dir / f"controller_scan_results_{safe_mode}.csv"
    md_path = out_dir / f"controller_scan_results_{safe_mode}.md"
    json_path = out_dir / f"controller_scan_results_{safe_mode}.json"

    write_csv(results, csv_path)
    write_markdown(results, md_path, repo_root, mode)
    write_json(results, json_path, repo_root, mode)

    counts: Dict[str, int] = {}
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    print("Category counts:")
    for cat in sorted(counts.keys()):
        print(f"  - {cat}: {counts[cat]}")

    print(f"\nTotal matched files: {len(results)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Markdown saved to: {md_path}")
    print(f"JSON saved to: {json_path}")
