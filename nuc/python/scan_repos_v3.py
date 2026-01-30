# scan_repos_v3.py
#
# Purpose:
#   Recursively scan a repo (e.g., crazyflie-firmware) and produce a *classified*
#   index of files relevant to the Crazyflie control pipeline.
#
# Output (written under <repo_root>/controller_scan/):
#   - controller_scan_results_v3.csv
#   - controller_scan_results_v3.md
#   - controller_scan_results_v3.json
#
# Usage:
#   python scan_repos_v3.py <repo_root>
#
# Notes:
#   - This script uses path-based gating + classification (core/controller/estimator/etc.)
#   - It adds a confidence score so you can sort/filter easily.

import os
import re
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# -----------------------------
# CONFIG
# -----------------------------

EXTENSIONS = {".c", ".h", ".cpp", ".hpp"}

# Path gating (relative to repo root)
# Keep this fairly broad; classification will narrow the "core" set.
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

# Classification rules (ordered; first strong match wins when categories conflict)
# category: (path_regexes, bonus_score)
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

    # Measurement models: part of estimator internals; usually OUT-OF-SCOPE for NUC mirroring.
    ("MEASUREMENT_MODEL", [
        r"^src/modules/src/kalman_core/mm_.*\.c$",
        r"^src/modules/interface/kalman_core/mm_.*\.h$",
    ], 35),

    # Inputs/setpoints & high-level planner-ish bits (useful for understanding interfaces; not controller math)
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

    # Explicitly out of scope for NUC controller replication
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

# Content keywords for scoring + "evidence"
CONTENT_KEYWORDS = [
    # core control concepts
    "stabilizer", "controller", "attitude", "position", "velocity", "rate", "thrust", "yawrate",
    # estimator concepts
    "stateEstimate", "kalman", "ekf", "complementary", "ukf",
    # pid helpers
    "pidInit", "pidUpdate", "PID",
    # params
    "PARAM", "param",
]

# Regex patterns that strongly suggest the file is part of the control spine
STRONG_CONTENT_REGEXES = [
    r"\bstateEstimate\b",
    r"\bcontroller(State|Init|Update)?\b",
    r"\bstabilizer\b",
    r"\bpid(Update|Init)\b",
]

MAX_EVIDENCE_ITEMS = 25  # keep outputs readable


# -----------------------------
# HELPERS
# -----------------------------

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
    """
    Returns (category, base_score, path_evidence).
    """
    evidence = []
    best_cat = "UNCLASSIFIED"
    best_score = 0

    for cat, regexes, bonus in CATEGORY_RULES:
        for rx in regexes:
            if re.search(rx, rel, flags=re.IGNORECASE):
                # prefer first match for deterministic results,
                # but allow higher-scoring categories to override.
                score = bonus
                if score > best_score:
                    best_cat = cat
                    best_score = score
                    evidence = [f"path_regex:{rx}"]
                break

    return best_cat, best_score, evidence


def score_by_content(text: str) -> Tuple[int, List[str]]:
    """
    Returns (content_score, evidence)
    """
    score = 0
    evidence: List[str] = []

    # keyword hits
    for kw in CONTENT_KEYWORDS:
        if kw in text:
            score += 1
            if len(evidence) < MAX_EVIDENCE_ITEMS:
                evidence.append(f"kw:{kw}")

    # strong regex hits
    for rx in STRONG_CONTENT_REGEXES:
        if re.search(rx, text, flags=re.IGNORECASE):
            score += 6
            if len(evidence) < MAX_EVIDENCE_ITEMS:
                evidence.append(f"rx:{rx}")

    return score, evidence


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


# -----------------------------
# SCAN
# -----------------------------

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

        # Composite score: path classification matters most
        # (content score can be noisy for generic files).
        score = base_score + clamp(content_score, 0, 40)

        # If it's UNCLASSIFIED but has strong content evidence, bump it to REVIEW
        if category == "UNCLASSIFIED" and any(ev.startswith("rx:") for ev in content_ev):
            category = "REVIEW_CANDIDATE"
            score += 10

        # If still basically no signal, skip it
        if score < 8:
            continue

        # Confidence buckets: easy to filter in CSV
        if score >= 85:
            confidence = "HIGH"
        elif score >= 55:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        evidence = (path_ev + content_ev)[:MAX_EVIDENCE_ITEMS]

        results.append({
            "relative_path": rel,
            "path": str(path),
            "category": category,
            "confidence": confidence,
            "score": score,
            "evidence": "; ".join(evidence),
        })

    # Stable ordering for diffs
    results.sort(key=lambda r: (r["category"], -r["score"], r["relative_path"]))
    return results


# -----------------------------
# OUTPUT
# -----------------------------

def write_csv(results: List[Dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "confidence", "score", "relative_path", "path", "evidence"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def write_json(results: List[Dict], out_path: Path, repo_root: Path) -> None:
    payload = {
        "repo": str(repo_root),
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "total": len(results),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(results: List[Dict], out_path: Path, repo_root: Path) -> None:
    # Group by category for readability
    grouped: Dict[str, List[Dict]] = {}
    for r in results:
        grouped.setdefault(r["category"], []).append(r)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Crazyflie Control Pipeline Scan (v3)\n\n")
        f.write(f"- Repo: `{repo_root}`\n")
        f.write(f"- Generated: {datetime.utcnow().isoformat()} UTC\n")
        f.write(f"- Total matches: {len(results)}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Category | Count | Top score |\n")
        f.write("|---|---:|---:|\n")
        for cat in sorted(grouped.keys()):
            items = grouped[cat]
            top = max(i["score"] for i in items) if items else 0
            f.write(f"| `{cat}` | {len(items)} | {top} |\n")

        # Details
        for cat in sorted(grouped.keys()):
            f.write(f"\n## `{cat}`\n\n")
            for r in grouped[cat]:
                f.write(f"- **{Path(r['relative_path']).name}** — {r['confidence']} (score={r['score']})  \n")
                f.write(f"  `{r['relative_path']}`  \n")
                if r["evidence"]:
                    f.write(f"  _{r['evidence']}_\n")
                f.write("\n")


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python scan_repos_v3.py <repo_root>")
        sys.exit(1)

    repo_root = Path(sys.argv[1]).resolve()
    print(f"\nScanning (classified v3): {repo_root}\n")

    results = scan_repo(repo_root)

    out_dir = repo_root / "controller_scan"
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "controller_scan_results_v3.csv"
    md_path = out_dir / "controller_scan_results_v3.md"
    json_path = out_dir / "controller_scan_results_v3.json"

    write_csv(results, csv_path)
    write_markdown(results, md_path, repo_root)
    write_json(results, json_path, repo_root)

    # Console summary
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
