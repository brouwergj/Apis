import os
import re
import csv
from pathlib import Path
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------

EXTENSIONS = {".c", ".h", ".cpp", ".hpp"}

# HARD path filters (relative to repo root)
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

PATH_KEYWORDS = [
    "controller",
    "stabilizer",
    "estimator",
    "pid",
    "mellinger",
    "indi",
]

CONTENT_KEYWORDS = [
    # control layers
    "position",
    "velocity",
    "attitude",
    "rate",
    "thrust",
    "yawrate",
    "roll",
    "pitch",

    # controller structures
    "PID",
    "pidInit",
    "pidUpdate",
    "controller",
    "stabilizer",

    # estimator outputs
    "stateEstimate",
    "kalman",
    "ekf",
    "complementary",

    # parameters / tuning
    "PARAM",
    "param",
    "posCtl",
    "velCtl",
    "attCtl",
    "rateCtl",
]

REGEX_PATTERNS = [
    r"controller_.*\.c",
    r"position_.*pid",
    r"attitude_.*pid",
    r"stabilizer",
    r"stateEstimate",
]

# -----------------------------
# PATH FILTERING
# -----------------------------

def is_included(path: Path, root: Path) -> bool:
    rel = str(path.relative_to(root)).replace("\\", "/")

    for ex in EXCLUDE_PREFIXES:
        if rel.startswith(ex):
            return False

    for inc in INCLUDE_PREFIXES:
        if rel.startswith(inc):
            return True

    return False


# -----------------------------
# SCAN LOGIC
# -----------------------------

def scan_file(path: Path):
    reasons = []

    for kw in PATH_KEYWORDS:
        if kw.lower() in str(path).lower():
            reasons.append(f"path:{kw}")

    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None

    for rx in REGEX_PATTERNS:
        if re.search(rx, text, re.IGNORECASE):
            reasons.append(f"regex:{rx}")

    for kw in CONTENT_KEYWORDS:
        if kw in text:
            reasons.append(f"content:{kw}")

    if reasons:
        return sorted(set(reasons))
    return None


def scan_repo(root: Path):
    hits = []

    for path in root.rglob("*"):
        if path.suffix.lower() not in EXTENSIONS:
            continue

        if not is_included(path, root):
            continue

        reasons = scan_file(path)
        if reasons:
            hits.append({
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "reasons": "; ".join(reasons),
            })

    return hits


# -----------------------------
# OUTPUT
# -----------------------------

def write_csv(results, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["relative_path", "path", "reasons"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def write_markdown(results, out_path, repo_root):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Crazyflie Controller Scan (Path-Gated)\n\n")
        f.write(f"- Repo: `{repo_root}`\n")
        f.write(f"- Generated: {datetime.utcnow().isoformat()} UTC\n")
        f.write(f"- Total matches: {len(results)}\n\n")

        current_dir = None
        for r in sorted(results, key=lambda x: x["relative_path"]):
            dir_name = os.path.dirname(r["relative_path"])
            if dir_name != current_dir:
                f.write(f"\n## `{dir_name or '.'}`\n\n")
                current_dir = dir_name

            f.write(f"- **{os.path.basename(r['relative_path'])}**  \n")
            f.write(f"  `{r['relative_path']}`  \n")
            f.write(f"  _{r['reasons']}_\n\n")


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python scan_repos_v2.py <repo_root>")
        sys.exit(1)

    repo_root = Path(sys.argv[1]).resolve()
    print(f"\nScanning (path-gated): {repo_root}\n")

    results = scan_repo(repo_root)

    out_dir = repo_root / "controller_scan"
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "controller_scan_results_v2.csv"
    md_path = out_dir / "controller_scan_results_v2.md"

    write_csv(results, csv_path)
    write_markdown(results, md_path, repo_root)

    print(f"Matches: {len(results)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Markdown saved to: {md_path}")
