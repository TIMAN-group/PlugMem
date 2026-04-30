"""Analyze JSONL output of run_tok_usage_bench.py.

Reads --in (JSONL of per-call records) and produces:

  1. Per-system, per-scope cumulative token curves vs R (retrievals).
  2. Per-system, per-scope dollar curves (using --pricing pricing.yaml).
  3. Crossover R* for tokens and dollars: smallest R at which a PlugMem
     system's cumulative cost equals the OpenClaw-default baseline.
  4. A markdown report at --report with the headline numbers.

Usage:
    uv run python scripts/bench/analyze.py \
        --in      data/bench/results.jsonl \
        --pricing scripts/bench/pricing.yaml \
        --report  data/bench/report.md \
        --plots   data/bench/plots/
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_pricing(path: Path) -> Dict[str, Dict[str, float]]:
    """Returns {model: {"input": $/M, "output": $/M}, "_default": {...}}."""
    import yaml  # PyYAML is already a transitive dep via plugmem.

    with open(path) as f:
        d = yaml.safe_load(f)
    out: Dict[str, Dict[str, float]] = dict(d.get("models", {}))
    out["_default"] = d.get("default", {"input": 1.0, "output": 3.0})
    return out


def cost_usd(rec: Dict[str, Any], pricing: Dict[str, Dict[str, float]]) -> float:
    p = pricing.get(rec["model"], pricing["_default"])
    return (
        rec["input_tokens"] / 1_000_000.0 * p["input"]
        + rec["output_tokens"] / 1_000_000.0 * p["output"]
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class Aggregate:
    """Summed cost for one (system_letter, R, scope) cell, averaged per trajectory."""
    system: str            # "A" | "B" | "C"
    R: int
    scope: str             # "EXPOSED" | "INTERNAL" | "INTERNAL_EMBED"
    n_trajectories: int
    tokens_in: float
    tokens_out: float
    usd: float

    @property
    def tokens(self) -> float:
        return self.tokens_in + self.tokens_out


def _split_system(system_field: str) -> Tuple[str, int]:
    """`A@R=3` -> ("A", 3)."""
    letter, _, rhs = system_field.partition("@R=")
    try:
        return letter, int(rhs)
    except ValueError:
        return letter, -1


def aggregate(
    records: List[Dict[str, Any]],
    pricing: Dict[str, Dict[str, float]],
) -> Dict[Tuple[str, int, str], Aggregate]:
    """Sum per-call records into (system, R, scope) cells, normalized per trajectory.

    EVAL-scoped records (judge LLM calls + judge_score sentinels) are excluded —
    they belong to evaluation overhead, not the system being measured.
    """
    sums: Dict[Tuple[str, int, str], List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    trajs: Dict[Tuple[str, int, str], set] = defaultdict(set)

    for r in records:
        if r.get("scope") == "EVAL":
            continue
        letter, R = _split_system(r.get("system", ""))
        if R < 0:
            continue
        key = (letter, R, r["scope"])
        sums[key][0] += r["input_tokens"]
        sums[key][1] += r["output_tokens"]
        sums[key][2] += cost_usd(r, pricing)
        trajs[key].add(r["trajectory_id"])

    out: Dict[Tuple[str, int, str], Aggregate] = {}
    for key, (ti, to, usd) in sums.items():
        n = max(1, len(trajs[key]))
        out[key] = Aggregate(
            system=key[0], R=key[1], scope=key[2],
            n_trajectories=n,
            tokens_in=ti / n, tokens_out=to / n, usd=usd / n,
        )
    return out


def aggregate_quality(records: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Tuple[float, int]]:
    """Group judge_score records by (system_letter, R) -> (mean_score, n).

    Skips records with score < 0 (parse failures).
    """
    scores: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for r in records:
        if r.get("scope") != "EVAL" or r.get("phase") != "judge_score":
            continue
        s = r.get("extra", {}).get("judge_score", -1)
        if s < 0:
            continue
        letter, R = _split_system(r.get("system", ""))
        if R < 0:
            continue
        scores[(letter, R)].append(s)

    out: Dict[Tuple[str, int], Tuple[float, int]] = {}
    for k, vs in scores.items():
        out[k] = (sum(vs) / len(vs), len(vs))
    return out


def curve(
    agg: Dict[Tuple[str, int, str], Aggregate],
    system: str,
    scope: Optional[str],
    metric: str = "tokens",  # "tokens" | "usd"
) -> List[Tuple[int, float]]:
    """Return [(R, value), ...] sorted by R. scope=None sums across all scopes."""
    by_r: Dict[int, float] = defaultdict(float)
    for (sys_, R, sc), a in agg.items():
        if sys_ != system:
            continue
        if scope is not None and sc != scope:
            continue
        by_r[R] += getattr(a, metric)
    return sorted(by_r.items())


def crossover(
    baseline: List[Tuple[int, float]],
    candidate: List[Tuple[int, float]],
) -> Optional[int]:
    """Smallest R at which candidate(R) <= baseline(R). None if it never happens."""
    base_map = dict(baseline)
    for R, v in sorted(candidate):
        if R in base_map and v <= base_map[R]:
            return R
    return None


# ---------------------------------------------------------------------------
# Plotting (optional — falls back to silent skip if matplotlib missing)
# ---------------------------------------------------------------------------

def plot_curves(
    agg: Dict[Tuple[str, int, str], Aggregate],
    plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        logger.warning("matplotlib not available; skipping plots")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    systems = sorted({s for (s, _, _) in agg})
    scopes = ["EXPOSED", "INTERNAL", "INTERNAL_EMBED"]

    # One figure per metric (tokens, usd) showing all (system, scope) lines.
    for metric in ("tokens", "usd"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for sys_ in systems:
            for sc in scopes:
                pts = curve(agg, sys_, sc, metric=metric)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=f"{sys_} / {sc}")
        ax.set_xlabel("Retrievals R")
        ax.set_ylabel("cumulative tokens" if metric == "tokens" else "cumulative USD")
        ax.set_title(f"Token-cost benchmark — {metric}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"curves_{metric}.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_report(
    agg: Dict[Tuple[str, int, str], Aggregate],
    out_path: Path,
    quality: Optional[Dict[Tuple[str, int], Tuple[float, int]]] = None,
) -> None:
    lines: List[str] = []
    lines.append("# Token-cost benchmark report\n")

    # ---- Per-system, per-scope summed table ----
    lines.append("## Per-system / per-scope cost (mean per trajectory)\n")
    lines.append("| system | R | scope | tokens_in | tokens_out | usd |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for key in sorted(agg.keys()):
        a = agg[key]
        lines.append(
            f"| {a.system} | {a.R} | {a.scope} | "
            f"{a.tokens_in:,.0f} | {a.tokens_out:,.0f} | ${a.usd:.4f} |"
        )

    # ---- Crossover summary ----
    lines.append("\n## Crossover R\\* (PlugMem cum-cost <= OpenClaw-default cum-cost)\n")
    lines.append("| candidate | scope | metric | R\\* |")
    lines.append("|---|---|---|---:|")

    # Total curves per system
    for system in ("B", "C"):
        for metric in ("tokens", "usd"):
            base = curve(agg, "A", scope=None, metric=metric)
            cand = curve(agg, system, scope=None, metric=metric)
            R_star = crossover(base, cand)
            lines.append(
                f"| {system} (total) | all | {metric} | "
                f"{R_star if R_star is not None else 'never'} |"
            )

    # Exposed-only curves — the headline cost the agent operator pays
    for system in ("B", "C"):
        for metric in ("tokens", "usd"):
            base = curve(agg, "A", scope="EXPOSED", metric=metric)
            cand = curve(agg, system, scope="EXPOSED", metric=metric)
            R_star = crossover(base, cand)
            lines.append(
                f"| {system} (exposed-only) | EXPOSED | {metric} | "
                f"{R_star if R_star is not None else 'never'} |"
            )

    # ---- Slope ratios at largest R ----
    lines.append("\n## Headline ratios (at max R)\n")
    rs = sorted({R for (_, R, _) in agg})
    if rs:
        Rmax = rs[-1]
        a_exposed = sum(
            v.tokens for (s, R, sc), v in agg.items()
            if s == "A" and R == Rmax and sc == "EXPOSED"
        )
        for system in ("B", "C"):
            b_exposed = sum(
                v.tokens for (s, R, sc), v in agg.items()
                if s == system and R == Rmax and sc == "EXPOSED"
            )
            ratio = a_exposed / b_exposed if b_exposed else float("inf")
            lines.append(
                f"- **Exposed-token reduction at R={Rmax}, {system} vs A:** "
                f"{ratio:.2f}× fewer tokens on the agent's hot path"
            )

    # ---- Answer quality (optional) -----------------------------------------
    if quality:
        lines.append("\n## Answer quality (LLM judge, 0-5 rubric)\n")
        lines.append("| system | R | mean_score | n | retention vs A |")
        lines.append("|---|---:|---:|---:|---:|")
        # Baseline: A's mean per R.
        a_mean = {R: m for (s, R), (m, _n) in quality.items() if s == "A"}
        for (s, R) in sorted(quality.keys()):
            mean, n = quality[(s, R)]
            base = a_mean.get(R)
            ret = f"{(mean - base):+.2f}" if base is not None and s != "A" else "—"
            lines.append(f"| {s} | {R} | {mean:.2f} | {n} | {ret} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("wrote report -> %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--pricing", type=Path,
                    default=Path(__file__).parent / "pricing.yaml")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--plots", type=Path, default=None,
                    help="Optional directory for PNG plots")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = load_jsonl(args.in_path)
    pricing = load_pricing(args.pricing)
    agg = aggregate(records, pricing)
    quality = aggregate_quality(records)

    render_report(agg, args.report, quality=quality)
    if args.plots is not None:
        plot_curves(agg, args.plots)


if __name__ == "__main__":
    main()
