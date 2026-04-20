#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


PPT_TOTAL_ALIASES = ["ppt_total", "presentation_total", "pitch_total", "ppt score"]
GITHUB_SUBMITTED_ALIASES = ["github_submitted", "github submission", "github_submitted?"]
PROJECT_TITLE_ALIASES = ["project_title", "project name", "title"]
TEAM_ID_ALIASES = ["team_id", "team id"]
TEAM_NUMBER_ALIASES = ["team_number", "team no", "team_number "]
TEAM_NAME_ALIASES = ["team_name", "team name"]

SUMMARY_METRIC_ALIASES = {
    "total teams": "total_teams",
    "successfully evaluated": "successfully_evaluated",
    "failed evaluations": "failed_evaluations",
    "github submissions": "github_submissions",
    "ppt total — mean": "ppt_mean",
    "ppt total - mean": "ppt_mean",
    "ppt total — min": "ppt_min",
    "ppt total - min": "ppt_min",
    "ppt total — max": "ppt_max",
    "ppt total - max": "ppt_max",
    "github total — mean": "github_mean",
    "github total - mean": "github_mean",
    "github total — min": "github_min",
    "github total - min": "github_min",
    "github total — max": "github_max",
    "github total - max": "github_max",
    "overall total — mean": "overall_mean",
    "overall total - mean": "overall_mean",
    "overall total — min": "overall_min",
    "overall total - min": "overall_min",
    "overall total — max": "overall_max",
    "overall total - max": "overall_max",
    "battle rows stored": "battle_rows_stored",
    "unique matchups": "unique_matchups",
    "relative scores computed": "relative_scores_computed",
    "relative score — mean": "relative_score_mean",
    "relative score - mean": "relative_score_mean",
    "total reconciliation flags": "total_reconciliation_flags",
    "unresolved flags": "unresolved_flags",
}

PPT_SUBSCORE_ALIASES = {
    "problem_specificity": ["problem_specificity", "problem specificity"],
    "solution_problem_fit": ["solution_problem_fit", "solution problem fit"],
    "technical_clarity": ["technical_clarity", "technical clarity"],
    "domain_depth": ["domain_depth", "domain depth"],
}

GITHUB_SUBSCORE_ALIASES = {
    "implementation_authenticity": ["implementation_authenticity", "implementation authenticity"],
    "stack_alignment": ["stack_alignment", "stack alignment"],
    "core_feature_presence": ["core_feature_presence", "core feature presence"],
    "code_intentionality": ["code_intentionality", "code intentionality"],
}


@dataclass
class ResolvedColumns:
    ppt_total: str
    github_total: Optional[str]
    overall_total: Optional[str]
    github_submitted: str
    project_title: Optional[str]
    reconciliation_flagged: Optional[str]
    team_id: Optional[str]
    team_number: Optional[str]
    team_name: Optional[str]
    subscores: Dict[str, str]


@dataclass
class DomainBundle:
    domain: str
    score_file: str
    reasoning_file: Optional[str]
    summary_file: Optional[str]


def normalize_colname(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    txt = str(value).lower().strip()
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt


def find_first_match(columns: Iterable[str], aliases: List[str]) -> Optional[str]:
    norm_map = {normalize_colname(c): c for c in columns}
    for alias in aliases:
        hit = norm_map.get(normalize_colname(alias))
        if hit:
            return hit
    return None


def to_bool_series(series: pd.Series) -> pd.Series:
    true_values = {"true", "t", "1", "yes", "y"}
    false_values = {"false", "f", "0", "no", "n"}

    def _to_bool(v: object) -> float:
        if pd.isna(v):
            return np.nan
        if isinstance(v, bool):
            return float(v)
        sv = str(v).strip().lower()
        if sv in true_values:
            return 1.0
        if sv in false_values:
            return 0.0
        return np.nan

    return series.map(_to_bool)


def infer_domain_from_filename(path: str) -> str:
    base = os.path.basename(path).lower()
    if "health" in base:
        return "healthcare"
    if "log" in base:
        return "logistics"
    if "fin" in base:
        return "fintech"
    return os.path.splitext(os.path.basename(path))[0]


def strip_empty_leading_column(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c) for c in df.columns]
    if cols and normalize_colname(cols[0]) in {"", "unnamed: 0"}:
        col0 = df.iloc[:, 0]
        if col0.isna().all() or (col0.astype(str).str.strip() == "").all():
            return df.iloc[:, 1:].copy()
    return df


def get_bundles() -> List[DomainBundle]:
    score_files = sorted(glob.glob("* - Scores.csv"))
    bundles: List[DomainBundle] = []
    for score in score_files:
        base = score.replace(" - Scores.csv", "")
        reasoning = f"{base} - Reasoning.csv"
        summary = f"{base} - Summary.csv"
        bundles.append(
            DomainBundle(
                domain=infer_domain_from_filename(score),
                score_file=score,
                reasoning_file=reasoning if os.path.exists(reasoning) else None,
                summary_file=summary if os.path.exists(summary) else None,
            )
        )
    return bundles


def resolve_columns(df: pd.DataFrame) -> ResolvedColumns:
    cols = list(df.columns)
    ppt_total = find_first_match(cols, PPT_TOTAL_ALIASES)
    github_submitted = find_first_match(cols, GITHUB_SUBMITTED_ALIASES)
    github_total = find_first_match(cols, ["github_total", "github total"])
    overall_total = find_first_match(cols, ["overall_total", "overall total"])
    project_title = find_first_match(cols, PROJECT_TITLE_ALIASES)
    reconciliation_flagged = find_first_match(cols, ["reconciliation_flagged", "reconciliation flagged"])
    team_id = find_first_match(cols, TEAM_ID_ALIASES)
    team_number = find_first_match(cols, TEAM_NUMBER_ALIASES)
    team_name = find_first_match(cols, TEAM_NAME_ALIASES)

    subscores: Dict[str, str] = {}
    for key, aliases in {**PPT_SUBSCORE_ALIASES, **GITHUB_SUBSCORE_ALIASES}.items():
        match = find_first_match(cols, aliases)
        if match:
            subscores[key] = match

    missing_core = []
    if not ppt_total:
        missing_core.append("ppt_total")
    if not github_submitted:
        missing_core.append("github_submitted")
    for key in PPT_SUBSCORE_ALIASES:
        if key not in subscores:
            missing_core.append(key)
    if missing_core:
        raise ValueError(f"Missing required columns: {', '.join(missing_core)}")

    return ResolvedColumns(
        ppt_total=ppt_total,
        github_total=github_total,
        overall_total=overall_total,
        github_submitted=github_submitted,
        project_title=project_title,
        reconciliation_flagged=reconciliation_flagged,
        team_id=team_id,
        team_number=team_number,
        team_name=team_name,
        subscores=subscores,
    )


def read_and_prepare_scores(path: str) -> Tuple[pd.DataFrame, ResolvedColumns]:
    df = strip_empty_leading_column(pd.read_csv(path))
    resolved = resolve_columns(df)
    domain = infer_domain_from_filename(path)

    df["source_file"] = os.path.basename(path)
    df["domain_label"] = domain
    df["ppt_total_num"] = pd.to_numeric(df[resolved.ppt_total], errors="coerce")
    if resolved.github_total:
        df["github_total_num"] = pd.to_numeric(df[resolved.github_total], errors="coerce")
    else:
        df["github_total_num"] = np.nan
    if resolved.overall_total:
        df["overall_total_num"] = pd.to_numeric(df[resolved.overall_total], errors="coerce")
    else:
        df["overall_total_num"] = np.nan
    df["github_submitted_bin"] = to_bool_series(df[resolved.github_submitted])

    for key, col in resolved.subscores.items():
        df[f"{key}_num"] = pd.to_numeric(df[col], errors="coerce")

    if resolved.project_title:
        df["project_title_norm"] = df[resolved.project_title].astype(str).str.strip()
        df.loc[df[resolved.project_title].isna(), "project_title_norm"] = ""
    else:
        df["project_title_norm"] = ""

    if resolved.team_id:
        df["team_id_norm"] = df[resolved.team_id].astype(str).str.strip().str.lower()
    else:
        df["team_id_norm"] = ""

    if resolved.team_name:
        df["team_name_norm"] = df[resolved.team_name].astype(str).str.strip().str.lower()
    else:
        df["team_name_norm"] = ""

    return df, resolved


def read_reasoning(path: str, domain: str) -> pd.DataFrame:
    df = strip_empty_leading_column(pd.read_csv(path))
    required = {"team_id", "team_name", "dimension", "score", "reasoning", "confidence"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Reasoning file {path} missing columns: {', '.join(missing)}")

    out = df.copy()
    out["source_file"] = os.path.basename(path)
    out["domain_label"] = domain
    out["team_id_norm"] = out["team_id"].astype(str).str.strip().str.lower()
    out["dimension_norm"] = out["dimension"].astype(str).str.strip().str.lower()
    out["reasoning_score_num"] = pd.to_numeric(out["score"], errors="coerce")
    out["confidence_norm"] = out["confidence"].astype(str).str.strip().str.lower()
    out["reasoning_norm"] = out["reasoning"].map(normalize_text)
    out["reasoning_len"] = out["reasoning"].astype(str).str.len()
    return out


def read_summary(path: str, domain: str) -> pd.DataFrame:
    df = strip_empty_leading_column(pd.read_csv(path))
    required = {"metric", "value"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Summary file {path} missing columns: {', '.join(missing)}")
    out = df.copy()
    out["source_file"] = os.path.basename(path)
    out["domain_label"] = domain
    out["metric_norm"] = out["metric"].astype(str).str.strip().str.lower()
    out["value_norm"] = out["value"].astype(str).str.strip()
    return out


def compute_ppt_stats(df: pd.DataFrame) -> Dict[str, float]:
    s = df["ppt_total_num"]
    return {
        "rows": int(len(df)),
        "ppt_non_null": int(s.notna().sum()),
        "ppt_missing": int(s.isna().sum()),
        "mean": float(s.mean()) if s.notna().any() else np.nan,
        "std_dev": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
        "variance": float(s.var(ddof=1)) if s.notna().sum() > 1 else np.nan,
        "min": float(s.min()) if s.notna().any() else np.nan,
        "max": float(s.max()) if s.notna().any() else np.nan,
        "unique_scores": int(s.nunique(dropna=True)),
    }


def compute_pearson(df: pd.DataFrame) -> Dict[str, float]:
    sub = df[["github_submitted_bin", "ppt_total_num"]].dropna()
    if len(sub) < 3:
        return {"n": int(len(sub)), "pearson_r": np.nan, "p_value": np.nan}
    if sub["github_submitted_bin"].nunique() < 2:
        return {"n": int(len(sub)), "pearson_r": np.nan, "p_value": np.nan}
    r, p = pearsonr(sub["github_submitted_bin"], sub["ppt_total_num"])
    return {"n": int(len(sub)), "pearson_r": float(r), "p_value": float(p)}


def compute_mismatch_count(df: pd.DataFrame) -> Dict[str, float]:
    score_cols = [
        "problem_specificity_num",
        "solution_problem_fit_num",
        "technical_clarity_num",
        "domain_depth_num",
    ]
    sub = df[["ppt_total_num", *score_cols]].copy()
    mask = sub[score_cols].notna().all(axis=1) & sub["ppt_total_num"].notna()
    if mask.sum() == 0:
        return {"rows_checked": 0, "mismatch_count": 0}
    calc = sub.loc[mask, score_cols].sum(axis=1)
    mismatch = (calc != sub.loc[mask, "ppt_total_num"]).sum()
    return {"rows_checked": int(mask.sum()), "mismatch_count": int(mismatch)}


def compute_out_of_range(df: pd.DataFrame) -> Dict[str, int]:
    subscore_cols = [
        "problem_specificity_num",
        "solution_problem_fit_num",
        "technical_clarity_num",
        "domain_depth_num",
    ]
    sub_oor = 0
    for col in subscore_cols:
        sub_oor += int(((df[col] < 0) | (df[col] > 5)).fillna(False).sum())
    ppt_oor = int(((df["ppt_total_num"] < 0) | (df["ppt_total_num"] > 20)).fillna(False).sum())
    return {"subscore_out_of_range": sub_oor, "ppt_total_out_of_range": ppt_oor}


def flag_ghost_rows(df: pd.DataFrame, resolved: ResolvedColumns) -> pd.DataFrame:
    is_perfect = df["ppt_total_num"] == 20
    github_missing_or_false = df["github_submitted_bin"].isna() | (df["github_submitted_bin"] == 0)
    title_missing = df["project_title_norm"] == ""

    g1 = is_perfect & github_missing_or_false
    g2 = is_perfect & (github_missing_or_false | title_missing)
    g3 = is_perfect & title_missing

    flagged = df[g1 | g2 | g3].copy()
    if flagged.empty:
        return flagged

    reasons = []
    for i in flagged.index:
        row_reasons = []
        if bool(g1.loc[i]):
            row_reasons.append("G1_perfect_without_github")
        if bool(g2.loc[i]):
            row_reasons.append("G2_perfect_without_github_or_title")
        if bool(g3.loc[i]):
            row_reasons.append("G3_perfect_without_title")
        reasons.append(";".join(row_reasons))
    flagged["ghost_reasons"] = reasons

    keep_cols = ["source_file", "domain_label", "ppt_total_num", "github_submitted_bin", "project_title_norm", "ghost_reasons"]
    for c in [resolved.team_id, resolved.team_number, resolved.team_name]:
        if c and c in flagged.columns:
            keep_cols.insert(2, c)
    keep_cols = list(dict.fromkeys(keep_cols))
    return flagged[keep_cols].copy()


def calculate_invalidity_flags(
    combined_df: pd.DataFrame,
    combined_pearson: Dict[str, float],
    ghost_count: int,
    reasoning_mismatch_count: int,
    duplicate_rate: float,
    contradiction_count: int,
) -> Dict[str, object]:
    perfect_share = float((combined_df["ppt_total_num"] == 20).mean(skipna=True))

    score_c1 = ghost_count > 0
    score_c2 = bool(np.isnan(combined_pearson["pearson_r"]) or abs(combined_pearson["pearson_r"]) < 0.1 or combined_pearson["p_value"] > 0.05)
    score_c3 = perfect_share >= 0.35
    score_invalid = score_c1 or ((1 if score_c2 else 0) + (1 if score_c3 else 0) >= 2)

    reasoning_r1 = reasoning_mismatch_count > 0
    reasoning_r2 = duplicate_rate >= 0.3
    reasoning_r3 = contradiction_count > 0

    final_invalid = score_invalid or reasoning_r1 or reasoning_r2 or reasoning_r3

    return {
        "criterion_1_ghosts_present": score_c1,
        "criterion_2_weak_or_nonsig_correlation": score_c2,
        "criterion_3_high_perfect_score_concentration": score_c3,
        "reasoning_r1_score_reasoning_mismatch": reasoning_r1,
        "reasoning_r2_high_duplicate_reasoning": reasoning_r2,
        "reasoning_r3_submission_reasoning_contradiction": reasoning_r3,
        "perfect_score_share": perfect_share,
        "final_verdict": "INVALID" if final_invalid else "NOT_PROVEN_INVALID",
    }


def save_score_visual(df: pd.DataFrame, output_path: str) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.histplot(
        data=df,
        x="ppt_total_num",
        hue="domain_label",
        multiple="layer",
        stat="count",
        bins=9,
        alpha=0.35,
        edgecolor="white",
        ax=axes[0],
    )
    axes[0].set_title("PPT Total Distribution by Domain")
    axes[0].set_xlabel("ppt_total")
    axes[0].set_ylabel("Count")

    box_df = df.copy()
    box_df["github_submitted_label"] = box_df["github_submitted_bin"].map({1.0: "Submitted", 0.0: "Not Submitted"})
    box_df["github_submitted_label"] = box_df["github_submitted_label"].fillna("Unknown")
    sns.boxplot(data=box_df, x="github_submitted_label", y="ppt_total_num", ax=axes[1], order=["Submitted", "Not Submitted", "Unknown"])
    sns.stripplot(
        data=box_df,
        x="github_submitted_label",
        y="ppt_total_num",
        ax=axes[1],
        order=["Submitted", "Not Submitted", "Unknown"],
        color="black",
        alpha=0.35,
        size=2.5,
    )
    axes[1].set_title("PPT Scores vs GitHub Submission Status")
    axes[1].set_xlabel("GitHub Submitted")
    axes[1].set_ylabel("ppt_total")

    fig.suptitle("Hack Hustle 2.0 Evaluation Audit: Score Evidence", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_reasoning_visual(reasoning_df: pd.DataFrame, duplicate_df: pd.DataFrame, output_path: str) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    conf = (
        reasoning_df.groupby(["domain_label", "confidence_norm"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    sns.barplot(data=conf, x="domain_label", y="count", hue="confidence_norm", ax=axes[0])
    axes[0].set_title("Reasoning Confidence Distribution")
    axes[0].set_xlabel("Domain")
    axes[0].set_ylabel("Count")

    if duplicate_df.empty:
        dup_rate = pd.DataFrame({"domain_label": sorted(reasoning_df["domain_label"].dropna().unique()), "duplicate_rate_pct": [0.0] * reasoning_df["domain_label"].nunique()})
    else:
        dup_rate = (
            duplicate_df.groupby("domain_label")
            .agg(duplicate_rows=("rows_in_cluster", "sum"))
            .reset_index()
        )
        total = reasoning_df.groupby("domain_label").size().reset_index(name="total_rows")
        dup_rate = dup_rate.merge(total, on="domain_label", how="right").fillna({"duplicate_rows": 0})
        dup_rate["duplicate_rate_pct"] = 100.0 * dup_rate["duplicate_rows"] / dup_rate["total_rows"]

    sns.barplot(data=dup_rate, x="domain_label", y="duplicate_rate_pct", ax=axes[1], color="#dd8452")
    axes[1].set_title("Templated Reasoning Concentration")
    axes[1].set_xlabel("Domain")
    axes[1].set_ylabel("Duplicate Rows (%)")

    fig.suptitle("Hack Hustle 2.0 Evaluation Audit: Reasoning Evidence", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def expected_dimensions_for_team(score_row: pd.Series) -> List[str]:
    expected = []
    for dim in [*PPT_SUBSCORE_ALIASES.keys(), *GITHUB_SUBSCORE_ALIASES.keys()]:
        value = score_row.get(f"{dim}_num", np.nan)
        if pd.notna(value):
            expected.append(dim)
    return expected


def analyze_reasoning(
    bundle: DomainBundle,
    score_df: pd.DataFrame,
    reasoning_df: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if reasoning_df is None:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, pd.DataFrame([
            {
                "domain_label": bundle.domain,
                "score_file": os.path.basename(bundle.score_file),
                "reasoning_file": "MISSING",
                "score_rows": len(score_df),
                "reasoning_rows": 0,
                "score_unique_teams": score_df["team_id_norm"].nunique(),
                "reasoning_unique_teams": 0,
                "unmatched_reasoning_rows": 0,
            }
        ])

    join_cols = [
        "team_id_norm",
        "github_submitted_bin",
        "source_file",
        "domain_label",
        "project_title_norm",
        "ppt_total_num",
        "github_total_num",
        "overall_total_num",
    ] + [f"{k}_num" for k in [*PPT_SUBSCORE_ALIASES.keys(), *GITHUB_SUBSCORE_ALIASES.keys()]]

    merged = reasoning_df.merge(score_df[join_cols], on=["team_id_norm", "domain_label"], how="left", suffixes=("", "_score"))
    merged["expected_score"] = merged.apply(lambda r: r.get(f"{r['dimension_norm']}_num", np.nan), axis=1)
    merged["score_match"] = (merged["reasoning_score_num"] == merged["expected_score"]) | (
        merged["reasoning_score_num"].isna() & merged["expected_score"].isna()
    )
    merged["joined_to_score"] = merged["ppt_total_num"].notna()
    merged["reasoning_blank"] = merged["reasoning_norm"] == ""

    contradiction_mask = (
        merged["dimension_norm"].isin(list(GITHUB_SUBSCORE_ALIASES.keys()))
        & ((merged["github_submitted_bin"] == 0) | merged["github_submitted_bin"].isna())
    )
    merged["submission_contradiction"] = contradiction_mask

    score_alignment = merged[
        [
            "source_file",
            "domain_label",
            "team_id",
            "team_name",
            "dimension",
            "reasoning_score_num",
            "expected_score",
            "score_match",
            "joined_to_score",
            "reasoning_blank",
            "submission_contradiction",
            "confidence",
            "reasoning",
        ]
    ].copy()

    # Coverage per team: expected dimensions from score rows vs present reasoning dimensions.
    reasoning_dims = (
        merged.groupby("team_id_norm")["dimension_norm"]
        .apply(lambda s: set(s.dropna().tolist()))
        .to_dict()
    )
    coverage_rows: List[Dict[str, object]] = []
    for _, row in score_df.iterrows():
        team_id = row["team_id_norm"]
        expected = set(expected_dimensions_for_team(row))
        present = reasoning_dims.get(team_id, set())
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        coverage_rows.append(
            {
                "domain_label": bundle.domain,
                "team_id_norm": team_id,
                "expected_dimensions_count": len(expected),
                "present_dimensions_count": len(present),
                "missing_dimensions": ";".join(missing),
                "extra_dimensions": ";".join(extra),
                "coverage_complete": len(missing) == 0,
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)

    confidence_df = (
        merged.groupby(["domain_label", "dimension_norm", "confidence_norm"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    dup_base = merged[merged["reasoning_norm"] != ""].copy()
    duplicate_clusters = (
        dup_base.groupby(["domain_label", "dimension_norm", "reasoning_norm"])
        .size()
        .reset_index(name="rows_in_cluster")
    )
    duplicate_clusters = duplicate_clusters[duplicate_clusters["rows_in_cluster"] > 1].copy()
    if not duplicate_clusters.empty:
        duplicate_clusters["sample_reasoning"] = duplicate_clusters["reasoning_norm"].str.slice(0, 220)

    ingestion = pd.DataFrame(
        [
            {
                "domain_label": bundle.domain,
                "score_file": os.path.basename(bundle.score_file),
                "reasoning_file": os.path.basename(bundle.reasoning_file or ""),
                "score_rows": len(score_df),
                "reasoning_rows": len(reasoning_df),
                "score_unique_teams": score_df["team_id_norm"].nunique(),
                "reasoning_unique_teams": reasoning_df["team_id_norm"].nunique(),
                "unmatched_reasoning_rows": int((~merged["joined_to_score"]).sum()),
            }
        ]
    )

    return score_alignment, coverage_df, confidence_df, duplicate_clusters, ingestion


def get_computed_summary_metrics(score_df: pd.DataFrame, reasoning_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if reasoning_df is None or reasoning_df.empty:
        success = int(score_df["ppt_total_num"].notna().sum())
    else:
        success = int(reasoning_df["team_id_norm"].nunique())

    total_teams = int(score_df["team_id_norm"].nunique())
    failed = max(total_teams - success, 0)
    github_sub = int((score_df["github_submitted_bin"] == 1).sum())

    result: Dict[str, object] = {
        "total_teams": total_teams,
        "successfully_evaluated": success,
        "failed_evaluations": failed,
        "github_submissions": github_sub,
        "ppt_mean": float(score_df["ppt_total_num"].mean()) if score_df["ppt_total_num"].notna().any() else np.nan,
        "ppt_min": float(score_df["ppt_total_num"].min()) if score_df["ppt_total_num"].notna().any() else np.nan,
        "ppt_max": float(score_df["ppt_total_num"].max()) if score_df["ppt_total_num"].notna().any() else np.nan,
        "github_mean": float(score_df["github_total_num"].mean()) if score_df["github_total_num"].notna().any() else np.nan,
        "github_min": float(score_df["github_total_num"].min()) if score_df["github_total_num"].notna().any() else np.nan,
        "github_max": float(score_df["github_total_num"].max()) if score_df["github_total_num"].notna().any() else np.nan,
        "overall_mean": float(score_df["overall_total_num"].mean()) if score_df["overall_total_num"].notna().any() else np.nan,
        "overall_min": float(score_df["overall_total_num"].min()) if score_df["overall_total_num"].notna().any() else np.nan,
        "overall_max": float(score_df["overall_total_num"].max()) if score_df["overall_total_num"].notna().any() else np.nan,
        "battle_rows_stored": 0,
        "unique_matchups": 0,
        "relative_scores_computed": 0,
        "relative_score_mean": "N/A",
        "total_reconciliation_flags": 0,
        "unresolved_flags": 0,
    }
    if "reconciliation_flagged" in score_df.columns:
        flags = to_bool_series(score_df["reconciliation_flagged"])
        result["total_reconciliation_flags"] = int((flags == 1).sum())
        result["unresolved_flags"] = int((flags == 1).sum())
    return result


def parse_reported_value(raw: str) -> object:
    txt = str(raw).strip()
    if txt == "":
        return np.nan
    if txt.lower() == "n/a":
        return "N/A"
    try:
        return float(txt)
    except ValueError:
        return txt


def compare_values(expected: object, reported: object) -> Tuple[str, object, object]:
    if (isinstance(expected, float) and np.isnan(expected)) and (isinstance(reported, float) and np.isnan(reported)):
        return "match", expected, reported

    if expected == "N/A" and reported == "N/A":
        return "match", expected, reported

    if isinstance(expected, (int, float, np.floating)) and not (isinstance(expected, float) and np.isnan(expected)):
        if isinstance(reported, (int, float, np.floating)) and not (isinstance(reported, float) and np.isnan(reported)):
            if abs(float(expected) - float(reported)) < 1e-9:
                return "match", expected, reported
            if abs(float(expected) - float(reported)) <= 0.11:
                return "rounded_match", expected, reported
            return "mismatch", expected, reported
        return "mismatch", expected, reported

    if isinstance(expected, str) and isinstance(reported, str):
        return ("match" if expected == reported else "mismatch"), expected, reported

    return "mismatch", expected, reported


def reconcile_summary(
    domain: str,
    score_file: str,
    summary_df: Optional[pd.DataFrame],
    computed: Dict[str, object],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if summary_df is None:
        for key, expected in computed.items():
            rows.append(
                {
                    "domain_label": domain,
                    "score_file": os.path.basename(score_file),
                    "metric_key": key,
                    "reported_metric": "MISSING_SUMMARY_FILE",
                    "reported_value": np.nan,
                    "expected_value": expected,
                    "status": "missing_metric",
                }
            )
        return pd.DataFrame(rows)

    reported_map = {}
    for _, row in summary_df.iterrows():
        metric_norm = row["metric_norm"]
        if metric_norm in SUMMARY_METRIC_ALIASES:
            key = SUMMARY_METRIC_ALIASES[metric_norm]
            reported_map[key] = {
                "metric": row["metric"],
                "value": parse_reported_value(row["value"]),
            }

    for key, expected in computed.items():
        if key not in reported_map:
            rows.append(
                {
                    "domain_label": domain,
                    "score_file": os.path.basename(score_file),
                    "metric_key": key,
                    "reported_metric": "MISSING_METRIC",
                    "reported_value": np.nan,
                    "expected_value": expected,
                    "status": "missing_metric",
                }
            )
            continue

        reported = reported_map[key]["value"]
        status, expected_out, reported_out = compare_values(expected, reported)
        rows.append(
            {
                "domain_label": domain,
                "score_file": os.path.basename(score_file),
                "metric_key": key,
                "reported_metric": reported_map[key]["metric"],
                "reported_value": reported_out,
                "expected_value": expected_out,
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    bundles = get_bundles()
    if not bundles:
        raise FileNotFoundError("No files matching '* - Scores.csv' were found in the current directory.")

    all_score_frames: List[pd.DataFrame] = []
    resolved_by_file: Dict[str, ResolvedColumns] = {}

    reasoning_alignment_list: List[pd.DataFrame] = []
    reasoning_coverage_list: List[pd.DataFrame] = []
    reasoning_confidence_list: List[pd.DataFrame] = []
    reasoning_duplicates_list: List[pd.DataFrame] = []
    ingestion_list: List[pd.DataFrame] = []
    summary_reconciliation_list: List[pd.DataFrame] = []

    summary_rows: List[Dict[str, object]] = []
    correlation_rows: List[Dict[str, object]] = []
    mismatch_rows: List[Dict[str, object]] = []
    range_rows: List[Dict[str, object]] = []
    ghost_outputs: List[pd.DataFrame] = []

    all_reasoning_frames: List[pd.DataFrame] = []

    for bundle in bundles:
        score_df, resolved = read_and_prepare_scores(bundle.score_file)
        all_score_frames.append(score_df)
        resolved_by_file[os.path.basename(bundle.score_file)] = resolved

        reasoning_df = read_reasoning(bundle.reasoning_file, bundle.domain) if bundle.reasoning_file else None
        summary_df = read_summary(bundle.summary_file, bundle.domain) if bundle.summary_file else None

        if reasoning_df is not None:
            all_reasoning_frames.append(reasoning_df)

        stats = compute_ppt_stats(score_df)
        pearson = compute_pearson(score_df)
        mismatch = compute_mismatch_count(score_df)
        ranges = compute_out_of_range(score_df)
        ghosts = flag_ghost_rows(score_df, resolved)

        summary_rows.append({"scope": os.path.basename(bundle.score_file), **stats})
        correlation_rows.append({"scope": os.path.basename(bundle.score_file), **pearson})
        mismatch_rows.append({"scope": os.path.basename(bundle.score_file), **mismatch})
        range_rows.append({"scope": os.path.basename(bundle.score_file), **ranges})
        if not ghosts.empty:
            ghost_outputs.append(ghosts)

        reasoning_alignment, reasoning_coverage, reasoning_confidence, reasoning_duplicates, ingestion_df = analyze_reasoning(
            bundle,
            score_df,
            reasoning_df,
        )

        if not reasoning_alignment.empty:
            reasoning_alignment_list.append(reasoning_alignment)
        if not reasoning_coverage.empty:
            reasoning_coverage_list.append(reasoning_coverage)
        if not reasoning_confidence.empty:
            reasoning_confidence_list.append(reasoning_confidence)
        if not reasoning_duplicates.empty:
            reasoning_duplicates_list.append(reasoning_duplicates)
        ingestion_list.append(ingestion_df)

        computed_summary = get_computed_summary_metrics(score_df, reasoning_df)
        summary_reconciliation = reconcile_summary(bundle.domain, bundle.score_file, summary_df, computed_summary)
        summary_reconciliation_list.append(summary_reconciliation)

    combined = pd.concat(all_score_frames, ignore_index=True)
    combined_reasoning = pd.concat(all_reasoning_frames, ignore_index=True) if all_reasoning_frames else pd.DataFrame()

    summary_rows.append({"scope": "COMBINED", **compute_ppt_stats(combined)})
    combined_pearson = compute_pearson(combined)
    correlation_rows.append({"scope": "COMBINED", **combined_pearson})
    mismatch_rows.append({"scope": "COMBINED", **compute_mismatch_count(combined)})
    range_rows.append({"scope": "COMBINED", **compute_out_of_range(combined)})

    summary_df = pd.DataFrame(summary_rows)
    correlation_df = pd.DataFrame(correlation_rows)
    mismatch_df = pd.DataFrame(mismatch_rows)
    range_df = pd.DataFrame(range_rows)
    ghost_df = pd.concat(ghost_outputs, ignore_index=True) if ghost_outputs else pd.DataFrame()

    reasoning_alignment_df = pd.concat(reasoning_alignment_list, ignore_index=True) if reasoning_alignment_list else pd.DataFrame()
    reasoning_coverage_df = pd.concat(reasoning_coverage_list, ignore_index=True) if reasoning_coverage_list else pd.DataFrame()
    reasoning_confidence_df = pd.concat(reasoning_confidence_list, ignore_index=True) if reasoning_confidence_list else pd.DataFrame()
    reasoning_duplicates_df = pd.concat(reasoning_duplicates_list, ignore_index=True) if reasoning_duplicates_list else pd.DataFrame()
    ingestion_df = pd.concat(ingestion_list, ignore_index=True) if ingestion_list else pd.DataFrame()
    summary_reconciliation_df = pd.concat(summary_reconciliation_list, ignore_index=True) if summary_reconciliation_list else pd.DataFrame()

    g1 = int(((combined["ppt_total_num"] == 20) & ((combined["github_submitted_bin"] == 0) | combined["github_submitted_bin"].isna())).sum())
    g2 = int(((combined["ppt_total_num"] == 20) & (((combined["github_submitted_bin"] == 0) | combined["github_submitted_bin"].isna()) | (combined["project_title_norm"] == ""))).sum())
    g3 = int(((combined["ppt_total_num"] == 20) & (combined["project_title_norm"] == "")).sum())

    reasoning_mismatch_count = int((~reasoning_alignment_df["score_match"]).sum()) if not reasoning_alignment_df.empty else 0
    contradiction_count = int(reasoning_alignment_df["submission_contradiction"].sum()) if not reasoning_alignment_df.empty else 0
    duplicate_rate = 0.0
    if not reasoning_alignment_df.empty and not reasoning_duplicates_df.empty:
        dup_total = int(reasoning_duplicates_df["rows_in_cluster"].sum())
        duplicate_rate = dup_total / max(len(reasoning_alignment_df), 1)

    verdict = calculate_invalidity_flags(
        combined,
        combined_pearson,
        ghost_count=g2,
        reasoning_mismatch_count=reasoning_mismatch_count,
        duplicate_rate=duplicate_rate,
        contradiction_count=contradiction_count,
    )

    summary_df.to_csv("audit_summary.csv", index=False)
    correlation_df.to_csv("audit_correlation.csv", index=False)
    mismatch_df.to_csv("audit_consistency.csv", index=False)
    range_df.to_csv("audit_ranges.csv", index=False)

    if not ghost_df.empty:
        ghost_df.to_csv("ghost_flags.csv", index=False)
    else:
        pd.DataFrame(columns=["note"]).to_csv("ghost_flags.csv", index=False)

    reasoning_alignment_df.to_csv("reasoning_consistency.csv", index=False)
    reasoning_coverage_df.to_csv("reasoning_coverage.csv", index=False)
    reasoning_confidence_df.to_csv("reasoning_confidence.csv", index=False)
    reasoning_duplicates_df.to_csv("reasoning_duplicates.csv", index=False)
    ingestion_df.to_csv("audit_ingestion.csv", index=False)
    summary_reconciliation_df.to_csv("summary_reconciliation.csv", index=False)

    save_score_visual(combined, "audit_visual_proof.png")
    save_reasoning_visual(combined_reasoning, reasoning_duplicates_df, "audit_reasoning_proof.png")

    print_section("PPT Variance Test")
    print(summary_df.to_string(index=False))

    print_section("GitHub Submission vs PPT Correlation (Pearson)")
    print(correlation_df.to_string(index=False))

    print_section("Score Consistency Check")
    print(mismatch_df.to_string(index=False))

    print_section("Range Check")
    print(range_df.to_string(index=False))

    print_section("Ghost Evaluation Counts")
    print(f"G1: ppt_total == 20 and github_submitted is False/missing: {g1}")
    print(f"G2: ppt_total == 20 and (github_submitted is False/missing OR project_title missing): {g2}")
    print(f"G3: ppt_total == 20 and project_title missing: {g3}")

    print_section("Reasoning Audit Summary")
    print(f"Reasoning rows: {len(reasoning_alignment_df)}")
    print(f"Reasoning score mismatches: {reasoning_mismatch_count}")
    print(f"Submission contradictions: {contradiction_count}")
    print(f"Duplicate reasoning rate: {duplicate_rate:.2%}")

    print_section("Summary Reconciliation")
    if summary_reconciliation_df.empty:
        print("No summary reconciliation rows generated.")
    else:
        print(summary_reconciliation_df.groupby("status").size().to_string())

    print_section("Final Audit Verdict")
    print(f"Criterion 1 (Ghosts present): {verdict['criterion_1_ghosts_present']}")
    print(f"Criterion 2 (Weak/Nonsignificant correlation): {verdict['criterion_2_weak_or_nonsig_correlation']}")
    print(f"Criterion 3 (High perfect-score concentration): {verdict['criterion_3_high_perfect_score_concentration']}")
    print(f"Reasoning R1 (Score mismatch): {verdict['reasoning_r1_score_reasoning_mismatch']}")
    print(f"Reasoning R2 (High duplicate reasoning): {verdict['reasoning_r2_high_duplicate_reasoning']}")
    print(f"Reasoning R3 (Submission contradiction): {verdict['reasoning_r3_submission_reasoning_contradiction']}")
    print(f"Perfect-score share: {verdict['perfect_score_share']:.2%}")
    print(f"VERDICT: {verdict['final_verdict']}")

    print_section("Generated Artifacts")
    print("- audit_summary.csv")
    print("- audit_correlation.csv")
    print("- audit_consistency.csv")
    print("- audit_ranges.csv")
    print("- ghost_flags.csv")
    print("- audit_ingestion.csv")
    print("- reasoning_consistency.csv")
    print("- reasoning_coverage.csv")
    print("- reasoning_confidence.csv")
    print("- reasoning_duplicates.csv")
    print("- summary_reconciliation.csv")
    print("- audit_visual_proof.png")
    print("- audit_reasoning_proof.png")


if __name__ == "__main__":
    main()