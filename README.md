# Hack Hustle 2.0 Evaluation Audit

## Executive Summary

The automated evaluation output for Hack Hustle 2.0 is not reliable enough to use as the final basis for shortlisting.

This conclusion is supported by three independent checks across the score, reasoning, and summary files for all domains.

## Key Findings

- Score layer: 197 teams received perfect PPT scores (20/20), including 105 teams with no GitHub submission and 197 teams with missing project titles.
- Correlation layer: GitHub submission and PPT score show near-zero relationship ($r = 0.0359$, $p = 0.4259$).
- Reasoning layer: 1108 reasoning rows repeat the same GitHub-not-submitted rationale pattern, with 2 score-to-reasoning mismatches and 1108 submission contradictions.
- Summary layer: at least one material reconciliation error exists, including healthcare `overall_min` reported as 0.0 instead of the recomputed 12.0.

## Recommendation

Do not publish the current AI-only shortlist as final.

Freeze the shortlist, conduct a manual faculty/jury review of the flagged entries, and reconcile the summary metrics before any final announcement.

## Supporting Evidence

- Score audit: [audit_visual_proof.png](audit_visual_proof.png)
- Reasoning audit: [audit_reasoning_proof.png](audit_reasoning_proof.png)
