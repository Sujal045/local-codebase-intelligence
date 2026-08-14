def compute_genuineness(job: dict) -> dict:
    """Return a spam / stale score for an inbox job listing."""
    reasons = []
    if not job.get("company"):
        reasons.append("missing company")
    score = 1.0 - (0.2 * len(reasons))
    return {"score": score, "is_spam": bool(reasons), "reasons": reasons}


def rank_score(match_score: float, genuineness_score: float) -> float:
    return (0.7 * match_score) + (0.3 * genuineness_score)
