"""Spec 0005 G2: the forgetting gate rejects a version whose earlier stage drops by more than 2 pp."""
from cerbeto.school.scheduler import decide, stage_scores


def test_g2_forgetting_gate():
    ref = {"preschool": 0.90}
    assert decide({"preschool": 0.885, "grade1": 0.5}, "grade1", 0.4, ref, 2)[0]
    ok, why = decide({"preschool": 0.875, "grade1": 0.6}, "grade1", 0.4, ref, 2)
    assert not ok and "preschool forgot" in why


def test_current_stage_must_not_drop():
    assert decide({"preschool": 0.5}, "preschool", 0.5, {}, 2)[0]
    assert not decide({"preschool": 0.49}, "preschool", 0.5, {}, 2)[0]
    assert decide({"preschool": 0.1}, "preschool", None, {}, 2)[0]


def test_stage_score_uses_code_graded_skills_only():
    v = lambda a, g="code": {x: {"accuracy": a, "graded_by": g} for x in ("id", "ood_hard", "ood_combo")}
    assert stage_scores({"preschool": {"a": v(0.5), "b": v(1.0), "soft": v(0.0, "form"), "_passed": False}}) == \
        {"preschool": 0.75}
