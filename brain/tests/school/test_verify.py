"""Spec 0005 G1: the verifier rejects every explanation whose answer contradicts the generator's truth."""
from cerbeto.curriculum import generate
from cerbeto.school.day import tutor


def contradicting(task):
    if task.answer_type == "number":
        wrong = str(int(task.answer) + 1)
        return f"Steps: So it is {wrong}.\nAnswer: {wrong}"
    if task.answer_type == "yesno":
        wrong = "no" if task.answer == "yes" else "yes"
        return f"Steps: The answer is {wrong}.\nAnswer: {wrong}"
    if task.answer_type == "unknown":
        return "Steps: It is in the box.\nAnswer: a ball"
    wrong = task.reject[0] if task.reject else "nothing"
    return f"Steps: It is {wrong}.\nAnswer: {wrong}"


def cases(n=100):
    skills = ["arithmetic10", "arithmetic100", "word_problems", "counting", "order", "cause_effect", "comparisons",
              "world.facts", "world.sizes", "world.categories", "social.agree_disagree", "reading"]
    out, i = [], 0
    while len(out) < n:
        t = generate(skills[i % len(skills)], "train", i)
        i += 1
        if t.answer_type != "text" or t.reject:
            out.append(t)
    return out


def test_g1_verifier_rejects_contradicting_explanations():
    tasks = cases()
    assert len(tasks) == 100
    assert all(tutor(t, lambda m, think: contradicting(t), False) is None for t in tasks)


def test_verifier_accepts_a_right_explanation_with_the_truth():
    t = next(t for i in range(100) if (t := generate("arithmetic10", "train", i)).answer_type == "number")
    assert tutor(t, lambda m, think: f"Steps: {' '.join(t.steps)} So it is {t.answer}.\nAnswer: {t.answer}", False)
