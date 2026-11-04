"""Spec 0005 teacher exam: parsing, consistency by code, admission thresholds, exam prompts never sent (N1)."""
from cerbeto.curriculum import generate, skills
from cerbeto.exams.grader import answer_number
from cerbeto.teachers.exam import consistent, examine, parse, sample


def perfect_teacher(tasks):
    """Answers with the generator's own reply; explains numbers and texts with the truth."""
    by_prompt = {t.prompt: t for t in tasks}

    def ask(messages, task=None):
        t = by_prompt[messages[-1]["content"]]
        steps = "The task does not say." if t.answer_type == "unknown" else f"So it is {t.answer}."
        return f"Steps: {steps}\nAnswer: {t.reply}"
    return ask


def test_parse_two_lines_and_fallback():
    assert parse("Steps: 3 + 4 = 7.\nAnswer: 7") == ("3 + 4 = 7.", "7")
    assert parse("steps: a\n\nanswer: Yes, it is.") == ("a", "Yes, it is.")
    assert parse("Seven.") == ("", "Seven.")


def test_consistency_is_checked_by_code():
    num = next(t for i in range(200) if (t := generate("arithmetic10", "train", i)).answer_type == "number")
    assert consistent(num, f"We add and get {num.answer}.", num.answer)
    assert not consistent(num, f"We add and get {int(num.answer) + 1}.", num.answer)
    unknown = generate("arithmetic10", "train", 1, unanswerable=True)
    assert consistent(unknown, "The task does not say.", "I don't know.")
    assert not consistent(unknown, "It is 4.", "4")
    yesno = next(t for i in range(500) if (t := generate("social.agree_disagree", "train", i)).answer_type == "yesno")
    assert consistent(yesno, "whatever", "yes") is None


def test_perfect_teacher_is_admitted_and_a_sloppy_one_is_not():
    tasks = sample(10, seed=1)
    assert len(tasks) == 10 * len(skills())
    good = examine(tasks, perfect_teacher(tasks))["summary"]
    assert good["all"]["correct"] == 1.0 and good["all"]["consistent"] == 1.0 and good["passed"]

    def sloppy(messages, task=None):   # right answers, but every number explanation reaches a wrong result
        reply = perfect_teacher(tasks)(messages)
        steps, answer = parse(reply)
        n = answer_number(steps)
        return reply if n is None else f"Steps: So it is {n + 1}.\nAnswer: {answer}"
    bad = examine(tasks, sloppy)["summary"]
    assert bad["all"]["correct"] == 1.0 and not bad["passed"]


def test_n1_teacher_exam_never_samples_an_exam_prompt():
    tasks = sample(5, seed=2)
    exclude = {tasks[0].hash, tasks[7].hash}
    again = sample(5, seed=2, exclude=exclude)
    assert not exclude & {t.hash for t in again}
