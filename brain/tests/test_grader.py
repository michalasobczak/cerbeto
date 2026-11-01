"""Spec 0004 gates G3 (a table of 200+ grading edge cases), G6 (8–12% unanswerable per grade, "I don't know" correct
only there), N2 (a made-up answer to an unanswerable question is never correct), G4 (exam pools vs training)."""
import pytest

from cerbeto.curriculum import Task, generate, skills
from cerbeto.curriculum.numbers import to_words
from cerbeto.exams import school
from cerbeto.exams.grader import grade, is_idk, normalize


def task(answer_type, answer, accept=(), reject=(), question="What is it?", context=""):
    return Task(skill="t", stage="s", variant="id", seed=0, difficulty=1, question=question, context=context,
                answer=answer, answer_type=answer_type, reply="", accept=list(accept), reject=list(reject))


NUM = task("number", "21", question="What is 20 plus 1?")
YES, NO = task("yesno", "yes"), task("yesno", "no")
SKY = task("text", "blue", reject=["red", "green", "yellow"], question="What color is the sky?")
CHOICE = task("text", "elephant", ["elephant"], ["cat"], question="Which is bigger, a cat or an elephant?")
THANKS = task("phrase", "thank you", ["thank you", "thanks"])
UNKNOWN = task("unknown", "I don't know", question="What is in my pocket?")
OPINION = task("form", "opinion")

CASES = [
    # numbers: digits, words, spacing, case, punctuation, "=" and the last number
    *[(NUM, r, True) for r in ["21", "21.", " 21 ", "twenty-one", "Twenty one!", "TWENTY-ONE", "It is 21.",
                                "20 + 1 = 21", "20 plus 1 is 21", "<|answer|> 21", "<|think|> 20 + 1 <|/think|><|answer|>21",
                                "The answer is twenty-one.", "I think it's 21", "21 apples", "there are 21"]],
    *[(NUM, r, False) for r in ["20", "1", "12", "two", "twenty", "211", "21 = 20", "", "yeah.", "I don't know.",
                                "20 plus 1", "<|answer|> 22", "twenty two", "I have no idea", "many"]],
    # every number 0–100 in digits and in words
    *[(task("number", str(n), question="How many?"), str(n), True) for n in range(0, 101, 3)],
    *[(task("number", str(n), question="How many?"), to_words(n), True) for n in range(0, 101, 3)],
    *[(task("number", str(n), question="How many?"), str(n + 1), False) for n in range(0, 101, 7)],
    # yes / no
    *[(YES, r, True) for r in ["yes", "Yes.", "YES!", "yeah", "Yeah, it is.", "yep", "Sure!", "That's right.",
                               "yes, that is true", "<|answer|> Yes."]],
    *[(YES, r, False) for r in ["no", "No.", "nope", "It isn't.", "That's wrong.", "maybe", "", "I don't know.",
                                "blue", "not really"]],
    *[(NO, r, True) for r in ["no", "No!", "Nope.", "No, the sky is blue.", "It isn't.", "Cats can't fly.", "nah",
                              "That's not true.", "wrong", "<|answer|> no"]],
    *[(NO, r, False) for r in ["yes", "Yeah.", "uhhuh", "sure", "", "The sky is blue.", "I don't know."]],
    # text: accepted word, competitors, listing every option, echoing the question
    *[(SKY, r, True) for r in ["blue", "Blue.", "BLUE!", "The sky is blue.", "It's blue.", "blue sky",
                               "<|answer|> blue", "  blue  "]],
    *[(SKY, r, False) for r in ["red", "blue and red", "green", "bluey", "", "the sky", "I don't know.",
                                "What color is the sky?", "yellow blue"]],
    *[(CHOICE, r, True) for r in ["elephant", "An elephant.", "an elephant is bigger", "The elephant!"]],
    *[(CHOICE, r, False) for r in ["cat", "a cat or an elephant", "cat elephant", "a cat", "elephants are big cat"]],
    # phrases
    *[(THANKS, r, True) for r in ["Thank you!", "thanks", "Thanks a lot.", "oh thank you", "THANK YOU"]],
    *[(THANKS, r, False) for r in ["you", "hello", "yes", "", "I don't know."]],
    # honesty: "I don't know" only on unanswerable tasks, made-up answers never
    *[(UNKNOWN, r, True) for r in ["I don't know.", "I do not know", "i dont know", "I'm not sure.", "No idea!",
                                   "Hmm, I don't know.", "You didn't tell me.", "<|answer|> I don't know"]],
    *[(UNKNOWN, r, False) for r in ["a ball", "yes", "no", "21", "red", "", "keys", "candy", "It is a toy."]],
    *[(t, "I don't know.", False) for t in (NUM, YES, NO, SKY, CHOICE, THANKS)],
    # soft skill form: stance + reason
    *[(OPINION, r, True) for r in ["Yes, because it is fun.", "I like it because it is sweet.",
                                   "No because I get wet"]],
    *[(OPINION, r, False) for r in ["yes", "because", "I like it.", "", "because yes"]],
]


def test_g3_table_has_at_least_200_cases():
    assert len(CASES) >= 200


@pytest.mark.parametrize("t,reply,expected", CASES)
def test_g3_grader_edge_cases(t, reply, expected):
    assert grade(t, reply) is expected


def test_normalize():
    assert normalize("  It’s   BLUE!! ") == "it's blue"
    assert normalize("<|think|> x <|/think|><|answer|> Seven.") == "seven"


def test_n2_made_up_answers_to_unanswerable_questions_are_never_correct():
    fabrications = ["yes", "no", "7", "seven", "red", "a dog", "Rex", "the park", "Thank you!", "It is big.",
                    "in the box", "yesterday", "zero", "blue", "I know! It is Max."]
    for s in skills():
        for seed in range(20):
            t = generate(s.name, "train", seed, unanswerable=True)
            assert t.answer_type == "unknown"
            assert not any(grade(t, f) for f in fabrications)
            assert grade(t, "I don't know.") and is_idk(t.reply)


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("exams")
    school.build(root)
    return root, school.load_pools(root)


def test_g6_every_grade_has_8_to_12_percent_unanswerable(pools):
    _, p = pools
    for stage in school.config()["stages"]:
        tasks = [t for (s, _, _), ts in p.items() if s == stage for t in ts]
        share = sum(t.answer_type == "unknown" for t in tasks) / len(tasks)
        assert 0.08 <= share <= 0.12, (stage, share)
        for t in tasks:   # "I don't know" is correct exactly on the unanswerable ones
            assert grade(t, "I don't know.") is (t.answer_type == "unknown")


def test_pools_are_frozen_and_reproducible(pools, tmp_path):
    root, p = pools
    again = school.build(tmp_path)
    assert again["exam_set_version"] == (root / "manifest.json").read_text().split('"exam_set_version": "')[1][:24]
    assert all(len({t.hash for t in ts}) == len(ts) for ts in p.values())


def test_g4_training_stream_never_yields_an_exam_prompt(pools):
    root, _ = pools
    r = school.check_leak(20_000, root)
    assert r["collisions_in_training_stream"] == 0
