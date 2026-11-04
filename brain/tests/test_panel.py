"""`cerbeto panel`: read-only views of the hippocampus — filters and the teacher's raw reply."""
from cerbeto import panel
from cerbeto.memory.hippocampus import Experience, Hippocampus


def test_lessons_filters_and_teacher_reply(tmp_path, monkeypatch):
    db = tmp_path / "brain.sqlite"
    h = Hippocampus(db)
    h.add([Experience("school", 1, "preschool", "world.facts", "teacher", "h1", "<|human|> Sky?\n<|student|>",
                      "<|answer|> blue<|eos|>", "red", False, True, teacher_reply="Steps: The sky is blue.\nAnswer: blue"),
           Experience("school", 2, "preschool", "world.sizes", "code", "h2", "<|human|> Big?\n<|student|>",
                      "<|answer|> cow<|eos|>", "cow", True, True, asked_as="Which is bigger?")])
    monkeypatch.setattr(panel, "DB", db)
    assert panel.lessons({})["total"] == 2
    wrong = panel.lessons({"correct": "0"})
    assert wrong["total"] == 1 and wrong["rows"][0]["teacher_reply"].startswith("Steps:")
    assert panel.lessons({"rephrased": "1"})["rows"][0]["asked_as"] == "Which is bigger?"
    assert panel.lessons({"q": "cow", "day": "2"})["total"] == 1
    assert {s["skill"] for s in panel.skills(1)} == {"world.facts"}
