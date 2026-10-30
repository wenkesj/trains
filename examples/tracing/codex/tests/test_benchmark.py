from unittest.mock import Mock

import pytest

from codex_tracing.benchmark import Exercise, validate_split

SOURCE_REVISION = "6105f6d9d3ef66ce3d5c8f529d2882af3d76b706"


def test_candidate_and_verifier_keep_tests_and_reference_solutions_separate(tmp_path, monkeypatch):
    source = tmp_path / "source/exercises/practice/example"
    (source / ".meta").mkdir(parents=True)
    (source / ".docs").mkdir()
    (source / ".meta/config.json").write_text(
        '{"files":{"solution":["solution.py"],"test":["solution_test.py"],"example":[".meta/example.py"]}}'
    )
    (source / ".docs/instructions.md").write_text("Implement a function")
    (source / ".meta/example.py").write_text("secret_reference_solution")
    (source / "solution.py").write_text("pass")
    (source / "solution_test.py").write_text("secret_held_out_test")
    monkeypatch.setattr(
        "codex_tracing.benchmark.subprocess.check_output", Mock(return_value=SOURCE_REVISION)
    )
    exercise = Exercise.load(tmp_path / "source", "example")
    assert exercise.revision == SOURCE_REVISION
    candidate = tmp_path / "candidate"
    exercise.candidate(candidate)
    assert sorted(p.name for p in candidate.iterdir()) == ["INSTRUCTIONS.md", "solution.py"]
    (candidate / "solution.py").write_text("real_candidate_answer")
    verifier = tmp_path / "verifier"
    exercise.verifier(candidate, verifier)
    assert (verifier / "solution.py").read_text() == "real_candidate_answer"
    assert (verifier / "solution_test.py").read_text() == "secret_held_out_test"
    assert not (verifier / ".meta").exists()


def test_verifier_refuses_symlinked_candidate_solution(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "solution.py").symlink_to(tmp_path / "elsewhere.py")
    exercise = Exercise("example", tmp_path, ("solution.py",), (), "", SOURCE_REVISION, "checksum")
    with pytest.raises(ValueError, match="regular solution"):
        exercise.verifier(candidate, tmp_path / "verifier")


@pytest.mark.parametrize(
    "train,evaluation", [((), ("one",)), (("one",), ("one",)), (("one", "one"), ("two",))]
)
def test_training_and_evaluation_splits_are_disjoint_and_unique(train, evaluation):
    with pytest.raises(ValueError):
        validate_split(train, evaluation)


def test_deleted_solution_remains_a_candidate_for_the_upstream_verifier(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (tmp_path / "solution_test.py").write_text("import solution\n")
    exercise = Exercise(
        "example",
        tmp_path,
        ("solution.py",),
        ("solution_test.py",),
        "",
        SOURCE_REVISION,
        "checksum",
    )
    verifier = tmp_path / "verifier"
    exercise.verifier(candidate, verifier)
    assert (verifier / "solution_test.py").read_text() == "import solution\n"
    assert not (verifier / "solution.py").exists()


def test_verifier_refuses_solution_under_a_symlinked_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "solution.py").write_text("outside_candidate_boundary")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "package").symlink_to(outside, target_is_directory=True)
    exercise = Exercise(
        "example", tmp_path, ("package/solution.py",), (), "", SOURCE_REVISION, "checksum"
    )
    with pytest.raises(ValueError, match="regular solution"):
        exercise.verifier(candidate, tmp_path / "verifier")
