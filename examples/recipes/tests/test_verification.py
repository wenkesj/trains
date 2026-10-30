import json
import subprocess
import sys

from recipes.verification import UNITTEST_VERIFIER


def test_candidate_memory_and_time_errors_keep_native_test_results(tmp_path):
    (tmp_path / "budget_test.py").write_text(
        "import unittest\n"
        "class BudgetTest(unittest.TestCase):\n"
        "    def test_memory(self):\n"
        "        values = []\n"
        "        while True:\n"
        "            values.append(bytearray(1024 * 1024))\n"
        "    def test_timeout(self):\n"
        "        while True:\n"
        "            pass\n"
        "    def test_valid(self):\n"
        "        self.assertEqual(2 + 2, 4)\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", UNITTEST_VERIFIER],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    grade = json.loads(result.stdout.strip().splitlines()[-1])
    assert grade["tests"] == 3 and grade["failed"] == 2
    assert grade["reward"] == 1 / 3 and not grade["passed"]
    assert "MemoryError" in result.stderr and "TimeoutError" in result.stderr
