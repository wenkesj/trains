# Resource-bounded execution of unchanged upstream unittest suites.

UNITTEST_VERIFIER = """
import json
import resource
import signal
import unittest

memory_limit = 256 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

def expired(signum, frame):
    raise TimeoutError('Candidate exceeded the unittest time budget')

signal.signal(signal.SIGALRM, expired)
signal.setitimer(signal.ITIMER_REAL, 5)
suite = unittest.defaultTestLoader.discover('.', pattern='*_test.py')
signal.setitimer(signal.ITIMER_REAL, 0)
budget = min(1.0, 40.0 / max(suite.countTestCases(), 1))

class BoundedResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        signal.setitimer(signal.ITIMER_REAL, budget)

    def stopTest(self, test):
        signal.setitimer(signal.ITIMER_REAL, 0)
        super().stopTest(test)

result = unittest.TextTestRunner(verbosity=2, resultclass=BoundedResult).run(suite)
total = result.testsRun
failed = len(result.errors) + len(result.failures) + len(result.skipped)
print(json.dumps({
    'tests': total,
    'failed': failed,
    'reward': (total-failed)/total if total else 0,
    'passed': bool(total and not failed),
    'verifier': {'memory_limit_bytes': memory_limit, 'test_timeout_seconds': budget},
}))
"""
