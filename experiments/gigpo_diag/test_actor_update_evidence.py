"""CPU checks for failure preservation and strict verification evidence."""
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.gigpo_diag import actor_update_evidence as evidence


class UnpicklableCompilerError(RuntimeError):
    def __init__(self, backend, inner_exception):
        self.inner_exception = inner_exception
        super().__init__(f"{backend}: {inner_exception}")


class Config(dict):
    __getattr__ = dict.__getitem__


class ActorEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, {"GIGPO_ACTOR_VERIFY": "1", "GIGPO_RUN_DIR": str(self.root),
                                      "GIGPO_CODE_COMMIT": "test-commit"})
        env.start()
        self.addCleanup(env.stop)
        self.worker = SimpleNamespace(rank=0, config=SimpleNamespace(
            actor=Config(use_torch_compile=True, entropy_coeff=0.001),
            model=SimpleNamespace(use_remove_padding=True)))
        self.data = SimpleNamespace(meta_info={"diag_update_target": 1}, batch={})

    def test_disabled_observer_is_transparent(self):
        expected = object()
        with patch.dict(os.environ, {"GIGPO_ACTOR_VERIFY": "0"}):
            actual = evidence.record_actor_update(lambda worker, data: data)(object(), expected)
        self.assertIs(actual, expected)
        self.assertFalse(list(self.root.iterdir()))

    def test_original_unpicklable_failure_is_saved_before_reraise(self):
        try:
            raise FileNotFoundError("compiler-cache-fixture.so")
        except FileNotFoundError as inner:
            original = UnpicklableCompilerError("fixture_backend", inner)
        with self.assertRaises(TypeError):
            pickle.loads(pickle.dumps(original))

        def fail(worker, data):
            raise original

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(UnpicklableCompilerError) as caught:
            evidence.record_actor_update(fail)(self.worker, self.data)
        self.assertIs(caught.exception, original)
        saved = list((self.root / "actor-update").glob("*.traceback.txt"))
        self.assertEqual(len(saved), 1)
        raw = saved[0].read_text(encoding="utf-8")
        self.assertIn("Explicit inner_exception:", raw)
        self.assertIn("FileNotFoundError: compiler-cache-fixture.so", raw)
        with self.assertRaises(RuntimeError):
            evidence.verify(self.root)

    def test_failed_evidence_write_does_not_replace_original_error(self):
        original = ValueError("native failure")
        (self.root / "actor-update").write_text("not a directory")
        def fail(worker, data):
            raise original
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(ValueError) as caught:
            evidence.record_actor_update(fail)(self.worker, self.data)
        self.assertIs(caught.exception, original)

    def make_success(self):
        for rank in range(4):
            self.worker.rank = rank
            pid = 1000 + rank
            result = SimpleNamespace(meta_info={"metrics": {"actor/grad_norm": [0.5, 0.4]}})
            with patch.object(evidence.os, "getpid", return_value=pid):
                actual = evidence.record_actor_update(lambda worker, data: result)(self.worker, self.data)
                self.assertIs(actual, result)
            self.gzip_rows(f"optimizer.pid-{pid}.jsonl.gz", [{"optimizer_step_local": 1}])
        self.gzip_rows("updates.pid-2000.jsonl.gz", [{"policy_update": 1, "native_actor_update_returned": True}])

    def gzip_rows(self, filename, rows):
        (self.root / "diagnostics").mkdir(exist_ok=True)
        with gzip.open(self.root / "diagnostics" / filename, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def test_only_complete_four_rank_update_passes(self):
        self.make_success()
        with contextlib.redirect_stdout(io.StringIO()):
            evidence.verify(self.root)
        result = json.loads((self.root / "ACTOR_UPDATE_VERIFIED.json").read_text())
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["workers"]), 4)

    def test_missing_rank_rejected(self):
        self.make_success()
        (self.root / "actor-update" / "rank-3-pid-1003.jsonl").unlink()
        with self.assertRaises(RuntimeError):
            evidence.verify(self.root)

    def test_missing_optimizer_or_extra_driver_update_rejected(self):
        self.make_success()
        self.gzip_rows("optimizer.pid-1003.jsonl.gz", [])
        with self.assertRaises(RuntimeError):
            evidence.verify(self.root)
        self.gzip_rows("optimizer.pid-1003.jsonl.gz", [{"optimizer_step_local": 1}])
        self.gzip_rows("updates.pid-2000.jsonl.gz", [{"policy_update": 1, "native_actor_update_returned": True}] * 2)
        with self.assertRaises(RuntimeError):
            evidence.verify(self.root)

    def test_nonfinite_gradient_or_wrong_compile_path_rejected(self):
        self.make_success()
        path = self.root / "actor-update" / "rank-0-pid-1000.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for key, value, index in (("finite_grad_norms", False, 1), ("use_torch_compile", False, 0),
                                  ("use_remove_padding", False, 0), ("entropy_coeff", 0, 0)):
            with self.subTest(key=key):
                rows = [dict(row) for row in original]
                rows[index][key] = value
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(RuntimeError):
                    evidence.verify(self.root)
                self.assertFalse((self.root / "ACTOR_UPDATE_VERIFIED.json").exists())


if __name__ == "__main__":
    unittest.main()
