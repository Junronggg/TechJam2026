from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from techjam_agent.autonomous_branch import (
    CodeBranchError,
    load_code_branch,
    materialize_code_branch,
    validate_source,
)


SOURCE = '''
import numpy as np

def fit_validate(runner, config, checkpoint):
    return {"GAUC": 0.1, "nDCG@5": 0.2, "primary": 0.15}

def finalize(runner, config, checkpoint, output):
    return {"GAUC": 0.1, "nDCG@5": 0.2, "primary": 0.15}
'''


class AutonomousBranchTests(unittest.TestCase):
    def test_materialize_and_reload_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = materialize_code_branch(
                root, {"branch_name": "tiny branch", "source": SOURCE}
            )
            self.assertTrue((root / manifest["source_path"]).is_file())
            self.assertEqual(len(manifest["sha256"]), 64)
            module = load_code_branch(root, manifest["source_path"], manifest["sha256"])
            self.assertEqual(module.fit_validate(None, {}, None)["primary"], 0.15)
            again = materialize_code_branch(
                root, {"branch_name": "different name", "source": SOURCE}
            )
            self.assertEqual(manifest["sha256"], again["sha256"])

    def test_unsafe_import_and_path_are_rejected(self):
        with self.assertRaises(CodeBranchError):
            validate_source(
                "import os\n"
                "def fit_validate(a,b,c): pass\n"
                "def finalize(a,b,c,d): pass\n"
            )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CodeBranchError):
                load_code_branch(Path(directory), "../outside.py")
        with self.assertRaises(CodeBranchError):
            validate_source(
                "def fit_validate(a, b, c): return a._owner\n"
                "def finalize(a, b, c, d): return {}\n"
            )

    def test_entry_point_contract_is_required(self):
        with self.assertRaises(CodeBranchError):
            validate_source("def fit_validate(a, b, c): pass\n")


if __name__ == "__main__":
    unittest.main()
