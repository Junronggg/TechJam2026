from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from techjam_agent.local_env import load_local_env


class LocalEnvTests(unittest.TestCase):
    def test_loads_values_and_preserves_existing_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# comment\nOPENAI_API_KEY='local-secret'\n"
                "OPENAI_MODEL=openai/gpt-4.1-mini\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-secret"}, clear=True):
                load_local_env(path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "shell-secret")
                self.assertEqual(os.environ["OPENAI_MODEL"], "openai/gpt-4.1-mini")

    def test_rejects_malformed_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("NOT_AN_ASSIGNMENT\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid .env entry"):
                load_local_env(path)


if __name__ == "__main__":
    unittest.main()
