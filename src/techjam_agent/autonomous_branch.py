"""Validation and storage for LLM-generated model branches.

The research loop is intentionally open-ended at the *hypothesis* level: a
planner may submit a small Python implementation instead of selecting one of
the built-in operators.  Generated code is still subject to a conservative
static gate and is executed only by the existing isolated worker.  This is a
defence-in-depth boundary, not a claim that Python can be made a perfect
sandbox by AST inspection alone.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any


MAX_SOURCE_CHARS = 24_000
MAX_BRANCH_NAME_CHARS = 80
BRANCH_DIRECTORY = Path("artifacts") / "code-branches"
REQUIRED_FUNCTIONS = {"fit_validate": 3, "finalize": 4}
ALLOWED_IMPORTS = {"math", "numpy", "torch", "typing", "collections"}
FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
}
FORBIDDEN_ATTRIBUTES = {
    "system", "popen", "remove", "unlink", "rmtree", "run", "Popen",
    "check_call", "check_output", "call", "urlopen", "request", "connect",
    "send", "recv", "load", "dump", "loads", "dumps", "save", "savetxt",
    "savez", "savez_compressed", "mkdir", "write_text", "write_bytes",
    "read_text", "read_bytes", "replace", "rename", "rmdir", "chmod",
}


class CodeBranchError(ValueError):
    """Raised when a generated branch fails the local contract or safety gate."""


class AutonomousRuntime:
    """Small capability facade exposed to generated code.

    The underlying runner is intentionally kept in a private slot. The source
    gate rejects private attribute access, so a branch can use only these
    explicitly documented operations and cannot reach the Controller, raw data
    modules, or test labels.
    """

    __slots__ = ("_owner", "_allow_finalize")

    def __init__(self, owner: Any, *, allow_finalize: bool = False) -> None:
        self._owner = owner
        self._allow_finalize = bool(allow_finalize)

    def autonomous_encoded(self, config: dict[str, Any], split: str = "train_valid"):
        return self._owner.autonomous_encoded(config, split)

    def autonomous_dense_matrices(self, config: dict[str, Any], split: str = "train_valid"):
        return self._owner.autonomous_dense_matrices(config, split)

    def autonomous_evaluate(self, users: Any, labels: Any, scores: Any) -> dict[str, Any]:
        return self._owner.autonomous_evaluate(users, labels, scores)

    def autonomous_write_validation_slices(self, checkpoint: Path, scores: Any) -> None:
        return self._owner.autonomous_write_validation_slices(checkpoint, scores)

    def autonomous_save_checkpoint(self, checkpoint: Path, payload: Any) -> None:
        return self._owner.autonomous_save_checkpoint(checkpoint, payload)

    def autonomous_load_checkpoint(self, checkpoint: Path) -> Any:
        return self._owner.autonomous_load_checkpoint(checkpoint)

    def autonomous_write_submission(self, scores: Any, output: Path) -> dict[str, Any]:
        if not self._allow_finalize:
            raise RuntimeError("submission writing is unavailable during validation training")
        return self._owner.autonomous_write_submission(scores, output)

    def autonomous_run_builtin(
        self, model_name: str, config: dict[str, Any], checkpoint: Path
    ) -> dict[str, Any]:
        return self._owner.autonomous_run_builtin(model_name, config, checkpoint)

    def autonomous_finalize_builtin(
        self, model_name: str, config: dict[str, Any], checkpoint: Path, output: Path
    ) -> dict[str, Any]:
        if not self._allow_finalize:
            raise RuntimeError("finalization is unavailable during validation training")
        return self._owner.autonomous_finalize_builtin(model_name, config, checkpoint, output)


def _branch_name(value: Any) -> str:
    name = str(value or "generated-branch").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    if not name:
        name = "generated-branch"
    return name[:MAX_BRANCH_NAME_CHARS]


def _syntax_gate(source: str) -> ast.Module:
    if not isinstance(source, str) or not source.strip():
        raise CodeBranchError("code_branch.source must be a non-empty string")
    if len(source) > MAX_SOURCE_CHARS:
        raise CodeBranchError(
            f"generated source exceeds the {MAX_SOURCE_CHARS} character limit"
        )
    try:
        tree = ast.parse(source, filename="generated_branch.py", mode="exec")
    except SyntaxError as exc:
        raise CodeBranchError(f"generated source is not valid Python: {exc}") from exc

    functions = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name, minimum_args in REQUIRED_FUNCTIONS.items():
        node = functions.get(name)
        if node is None or isinstance(node, ast.AsyncFunctionDef):
            raise CodeBranchError(f"generated source must define synchronous {name}(...)")
        positional = [*node.args.posonlyargs, *node.args.args]
        if len(positional) < minimum_args:
            raise CodeBranchError(
                f"{name} must accept at least {minimum_args} positional arguments"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    raise CodeBranchError("relative imports are not allowed in generated branches")
                modules = [node.module or ""]
            for module in modules:
                root = module.split(".", 1)[0]
                if root not in ALLOWED_IMPORTS:
                    raise CodeBranchError(
                        f"generated branch import {module!r} is outside the allow-list"
                    )
        if isinstance(node, ast.Name) and (
            node.id in FORBIDDEN_NAMES or node.id.startswith("__")
        ):
            raise CodeBranchError(f"generated branch uses forbidden name {node.id!r}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
                raise CodeBranchError(
                    f"generated branch uses forbidden attribute {node.attr!r}"
                )
        if isinstance(node, ast.Call):
            target = node.func.id if isinstance(node.func, ast.Name) else None
            if target in FORBIDDEN_NAMES:
                raise CodeBranchError(f"generated branch calls forbidden function {target!r}")
    return tree


def validate_source(source: str) -> str:
    """Validate and return source in a canonical text form."""
    _syntax_gate(source)
    return source.replace("\r\n", "\n").replace("\r", "\n")


def _safe_branch_path(root: Path, value: str | Path) -> Path:
    base = (root / BRANCH_DIRECTORY).resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise CodeBranchError("code branch path must stay under artifacts/code-branches") from exc
    if candidate.suffix != ".py":
        raise CodeBranchError("code branch path must point to a .py file")
    return candidate


def materialize_code_branch(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically persist a validated branch and return portable metadata."""
    if not isinstance(payload, dict):
        raise CodeBranchError("code_branch must be an object")
    source = validate_source(payload.get("source"))
    name = _branch_name(payload.get("branch_name"))
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    directory = root / BRANCH_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    # The filename is content-addressed so resubmitting the same branch (even
    # with a different display name) cannot create a second executable.
    source_path = directory / f"{digest[:16]}.py"
    manifest_path = source_path.with_suffix(".json")
    if source_path.is_file():
        existing = source_path.read_text(encoding="utf-8")
        if existing != source:
            raise CodeBranchError("code branch hash collision with different source")
    else:
        temporary = source_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(source, encoding="utf-8")
        temporary.replace(source_path)
    manifest = {
        "branch_name": name,
        "sha256": digest,
        "source_path": str(source_path.relative_to(root)).replace("\\", "/"),
        "base_model": payload.get("base_model"),
        "description": str(payload.get("description") or "").strip(),
        "source_characters": len(source),
    }
    temporary_manifest = manifest_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def load_code_branch(root: Path, value: str | Path, expected_sha256: str | None = None) -> ModuleType:
    """Load a branch after rechecking its path, hash, syntax, and entry points."""
    path = _safe_branch_path(root, value)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CodeBranchError(f"unable to read code branch {path}") from exc
    source = validate_source(source)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if expected_sha256 and digest != str(expected_sha256):
        raise CodeBranchError("code branch hash does not match the recorded manifest")
    module_name = f"techjam_generated_{digest[:16]}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CodeBranchError(f"unable to import code branch {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in REQUIRED_FUNCTIONS:
        function = getattr(module, name, None)
        if not callable(function):
            raise CodeBranchError(f"generated branch entry point {name!r} is not callable")
    return module
