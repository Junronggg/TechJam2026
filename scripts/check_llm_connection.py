from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.local_env import load_local_env


def main() -> int:
    load_local_env(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Check the configured LLM without printing secrets")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    args = parser.parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(json.dumps({"ok": False, "error": "OPENAI_API_KEY is not configured"}))
        return 2

    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": "Return JSON: {\"ok\": true}"}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 32,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(json.dumps({"ok": False, "provider": args.base_url, "model": args.model,
                          "error": f"HTTP {exc.code}"}))
        return 1
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "provider": args.base_url, "model": args.model,
                          "error": type(exc).__name__}))
        return 1

    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    print(json.dumps({
        "ok": bool(isinstance(payload, dict) and payload.get("choices")),
        "provider": args.base_url,
        "model": args.model,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
