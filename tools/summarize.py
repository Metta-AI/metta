#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel  # Works with google-generativeai package

def main() -> None:
    ap = argparse.ArgumentParser(prog="summarize", description="Summarise a codebase using Pydantic AI")
    ap.add_argument("paths", nargs="+", help="Paths to feed into codeclip")
    ap.add_argument("--max_tokens", type=int, default=10000, help="Max completion tokens")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"), help="LLM model name")
    ap.add_argument("--copy", action="store_true", help="Copy output to clipboard as well as stdout")
    args = ap.parse_args()

    # 1) collect context
    context = subprocess.run(["python", "-m", "manybot.codebot.codeclip.codeclip.cli", *args.paths], check=True, text=True, capture_output=True).stdout

    # 2) load prompt and substitute token cap
    prompt_path = Path(__file__).parent / "summarize_prompt.md"
    instructions = prompt_path.read_text(encoding="utf-8").replace("{{MAX_TOKENS}}", str(args.max_tokens))

    # 3) build agent
    model = GeminiModel(args.model)
    agent = Agent(model, instructions=instructions)

    # per-run settings include max_tokens
    model_settings = {"max_tokens": args.max_tokens, "temperature": 0.2}

    # 4) run the model
    user_message = (
        "Summarise the following codeclip context. "
        "Do not include any content outside the requested markdown summary.\n\n"
        f"{context}"
    )
    result = agent.run_sync(user_message, model_settings=model_settings)

    # 5) output
    if args.copy:
        import pyperclip  # type: ignore
        pyperclip.copy(result.output)
        sys.stderr.write("[summarize] copied to clipboard\n")
    print(result.output)

if __name__ == "__main__":
    main()
