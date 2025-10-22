from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
except ImportError as exc:  # pragma: no cover - dependency may be optional
    genai = None
    genai_errors = None
    types = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from metta.tools.pr_similarity import API_KEY_ENV, require_api_key

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_OUTPUT_PATH = Path("bug_tlog.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate a markdown bug TLOG by sending the PR history to Gemini 2.5 Flash."),
    )
    parser.add_argument(
        "pr_history",
        type=Path,
        help="Path to the pr_history JSON file to feed into Gemini.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write the markdown result (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name to call (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for Gemini (default: 0.4).",
    )
    parser.add_argument(
        "--max-input-lines",
        type=int,
        help=("Limit the attachment to the first N lines of JSON (after reformatting). Useful for quick iterations."),
    )
    return parser.parse_args()


def ensure_dependency_available() -> None:
    if genai is None or types is None or genai_errors is None:
        raise RuntimeError("google-genai is required. Install with `pip install google-genai`.") from IMPORT_ERROR


def read_pr_history(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _trim_entries_to_line_limit(
    entries: List[Dict[str, Any]],
    max_lines: int,
) -> Tuple[List[Dict[str, Any]], str]:
    limited: List[Dict[str, Any]] = []
    for entry in entries:
        candidate = limited + [entry]
        text = json.dumps(candidate, indent=2)
        line_count = text.count("\n") + 1
        if line_count > max_lines:
            break
        limited.append(entry)
    if not limited:
        return [], "[]\n"
    return limited, json.dumps(limited, indent=2) + "\n"


def prepare_payload(
    pr_entries: List[Dict[str, Any]],
    original_path: Path,
    max_input_lines: Optional[int],
) -> Tuple[List[Dict[str, Any]], str]:
    if max_input_lines is None:
        return pr_entries, original_path.read_text(encoding="utf-8")

    limited_entries, json_payload = _trim_entries_to_line_limit(pr_entries, max_input_lines)
    return limited_entries, json_payload


def build_system_prompt(pr_count: int) -> str:
    return (
        "Generate a TLOG from the PR history JSON. Focus on bug-related PRs whose lessons or patterns are "
        "likely relevant to future issues; ignore isolated fixes with no reusable insight. For each selected "
        "incident, capture what failed, how it was diagnosed, the implemented fix, and follow-up considerations. "
        "End with cross-incident themes. The returned text is written directly to a file, so omit greetings or "
        "meta commentary.\n"
        f"The attachment contains {pr_count} pull request entries; include only meaningful, reusable bug-related "
        "insights."
    )


def request_markdown(
    client: genai.Client,
    pr_entries: List[Dict[str, Any]],
    json_payload: str,
    model_name: str,
    temperature: float,
) -> str:
    prompt = build_system_prompt(len(pr_entries))
    payload_part = types.Part.from_text(
        text="PR history JSON excerpt:\n" + json_payload,
    )
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[payload_part, prompt],
            config=types.GenerateContentConfig(temperature=temperature),
        )
    except genai_errors.APIError as exc:  # pragma: no cover - surface clearer error
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc
    if not getattr(response, "text", None):
        raise RuntimeError("Gemini did not return any text.")
    return response.text


def write_output(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.pr_history.exists():
        raise FileNotFoundError(f"PR history file not found: {args.pr_history}")

    ensure_dependency_available()

    try:
        api_key = require_api_key(API_KEY_ENV)
    except EnvironmentError as error:
        raise RuntimeError(str(error)) from error

    pr_entries = read_pr_history(args.pr_history)
    selected_entries, json_payload = prepare_payload(
        pr_entries=pr_entries,
        original_path=args.pr_history,
        max_input_lines=args.max_input_lines,
    )

    client = genai.Client(api_key=api_key)

    markdown = request_markdown(
        client=client,
        pr_entries=selected_entries,
        json_payload=json_payload,
        model_name=args.model,
        temperature=args.temperature,
    )

    write_output(args.output, markdown)
    if args.max_input_lines is not None and len(selected_entries) < len(pr_entries):
        print(
            f"Used {len(selected_entries)} entries due to --max-input-lines={args.max_input_lines}.",
            file=sys.stderr,
        )
    print(f"Wrote Gemini response to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
