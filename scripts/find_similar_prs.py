import argparse
import sys
import textwrap
from pathlib import Path

from metta.tools.pr_similarity import API_KEY_ENV, DEFAULT_CACHE_PATH, DEFAULT_TOP_K, find_similar_prs, require_api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Embed an issue description and return the most similar PRs based on cached embeddings."),
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Path to the PR embedding cache (default: {DEFAULT_CACHE_PATH}).",
    )
    parser.add_argument(
        "--model",
        help="Embedding model identifier (default: inferred from cache metadata).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of similar PRs to return (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "description",
        help="Issue description to embed and compare against cached PR embeddings.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color codes in the output.",
    )
    return parser.parse_args()


def summarize_description(description: str, limit: int = 240) -> str:
    clean = " ".join(description.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def supports_color(args: argparse.Namespace) -> bool:
    if args.no_color:
        return False
    return sys.stdout.isatty()


def colorize(enabled: bool, code: str, text: str) -> str:
    if not enabled:
        return text
    RESET = "\033[0m"
    return f"{code}{text}{RESET}"


def format_description_block(text: str, width: int = 90, indent: int = 12, label: str = "Summary") -> str:
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent=" " * indent)
    wrapped = wrapper.fill(text)
    if not wrapped:
        return ""
    lines = wrapped.splitlines()
    if lines:
        lines[0] = f"{label}: {lines[0]}"
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    try:
        api_key = require_api_key(API_KEY_ENV)
    except EnvironmentError as error:
        sys.exit(f"error: {error}")
    description = args.description.strip()

    metadata, top_results = find_similar_prs(
        description,
        top_k=args.top_k,
        cache_path=args.cache_path,
        model_override=args.model,
        api_key=api_key,
    )
    model_name = args.model or metadata.model or "<unknown>"

    use_color = supports_color(args)
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"

    header = colorize(use_color, BOLD + CYAN, f"Top {len(top_results)} similar PRs")
    print(f"{header} {colorize(use_color, CYAN, f'(model={model_name})')}")
    print(colorize(use_color, CYAN, "─" * 72))

    for rank, (score, record) in enumerate(top_results, start=1):
        summary = summarize_description(record.description)
        summary_block = format_description_block(summary) if summary else ""

        title_line = f"{rank}. PR #{record.pr_number} — {record.title}"
        score_line = f"score={score:.4f}"
        print(colorize(use_color, BOLD + YELLOW, title_line))
        print(colorize(use_color, GREEN, f"    {score_line}"))

        meta_line = (
            f"    Author: {record.author} · Additions: {record.additions} · "
            f"Deletions: {record.deletions} · Files changed: {record.files_changed}"
        )
        print(colorize(use_color, MAGENTA, meta_line))

        if summary_block:
            for line in summary_block.splitlines():
                print(colorize(use_color, BLUE, f"    {line}"))

        commit_line = f"    Commit: {record.commit_sha} · Authored at: {record.authored_at}"
        print(colorize(use_color, CYAN, commit_line))
        print(colorize(use_color, CYAN, "-" * 72))


if __name__ == "__main__":
    main()
