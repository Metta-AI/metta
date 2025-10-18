from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import requests  # type: ignore[import-untyped]

DEFAULT_API_BASE = "https://cloud.vast.ai/api/v0"
DEFAULT_GPUS = ("4090", "5090")


class VastError(RuntimeError):
    """Raised when the Vast API cannot be queried successfully."""


@dataclass(slots=True)
class VastOffer:
    """Normalized view of a Vast listing."""

    offer_id: int
    gpu_name: str
    hourly_price: float
    num_gpus: int | None = None
    country: str | None = None
    reliability: float | None = None
    storage_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise VastError("Unexpected payload type returned from Vast API")

    if {"code", "message"}.issubset(payload.keys()):
        raise VastError(payload.get("message", "Vast API returned an error"))

    candidates = []
    if isinstance(payload.get("listings"), list):
        candidates = payload["listings"]
    elif isinstance(payload.get("listings"), dict):
        maybe = payload["listings"].get("data")
        if isinstance(maybe, list):
            candidates = maybe
    elif isinstance(payload.get("data"), list):
        candidates = payload["data"]
    elif isinstance(payload.get("offers"), list):
        candidates = payload["offers"]

    if not candidates:
        raise VastError("Unable to locate listings in Vast API response")
    return [entry for entry in candidates if isinstance(entry, dict)]


def _extract_first_numeric(entry: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _to_offer(entry: dict[str, Any]) -> VastOffer | None:
    gpu_name = entry.get("gpu_name") or entry.get("gpu_desc") or entry.get("gpu") or ""
    gpu_name = str(gpu_name)
    if not gpu_name:
        return None

    offer_id_value = entry.get("id") or entry.get("machine_id") or entry.get("offer_id")
    if offer_id_value is None:
        return None
    try:
        offer_id = int(offer_id_value)
    except (TypeError, ValueError):
        return None

    hourly_price = _extract_first_numeric(
        entry,
        ("dph_total", "dph", "total_hourly_cost", "hourly_price", "cost", "price"),
    )
    if hourly_price is None:
        return None

    num_gpus = _extract_first_numeric(entry, ("gpu_total", "num_gpus", "gpu_count"))
    country = entry.get("country") or entry.get("region") or entry.get("country_code")
    reliability = _extract_first_numeric(entry, ("reliability", "rep", "score"))
    storage_gb = _extract_first_numeric(entry, ("storage", "disk_space_gb", "disk_gb"))

    return VastOffer(
        offer_id=offer_id,
        gpu_name=gpu_name,
        hourly_price=hourly_price,
        num_gpus=int(num_gpus) if num_gpus is not None else None,
        country=str(country) if country else None,
        reliability=reliability,
        storage_gb=storage_gb,
    )


def fetch_vast_listings(
    api_base: str,
    api_key: str | None,
    timeout: float,
    params: dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    api_base = api_base.rstrip("/")
    url = f"{api_base}/listings/public"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    session = session or requests.Session()
    try:
        response = session.get(url, headers=headers, params=params or {}, timeout=timeout)
    except requests.RequestException as exc:
        raise VastError(f"Failed to contact Vast API: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        raise VastError("Received HTML from Vast API. Ensure API base is correct and provide a VAST_API_KEY.")

    if response.status_code >= 400:
        snippet = response.text[:200]
        raise VastError(f"Vast API responded with HTTP {response.status_code}: {snippet}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise VastError("Vast API returned malformed JSON") from exc

    return _normalize_entries(payload)


def filter_offers(entries: Iterable[dict[str, Any]], gpu_terms: Iterable[str]) -> list[VastOffer]:
    terms = [term.lower() for term in gpu_terms]
    offers: list[VastOffer] = []
    for entry in entries:
        offer = _to_offer(entry)
        if not offer:
            continue
        name_lower = offer.gpu_name.lower()
        if any(term in name_lower for term in terms):
            offers.append(offer)
    return offers


def summarize_offers(offers: Sequence[VastOffer]) -> dict[str, Any]:
    if not offers:
        return {"count": 0, "cheapest_hourly_price": None, "cheapest_offer_id": None}

    cheapest = min(offers, key=lambda offer: offer.hourly_price)
    return {
        "count": len(offers),
        "cheapest_hourly_price": round(cheapest.hourly_price, 4),
        "cheapest_offer_id": cheapest.offer_id,
        "cheapest_country": cheapest.country,
        "cheapest_gpu_name": cheapest.gpu_name,
    }


def render_table(offers: Sequence[VastOffer], limit: int) -> str:
    if not offers:
        return "No matching Vast.dev offers found."

    lines = [f"{'Offer':>8}  {'GPU':<28} {'Price/hr':>8} {'GPUs':>4} {'Country':<8} {'Reliability':>11}"]
    for offer in sorted(offers, key=lambda item: item.hourly_price)[:limit]:
        reliability = f"{offer.reliability:.2f}" if offer.reliability is not None else "-"
        num_gpus = str(offer.num_gpus) if offer.num_gpus is not None else "-"
        country = offer.country or "-"
        lines.append(
            f"{offer.offer_id:>8}  {offer.gpu_name:<28.28} "
            f"{offer.hourly_price:>8.4f} {num_gpus:>4} {country:<8} {reliability:>11}"
        )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check Vast.ai for Dev boxes offering 4090/5090 GPUs.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Vast API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--gpu",
        action="append",
        dest="gpu_terms",
        help="GPU name fragment to search for (can be repeated). Defaults to 4090 and 5090.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of offers to display (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of human-readable text.",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Optional path to a JSON file containing pre-fetched Vast listings (skips API call).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Vast API key. Defaults to the VAST_API_KEY environment variable.",
    )
    return parser


def execute(args: argparse.Namespace) -> tuple[list[VastOffer], dict[str, Any]]:
    gpu_terms = args.gpu_terms or list(DEFAULT_GPUS)

    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = _normalize_entries(payload)
    else:
        api_key = args.api_key or os.getenv("VAST_API_KEY")
        entries = fetch_vast_listings(
            api_base=args.api_base,
            api_key=api_key,
            timeout=args.timeout,
        )

    offers = filter_offers(entries, gpu_terms)
    summary = summarize_offers(offers)
    return offers, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        offers, summary = execute(args)
    except VastError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")

    if args.json:
        output = {
            "summary": summary,
            "offers": [offer.to_dict() for offer in sorted(offers, key=lambda item: item.hourly_price)],
        }
        json.dump(output, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    lines = [
        f"Found {summary['count']} matching Vast.dev offers.",
    ]
    if summary["cheapest_hourly_price"] is not None:
        lines.append(
            "Cheapest offer "
            f"#{summary['cheapest_offer_id']} ({summary['cheapest_gpu_name']}) at "
            f"${summary['cheapest_hourly_price']}/hr in {summary.get('cheapest_country') or 'unknown'}."
        )
    lines.append("")
    lines.append(render_table(offers, args.limit))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
