import json
from pathlib import Path
from types import SimpleNamespace

from devops.vast.check_vast_inventory import (
    VastOffer,
    _normalize_entries,
    execute,
    filter_offers,
    render_table,
    summarize_offers,
)


def test_normalize_entries_accepts_nested_listings() -> None:
    payload = {
        "listings": {
            "data": [
                {"id": 1, "gpu_name": "NVIDIA RTX 4090", "dph_total": 1.234},
            ]
        }
    }
    entries = _normalize_entries(payload)
    assert len(entries) == 1
    assert entries[0]["gpu_name"] == "NVIDIA RTX 4090"


def test_filter_offers_filters_by_gpu_term() -> None:
    entries = [
        {"id": 1, "gpu_name": "RTX 4090", "dph_total": 1.5},
        {"id": 2, "gpu_name": "RTX 3090", "dph_total": 1.0},
        {"id": 3, "gpu_name": "RTX 5090", "dph_total": 2.0},
    ]
    offers = filter_offers(entries, ["4090", "5090"])
    assert [offer.offer_id for offer in offers] == [1, 3]


def test_summarize_offers_returns_cheapest_offer() -> None:
    offers = [
        VastOffer(offer_id=1, gpu_name="RTX 4090", hourly_price=1.5, country="US"),
        VastOffer(offer_id=2, gpu_name="RTX 5090", hourly_price=1.1, country="DE"),
    ]
    summary = summarize_offers(offers)
    assert summary["count"] == 2
    assert summary["cheapest_offer_id"] == 2
    assert summary["cheapest_hourly_price"] == 1.1


def test_render_table_handles_empty_offers() -> None:
    output = render_table([], limit=5)
    assert "No matching Vast.dev offers found." in output


def test_execute_with_input_file(tmp_path: Path) -> None:
    payload = {
        "listings": [
            {"id": 1, "gpu_name": "RTX 4090", "dph_total": 1.2, "country": "US"},
            {"id": 2, "gpu_name": "RTX 5090", "dph_total": 1.0, "country": "DE"},
        ]
    }
    input_path = tmp_path / "listings.json"
    input_path.write_text(json.dumps(payload))

    args = SimpleNamespace(
        input=str(input_path),
        gpu_terms=["4090"],
        api_base="http://unused",
        api_key=None,
        timeout=5.0,
    )
    offers, summary = execute(args)
    assert len(offers) == 1
    assert summary["cheapest_offer_id"] == 1
