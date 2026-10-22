"""
Day 1: Validate the corpus manifest before any scraping happens.
Run this any time you edit corpus_manifest.yaml to catch scope drift early.
"""
import yaml
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent.parent / "config" / "corpus_manifest.yaml"


def load_manifest() -> dict:
    with open(MANIFEST_PATH, "r") as f:
        return yaml.safe_load(f)


def validate(manifest: dict) -> None:
    tiers = ["tier1_fundamentals", "tier2_filings", "tier2_india", "tier3_microstructure"]
    total_target = 0

    print("=== Corpus Manifest Validation ===\n")
    for tier_key in tiers:
        tier = manifest.get(tier_key)
        if tier is None:
            raise ValueError(f"Missing required tier: {tier_key}")

        target = tier["target_count"]
        total_target += target
        print(f"{tier_key}")
        print(f"  doc_type      : {tier['doc_type']}")
        print(f"  target_count  : {target}")
        print(f"  description   : {tier['description'].strip().splitlines()[0].strip()}")
        print()

    overall_target = manifest["target_total_docs"]
    print(f"Sum of tier targets : {total_target}")
    print(f"Declared total target: {overall_target}")

    # Raised from 50-150 on Day 16: the corpus grew on purpose (25 companies, 47 explainers).
    if not (50 <= total_target <= 300):
        raise ValueError(
            f"Total target docs ({total_target}) outside project bounds (50-300)."
        )

    if abs(total_target - overall_target) > 10:
        print(
            f"WARNING: tier targets sum ({total_target}) drifted from "
            f"declared total ({overall_target}). Update one or the other."
        )

    # Check metadata schema completeness
    required_fields = {"source", "tier", "doc_type", "title", "url", "date_fetched"}
    declared_fields = set(manifest["metadata_schema"]["fields"])
    missing = required_fields - declared_fields
    if missing:
        raise ValueError(f"metadata_schema missing required fields: {missing}")

    print("\n✅ Manifest valid. Ready for Day 2/3 scraping scripts.")


if __name__ == "__main__":
    manifest = load_manifest()
    validate(manifest)
