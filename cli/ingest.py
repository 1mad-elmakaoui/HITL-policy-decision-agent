"""CLI for the ZenML ingestion pipeline.

    python -m cli.ingest                 # run the ZenML pipeline
    python -m cli.ingest --local         # run the same steps without ZenML
    python -m cli.ingest --report        # show the current index and last eval
"""

from __future__ import annotations

import argparse
import json
import sys

from config.settings import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run policy-document ingestion.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="run the ingestion steps directly, without the ZenML orchestrator",
    )
    parser.add_argument(
        "--report", action="store_true", help="print index statistics and exit"
    )
    parser.add_argument(
        "--no-strict-verification",
        action="store_true",
        help="record verification findings as warnings instead of failing the run",
    )
    parser.add_argument("--config", default=None, help="path to a settings YAML file")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)

    if args.report:
        return _report(settings)

    if args.local:
        from ingestion.local_runner import run_ingestion_locally

        result = run_ingestion_locally(
            settings, strict_verification=not args.no_strict_verification
        )
        _print_summary(result)
        return 0

    try:
        from ingestion.pipelines.policy_ingestion import run_policy_ingestion
    except ImportError:
        print(
            "ZenML is not installed. Install it with:\n"
            "    pip install -e '.[ingestion]'\n"
            "or run the same steps without the orchestrator using --local.",
            file=sys.stderr,
        )
        return 2

    run_policy_ingestion(settings, strict_verification=not args.no_strict_verification)
    print("ZenML pipeline 'policy_ingestion' finished. Inspect runs with: zenml pipeline runs list")
    return _report(settings)


def _report(settings) -> int:
    from agent.retrieval.retriever import PolicyRetriever
    from shared.vector_store import VectorStoreUnavailable

    try:
        stats = PolicyRetriever.from_settings(settings).index_stats()
    except VectorStoreUnavailable as exc:
        print(f"Policy index unavailable: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=2))
    return 0


def _print_summary(result: dict) -> None:
    index = result.get("index", {})
    evaluation = result.get("evaluation", {})
    component = evaluation.get("component_level", {})

    print(f"Indexed {index.get('chunks_written', 0)} chunks "
          f"from {len(index.get('documents', []))} documents "
          f"into '{index.get('collection', '')}' ({index.get('backend', '')}).")
    if index.get("stale_chunks_removed"):
        print(f"Removed {index['stale_chunks_removed']} stale chunk(s) from earlier versions.")

    k = evaluation.get("k", 5)
    print("\nRetrieval evaluation (RAGOps component level):")
    for metric in (f"recall_at_{k}", f"precision_at_{k}", "mrr", f"ndcg_at_{k}"):
        if metric in component:
            print(f"  {metric:<16} {component[metric]:.3f}")

    calibration = evaluation.get("calibration", {})
    if calibration:
        print("\nEvidence-threshold calibration:")
        print(f"  weak / strong    {calibration.get('weak_evidence_score', 0):.2f} / "
              f"{calibration.get('strong_evidence_score', 0):.2f}")
        print(f"  on-corpus  top   {calibration.get('positive_min_top_score', 0):.2f} .. "
              f"{calibration.get('positive_max_top_score', 0):.2f}")
        print(f"  off-corpus max   {calibration.get('negative_max_top_score', 0):.2f} "
              f"({calibration.get('negatives_reaching_strong', 0)} reaching strong)")

    print(f"\n  gate: {'PASSED' if evaluation.get('passed') else 'FAILED'}")
    for failure in evaluation.get("failures", []):
        print(f"    - {failure}")


if __name__ == "__main__":
    raise SystemExit(main())