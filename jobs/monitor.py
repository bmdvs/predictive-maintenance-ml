"""Job task: event-based evaluation for validation, the shadow quarter and July 2025 to June 2026,
then monthly event performance, drift and the retraining decision."""

from jobs.common import setup
from pdm import splits, tasks


def main() -> None:
    store, cfg, _ = setup("monitor")
    b = splits.boundaries(cfg)
    for name, bounds in (
        ("validation", b["validation"]),
        ("shadow", b["shadow"]),
        ("jul2025_jun2026", (b["validation"][0], b["shadow"][1])),
    ):
        tasks.evaluation_task(store, cfg, name, bounds, name)
    tasks.monitoring_task(store, cfg)


if __name__ == "__main__":
    main()
