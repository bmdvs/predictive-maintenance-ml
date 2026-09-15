"""Job task: event-based evaluation on validation (operating curves, the comparison at the vendor
operating point, diagnosis agreement), then monitoring, which is skipped while the shadow quarter is
held out."""

from jobs.common import setup
from pdm import tasks


def main() -> None:
    store, cfg, _ = setup("monitor")
    tasks.evaluation_task(store, cfg)
    tasks.monitoring_task(store, cfg)


if __name__ == "__main__":
    main()
