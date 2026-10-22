"""Job task: prognosis, fusion and advisory over the scored record (batch scoring)."""

from jobs.common import setup
from pdm import tasks


def main() -> None:
    store, cfg, _ = setup("score")
    tasks.prognosis_task(store, cfg)
    tasks.fusion_advisory_task(store, cfg)


if __name__ == "__main__":
    main()
