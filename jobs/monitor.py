"""Job task: validation evaluation, then monitoring, skipped while the shadow quarter is held out
(pdm.entrypoints.monitor)."""

from pdm.entrypoints import monitor

if __name__ == "__main__":
    monitor()
