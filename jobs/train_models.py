"""Job task: the rule-floor comparison runs, the per-point models (NBM), controller health and the
survival model, and the supervised training gate. Runs are logged to MLflow under the experiment root."""

from jobs.common import setup
from pdm import tasks


def main() -> None:
    store, cfg, args = setup("train_models")
    root = args.experiment_root
    tasks.rule_floor_task(store, cfg, f"{root}/rule_floor")
    tasks.nbm_task(store, cfg, f"{root}/nbm")
    tasks.controller_health_task(store, cfg, f"{root}/controller_health")
    tasks.survival_task(store, cfg, f"{root}/survival")
    tasks.supervised_task(store, cfg)


if __name__ == "__main__":
    main()
