"""Champion / challenger promotion.

After training a challenger, compare it to the current Production model on the
held-out metric. Promote only if it beats the champion by promote_min_delta.
This is the gate that lets retraining run automatically without quietly
shipping a worse model.
"""
from __future__ import annotations

from mlflow import MlflowClient

from .config import Config, load_config


def latest_version(client: MlflowClient, name: str, stage: str | None = None):
    versions = client.search_model_versions(f"name='{name}'")
    if stage:
        versions = [v for v in versions if v.current_stage == stage]
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def _metric(client: MlflowClient, version, metric: str) -> float:
    run = client.get_run(version.run_id)
    return run.data.metrics.get(metric, float("-inf"))


def promote_if_better(cfg: Config) -> str:
    client = MlflowClient(tracking_uri=cfg.mlops.mlflow_tracking_uri)
    name = cfg.mlops.registered_model_name
    metric = cfg.mlops.promote_metric

    challenger = latest_version(client, name)
    if challenger is None:
        return "no model registered yet"

    champion = latest_version(client, name, stage="Production")
    chal_score = _metric(client, challenger, metric)

    if champion is None:
        client.transition_model_version_stage(name, challenger.version,
                                               "Production")
        return (f"promoted v{challenger.version} -> Production "
                f"(first model, {metric}={chal_score:.3f})")

    champ_score = _metric(client, champion, metric)
    if chal_score >= champ_score + cfg.mlops.promote_min_delta:
        client.transition_model_version_stage(name, champion.version, "Archived")
        client.transition_model_version_stage(name, challenger.version,
                                               "Production")
        return (f"promoted v{challenger.version} (challenger {metric}="
                f"{chal_score:.3f} > champion {champ_score:.3f})")

    client.transition_model_version_stage(name, challenger.version, "Archived")
    return (f"kept champion v{champion.version} ({metric}={champ_score:.3f}); "
            f"challenger {chal_score:.3f} did not clear gate")


if __name__ == "__main__":
    print("[evaluate]", promote_if_better(load_config()))
