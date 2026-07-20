from pathlib import Path

import yaml

from path_til.candidate import CANDIDATE_SETTINGS, default_method_config_path, reference_metrics


def test_candidate_reference_metrics_present():
    metrics = reference_metrics()
    assert metrics["positive_auc"] > 0.8
    assert metrics["positive_prc"] > 0.3


def test_default_method_config_exists():
    path = default_method_config_path("configs")
    assert path.is_file()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["hnscc_ratio"] == 0.50
    assert payload["tcga_ratio"] == 0.50
    assert CANDIDATE_SETTINGS["pretrained_path"]
