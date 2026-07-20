import yaml
from pathlib import Path


def test_source_pretrain_uses_train_and_test_only():
    path = Path("configs/source_pretrain_efficientnetv2_s.yaml")
    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    assert "train" in str(cfg["data"]["train_csv"]).lower()
    assert "test" in str(cfg["data"]["val_csv"]).lower()
    assert "Testset" not in text
