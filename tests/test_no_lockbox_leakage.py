from pathlib import Path


def test_external_testset_not_used_in_training_configs():
    config_dir = Path("configs")
    forbidden = "dataset/Testset"
    for path in config_dir.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if "external" in path.name:
            continue
        assert forbidden not in text, "{0} should not use Testset for training".format(path)
