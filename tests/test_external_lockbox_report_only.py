import unittest
from pathlib import Path

import yaml


def _training_relevant_text(payload) -> str:
    """Serialize config values excluding free-text notes/descriptions."""
    if not isinstance(payload, dict):
        return str(payload)
    parts = []
    for key, value in payload.items():
        if key in {"notes", "description"}:
            continue
        if isinstance(value, dict):
            parts.append(_training_relevant_text(value))
        elif isinstance(value, list):
            parts.extend(_training_relevant_text(item) for item in value)
        else:
            parts.append(str(value))
    return "\n".join(parts)


class ExternalLockboxReportOnlyTests(unittest.TestCase):
    def test_external_lockbox_not_used_in_training_configs(self):
        config_dir = Path("configs")
        for path in config_dir.glob("*.yaml"):
            if "external" in path.name:
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            text = _training_relevant_text(payload)
            self.assertNotIn(
                "dataset/Testset",
                text,
                msg="{0} should not use Testset for training".format(path),
            )


if __name__ == "__main__":
    unittest.main()
