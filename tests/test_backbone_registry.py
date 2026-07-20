import unittest

from path_til.backbone_registry import SUPPORTED_BACKBONES, get_backbone_spec


class BackboneRegistryTests(unittest.TestCase):
    def test_supported_backbones_include_screening_targets(self):
        self.assertIn("efficientnetv2_s", SUPPORTED_BACKBONES)
        self.assertIn("convnext_tiny", SUPPORTED_BACKBONES)

    def test_get_backbone_spec_returns_copy(self):
        spec = get_backbone_spec("efficientnetv2_s")
        self.assertEqual(spec["input_size"], 224)
        spec["input_size"] = 999
        self.assertEqual(get_backbone_spec("efficientnetv2_s")["input_size"], 224)

    def test_unknown_backbone_raises(self):
        with self.assertRaises(ValueError):
            get_backbone_spec("unknown_backbone")
