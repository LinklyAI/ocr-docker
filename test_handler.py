import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("runpod", SimpleNamespace(serverless=SimpleNamespace()))
sys.modules.setdefault("requests", SimpleNamespace())

import handler


class NormalizeSdkResultTest(unittest.TestCase):
    def test_reads_official_pipeline_result_fields(self):
        result = SimpleNamespace(
            json_result=[[{"label": "text", "content": "hello"}]],
            markdown_result="hello",
            to_dict=lambda: {
                "json_result": [[{"label": "text", "content": "hello"}]],
                "markdown_result": "hello",
            },
        )

        layout, markdown, raw = handler._normalize_sdk_result(result)

        self.assertEqual(len(layout[0]), 1)
        self.assertEqual(markdown, "hello")
        self.assertEqual(raw["markdown_result"], "hello")


class InitGlmOcrSdkTest(unittest.TestCase):
    def test_forces_local_selfhosted_mode(self):
        calls = []

        class FakeGlmOcr:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        with patch.dict(sys.modules, {"glmocr": SimpleNamespace(GlmOcr=FakeGlmOcr)}):
            parser = handler.init_glmocr_sdk()

        self.assertIsInstance(parser, FakeGlmOcr)
        self.assertEqual(
            calls,
            [
                {
                    "config_path": "/root/.config/glm-ocr/config.yaml",
                    "mode": "selfhosted",
                    "ocr_api_host": "localhost",
                    "ocr_api_port": handler.VLLM_PORT,
                    "layout_device": None,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
