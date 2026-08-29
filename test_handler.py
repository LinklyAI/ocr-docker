import os
import sys
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

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


class PrepareImageForSdkTest(unittest.TestCase):
    def test_materializes_unresized_data_url_as_local_file(self):
        image_bytes = BytesIO()
        Image.new("RGB", (32, 24), color="white").save(image_bytes, format="PNG")
        raw = image_bytes.getvalue()

        with (
            patch.object(handler, "MAX_IMAGE_SIDE", 1900),
            patch.object(handler, "_read_image_bytes", return_value=raw),
        ):
            path, cleanup_paths = handler._prepare_image_for_sdk(
                "data:image/png;base64,ignored", "job-1"
            )

        try:
            self.assertEqual(cleanup_paths, [path])
            self.assertTrue(path.endswith(".png"))
            self.assertEqual(Path(path).read_bytes(), raw)
        finally:
            for cleanup_path in cleanup_paths:
                os.remove(cleanup_path)

    def test_materializes_original_when_resizing_is_disabled(self):
        image_bytes = BytesIO()
        Image.new("RGB", (16, 16), color="black").save(image_bytes, format="JPEG")
        raw = image_bytes.getvalue()

        with (
            patch.object(handler, "MAX_IMAGE_SIDE", 0),
            patch.object(handler, "_read_image_bytes", return_value=raw),
        ):
            path, cleanup_paths = handler._prepare_image_for_sdk(
                "https://example.com/image.jpg", "job-2"
            )

        try:
            self.assertEqual(cleanup_paths, [path])
            self.assertTrue(path.endswith(".jpg"))
            self.assertEqual(Path(path).read_bytes(), raw)
        finally:
            for cleanup_path in cleanup_paths:
                os.remove(cleanup_path)


if __name__ == "__main__":
    unittest.main()
