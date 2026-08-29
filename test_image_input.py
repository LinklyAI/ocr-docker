import base64
import unittest

from image_input import decode_data_url


class DecodeDataUrlTest(unittest.TestCase):
    def test_decodes_base64_payload(self):
        payload = base64.b64encode(b"image-bytes").decode("ascii")
        self.assertEqual(
            decode_data_url(f"data:image/png;base64,{payload}"), b"image-bytes"
        )

    def test_decodes_percent_encoded_payload(self):
        self.assertEqual(decode_data_url("data:text/plain,hello%20world"), b"hello world")

    def test_rejects_non_data_url(self):
        with self.assertRaises(ValueError):
            decode_data_url("https://example.com/image.png")


if __name__ == "__main__":
    unittest.main()
