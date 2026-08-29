import unittest

from vllm_command import build_vllm_command


class BuildVllmCommandTest(unittest.TestCase):
    def build(self, **overrides):
        values = {
            "model_path": "/models/glm-ocr",
            "model_name": "zai-org/GLM-OCR",
            "port": 8080,
            "max_model_len": "16384",
            "gpu_memory_utilization": "0.95",
            "speculative_config": '{"method":"ngram"}',
            "enforce_eager": False,
        }
        values.update(overrides)
        return build_vllm_command(**values)

    def test_includes_speculative_config_when_set(self):
        command = self.build()
        index = command.index("--speculative-config")
        self.assertEqual(command[index + 1], '{"method":"ngram"}')

    def test_omits_speculative_config_when_empty(self):
        command = self.build(speculative_config="")
        self.assertNotIn("--speculative-config", command)

    def test_adds_batching_overrides(self):
        command = self.build(max_num_batched_tokens="8192", max_num_seqs="4")
        self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "8192")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "4")


if __name__ == "__main__":
    unittest.main()
