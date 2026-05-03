import os
import unittest
from unittest.mock import patch

from Orcar.gen_config import (
    get_model_name_for_routing,
    get_qwen_tokenizer_name,
    is_qwen_model,
    uses_openai_compatible_request,
)


class GenConfigModelRoutingTest(unittest.TestCase):
    def test_provider_prefixed_model_routes_by_leaf_name(self):
        self.assertEqual(
            get_model_name_for_routing("dashscope/qwen3-8b"),
            "qwen3-8b",
        )
        self.assertEqual(
            get_model_name_for_routing("anthropic/claude-3-5-sonnet-20241022"),
            "claude-3-5-sonnet-20241022",
        )

    def test_qwen_detection_handles_namespaces_and_qwq(self):
        self.assertTrue(is_qwen_model("dashscope/qwen3-8b"))
        self.assertTrue(is_qwen_model("Qwen/QwQ-32B"))
        self.assertFalse(is_qwen_model("openai/gpt-4o"))

    def test_openai_claude_haiku_uses_openai_compatible_request(self):
        self.assertTrue(
            uses_openai_compatible_request("openai/claude-haiku-4-5-20251001")
        )
        self.assertFalse(
            uses_openai_compatible_request("claude-haiku-4-5-20251001")
        )

    def test_qwen_tokenizer_uses_leaf_model_name(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                get_qwen_tokenizer_name("dashscope/qwen3-8b"),
                "Qwen/Qwen3-8B",
            )
            self.assertEqual(
                get_qwen_tokenizer_name("Qwen/Qwen2.5-Coder-7B-Instruct"),
                "Qwen/Qwen2.5-7B-Instruct",
            )


if __name__ == "__main__":
    unittest.main()
