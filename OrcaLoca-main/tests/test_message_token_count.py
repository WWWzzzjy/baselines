import json
import unittest

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from Orcar.formatter import (
    TokenCounter,
    build_unique_message_sequence,
    normalize_chat_message_for_token_count,
)


class CharEncoding:
    def encode(self, text: str) -> list[str]:
        return list(text)


def make_counter() -> TokenCounter:
    counter = object.__new__(TokenCounter)
    counter.encoding = CharEncoding()
    return counter


class MessageTokenCountTest(unittest.TestCase):
    def test_normalize_chat_message_for_token_count_keeps_role_and_content_only(self):
        message = ChatMessage(
            role=MessageRole.USER,
            content="hello",
            additional_kwargs={"ignored": "metadata"},
        )

        self.assertEqual(
            normalize_chat_message_for_token_count(message),
            {
                "role": "user",
                "content": "hello",
            },
        )

    def test_build_unique_message_sequence_deduplicates_replayed_context(self):
        records = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "issue"},
                ],
                "response": "observation",
            },
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "issue"},
                    {"role": "assistant", "content": "observation"},
                    {"role": "tool", "content": "repo result"},
                ],
                "response": "done",
            },
        ]

        self.assertEqual(
            build_unique_message_sequence(records),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "issue"},
                {"role": "assistant", "content": "observation"},
                {"role": "tool", "content": "repo result"},
                {"role": "assistant", "content": "done"},
            ],
        )

    def test_count_messages_counts_the_serialized_message_list(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "issue"},
            {"role": "assistant", "content": "observation"},
        ]
        counter = make_counter()

        self.assertEqual(
            counter.count_messages(messages),
            len(json.dumps(messages, ensure_ascii=False)),
        )


if __name__ == "__main__":
    unittest.main()
