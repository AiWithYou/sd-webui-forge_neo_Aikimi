import sys
import unittest

sys.argv = sys.argv[:1]

from backend.text_processing.qwen3vl_engine import Qwen3VLTextProcessingEngine


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        input_ids = []
        for text in texts:
            image_tokens = [151655] * text.count("<|image_pad|>")
            input_ids.append([1, *image_tokens])
        return {"input_ids": input_ids}


class Qwen3VLTokenizeTests(unittest.TestCase):
    def make_engine(self):
        tokenizer = RecordingTokenizer()
        engine = object.__new__(Qwen3VLTextProcessingEngine)
        engine.tokenizer = tokenizer
        engine.llama_template = "text:{}"
        engine.image_template = "image:<|vision_start|><|image_pad|><|vision_end|>{}"
        engine.vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
        engine.id_image = 151655
        return engine, tokenizer

    def test_empty_text_uses_nonempty_text_template(self):
        engine, tokenizer = self.make_engine()

        tokenized = engine.tokenize([""])

        self.assertGreater(len(tokenized[0]), 0)
        self.assertEqual(["text: "], tokenizer.calls[0])

    def test_empty_text_with_images_uses_nonempty_image_template(self):
        engine, tokenizer = self.make_engine()

        tokenized = engine.tokenize([""], images=[object()])

        self.assertGreater(len(tokenized[0]), 0)
        self.assertEqual(
            ["image:<|vision_start|><|image_pad|><|vision_end|> "],
            tokenizer.calls[0],
        )

    def test_image_arguments_do_not_leak_between_calls(self):
        engine, tokenizer = self.make_engine()

        engine.tokenize(["first"], images=[object(), object()])
        engine.tokenize(["second"])

        self.assertEqual(
            "image:<|vision_start|><|image_pad|><|vision_end|>"
            "<|vision_start|><|image_pad|><|vision_end|>first",
            tokenizer.calls[0][0],
        )
        self.assertEqual("text:second", tokenizer.calls[1][0])

    def test_image_parameters_use_none_defaults(self):
        self.assertEqual((None,), Qwen3VLTextProcessingEngine.tokenize.__defaults__)
        self.assertEqual((None,), Qwen3VLTextProcessingEngine.tokenize_line.__defaults__)

    def test_plain_prompt_is_templated_once(self):
        engine, tokenizer = self.make_engine()

        chunks = engine.tokenize_line("a red cat")

        self.assertEqual([["text:a red cat"]], tokenizer.calls)
        self.assertEqual(1, len(chunks))
        self.assertEqual([1.0] * len(chunks[0].tokens), chunks[0].multipliers)

    def test_emphasis_syntax_is_literal_and_templated_once(self):
        engine, tokenizer = self.make_engine()

        chunks = engine.tokenize_line("a (red:1.2) cat")

        self.assertEqual([["text:a (red:1.2) cat"]], tokenizer.calls)
        self.assertEqual(1, len(chunks))
        self.assertEqual([1.0] * len(chunks[0].tokens), chunks[0].multipliers)

    def test_image_prompt_is_templated_once_and_replaces_each_image_token(self):
        engine, tokenizer = self.make_engine()
        images = [object(), object()]

        chunks = engine.tokenize_line("a (red:1.2) cat", images=images)

        self.assertEqual(
            [[
                "image:<|vision_start|><|image_pad|><|vision_end|>"
                "<|vision_start|><|image_pad|><|vision_end|>a (red:1.2) cat"
            ]],
            tokenizer.calls,
        )
        image_tokens = [
            token for token in chunks[0].tokens if isinstance(token, dict)
        ]
        self.assertEqual(2, len(image_tokens))
        self.assertIs(images[0], image_tokens[0]["data"])
        self.assertIs(images[1], image_tokens[1]["data"])
        self.assertEqual([1.0] * len(chunks[0].tokens), chunks[0].multipliers)

    def test_empty_line_keeps_nonempty_template(self):
        engine, tokenizer = self.make_engine()

        chunks = engine.tokenize_line("")

        self.assertEqual([["text: "]], tokenizer.calls)
        self.assertEqual(1, len(chunks))
        self.assertEqual([1.0] * len(chunks[0].tokens), chunks[0].multipliers)


if __name__ == "__main__":
    unittest.main()
