import unittest
import sys

sys.argv = sys.argv[:1]

from backend.text_processing.qwen3vl_engine import Qwen3VLTextProcessingEngine


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return {"input_ids": [[index + 1 for index, _ in enumerate(text)] for text in texts]}


class Qwen3VLTokenizeTests(unittest.TestCase):
    def make_engine(self):
        tokenizer = RecordingTokenizer()
        engine = object.__new__(Qwen3VLTextProcessingEngine)
        engine.tokenizer = tokenizer
        engine.llama_template = "text:{}"
        engine.image_template = "image:<|vision_start|><|image_pad|><|vision_end|>{}"
        engine.vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
        return engine, tokenizer

    def test_empty_text_uses_text_template(self):
        engine, tokenizer = self.make_engine()

        tokenized = engine.tokenize([""])

        self.assertGreater(len(tokenized[0]), 0)
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual("text:", tokenizer.calls[0][0])

    def test_empty_text_with_images_uses_image_template(self):
        engine, tokenizer = self.make_engine()

        tokenized = engine.tokenize([""], images=[object()])

        self.assertGreater(len(tokenized[0]), 0)
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual("image:<|vision_start|><|image_pad|><|vision_end|>", tokenizer.calls[0][0])
