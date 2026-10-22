import unittest

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from psycholing_metrics.surprisal.concatenated import ConcatenatedSurprisalExtractor


class TestChunkingTrailingWhitespace(unittest.TestCase):
    """BERT-style tokenizers drop trailing whitespace, so the last token's end
    offset falls short of len(text). The chunking loop must still recognise the
    last chunk instead of trying to start another one."""

    @classmethod
    def setUpClass(cls):
        # Real tokenizer of the affected model, paired with a tiny random model
        # so the test doesn't need the 1.5B checkpoint.
        model_name = "uer/gpt2-xlarge-chinese-cluecorpussmall"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch.manual_seed(0)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=32,
                n_embd=8,
                n_layer=1,
                n_head=1,
            )
        ).eval()

        extractor = ConcatenatedSurprisalExtractor.__new__(
            ConcatenatedSurprisalExtractor
        )
        extractor.tokenizer = tokenizer
        extractor.model = model
        extractor.model_name = model_name
        cls.extractor = extractor

    def _assert_covers_all_tokens(self, text, overlap_size):
        log_probs, offset_mapping = self.extractor.compute_surprisal(
            target_text=text, overlap_size=overlap_size, allow_overlap=True
        )
        expected = self.extractor.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        self.assertEqual(len(log_probs), len(expected))
        self.assertEqual([tuple(o) for o in offset_mapping], expected)

    def test_short_text(self):
        for suffix in ["", " ", "\n"]:
            with self.subTest(suffix=repr(suffix)):
                self._assert_covers_all_tokens("我今天去了学校。" + suffix, 512)

    def test_chunked_text(self):
        # ~80 tokens with a 32-token window forces several chunks.
        for suffix in ["", " ", "\n"]:
            with self.subTest(suffix=repr(suffix)):
                self._assert_covers_all_tokens("我今天去了学校。" * 10 + suffix, 8)


if __name__ == "__main__":
    unittest.main()
