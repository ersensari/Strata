"""Mellum regex boundaries must prevent merges that Qwen permits."""
import unittest
from tools.strata_tokenizer import Tokenizer, BYTE_TO_UNICODE

class MellumTokenizer(unittest.TestCase):
    def tokenizer(self, pre, merges):
        tokens=[BYTE_TO_UNICODE[b] for b in range(256)]
        for merge in merges:
            for item in [*merge.split(' '), merge.replace(' ', '')]:
                if item not in tokens:tokens.append(item)
        return Tokenizer(tokens,merges,[1]*len(tokens),pre=pre)

    def test_newline_cannot_merge_into_punctuation(self):
        merges=[', '+BYTE_TO_UNICODE[10]]
        mellum=self.tokenizer('mellum2',merges)
        qwen=self.tokenizer('qwen35',merges)
        self.assertEqual(mellum.encode(',\nx'),[ord(','),10,ord('x')])
        self.assertNotEqual(mellum.encode(',\nx'),qwen.encode(',\nx'))

    def test_single_digits_cannot_merge(self):
        tk=self.tokenizer('mellum2',['1 2','12 3'])
        self.assertEqual(tk.encode('123'),[49,50,51])

    def test_uppercase_contraction_is_not_qwen_contraction(self):
        merges=["' R","'R E"]
        mellum=self.tokenizer('mellum2',merges)
        qwen=self.tokenizer('qwen35',merges)
        self.assertNotEqual(mellum.encode("WE'RE"),qwen.encode("WE'RE"))

    def test_unicode_and_code_roundtrip(self):
        tk=self.tokenizer('mellum2',[])
        for text in ['İstanbul, 125.50 TL\n','def f(x):\n    return x + 123\n',"WE'RE don't e\u0301 😀"]:
            self.assertEqual(tk.decode(tk.encode(text)),text)
