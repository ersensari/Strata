"""Optional backend boundaries, including original Qwen defaults and declared tools."""
import unittest
from serve.frontend import OutputParser, openai_to_messages
from serve.server import engine_args

class PolyBackend(unittest.TestCase):
    def test_vision_flag_is_only_added_for_the_original_engine(self):
        for exe, expected in [('strata', True), ('strata-poly', False)]:
            cfg={'exe':exe,'args':['--model','x.gguf'],'vision':{'gpu':False}}
            self.assertEqual('--vision' in engine_args(cfg), expected)

    def test_mtp_off_reaches_the_embedded_poly_drafter(self):
        self.assertIn('--no-draft', engine_args({'exe':'strata-poly','args':[], 'mtp':False}))

    def test_original_converter_defaults_are_preserved(self):
        req={'messages':[{'role':'assistant','tool_calls':[{'id':'c1','function':{'name':'f','arguments':'{}'}}]}]}
        plain=openai_to_messages(req)[0]
        poly=openai_to_messages(req,preserve_tool_ids=True)[0]
        self.assertNotIn('id',plain[0]['tool_calls'][0])
        self.assertEqual(poly[0]['tool_calls'][0]['id'],'c1')

    def test_undeclared_poly_calls_stay_text(self):
        for fmt,text in [('gemma','<|tool_call>call:missing{}<tool_call|>'),
                         ('glm','<tool_call>missing</tool_call>')]:
            with self.subTest(fmt=fmt):
                p=OutputParser(thinking=False,tools=[{'name':'allowed'}],stream_tools=True,
                               declared_only=True,call_format=fmt)
                events=[]
                for c in text:events+=p.feed(c)
                events+=p.finish()
                self.assertFalse(any(e.kind.startswith('tool_') for e in events))
                self.assertEqual(''.join(e.text for e in events if e.kind=='content'),text)
