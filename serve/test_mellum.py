"""Mellum's JSON calls are selected only by its own tokenizer family."""
import unittest
from serve.frontend import OutputParser, parse_mellum_tool_call

class MellumCalls(unittest.TestCase):
    TOOLS=[{'name':'write','parameters':{'type':'object','properties':{'text':{'type':'string'}}}}]
    def parse(self, text, chunk=1, declared=True):
        p=OutputParser(thinking=False,tools=self.TOOLS if declared else None,declared_only=True,
                       stream_tools=True,call_format='mellum')
        ev=[]
        for i in range(0,len(text),chunk):ev+=p.feed(text[i:i+chunk])
        return ev+p.finish()
    def test_json_calls_across_stream_boundaries(self):
        text='<tool_call>{"name":"write","arguments":{"text":"literal </tool_call> and \\\"quote\\\""}}</tool_call>'
        for chunk in [1,7,len(text)]:
            ev=self.parse(text,chunk);calls=[e.call for e in ev if e.kind=='tool_call']
            self.assertEqual([(c.name,c.arguments) for c in calls],[('write',{'text':'literal </tool_call> and "quote"'})])
            self.assertEqual([e.call.id for e in ev if e.kind=='tool_start'],[calls[0].id])
    def test_unknown_or_undeclared_tool_stays_text(self):
        for text in ['<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
                     '<tool_call>{"name":"write","arguments":{}}</tool_call>']:
            ev=self.parse(text,declared=False)
            self.assertFalse([e for e in ev if e.kind=='tool_call'])
            self.assertEqual(''.join(e.text for e in ev if e.kind=='content'),text)
    def test_malformed_calls_are_not_executed(self):
        for raw in ['{"name":"write","arguments":', '[]', '{"name":"write","arguments":123}']:
            self.assertEqual(parse_mellum_tool_call(raw),[])
    def test_parallel_blocks(self):
        text='<tool_call>{"name":"write","arguments":{"text":"a"}}</tool_call>\n<tool_call>{"name":"write","arguments":{"text":"b"}}</tool_call>'
        self.assertEqual([e.call.arguments for e in self.parse(text) if e.kind=='tool_call'],[{'text':'a'},{'text':'b'}])

    def test_thinking_is_opened_by_the_model_not_the_prompt(self):
        p=OutputParser(thinking=True,call_format='mellum')
        ev=[]
        for char in '<think>work</think>answer':ev+=p.feed(char)
        ev+=p.finish()
        self.assertEqual(''.join(e.text for e in ev if e.kind=='reasoning'),'work')
        self.assertEqual(''.join(e.text for e in ev if e.kind=='content'),'answer')

    def test_answer_without_think_tag_is_content(self):
        p=OutputParser(thinking=True,call_format='mellum')
        ev=p.feed('answer')+p.finish()
        self.assertEqual(''.join(e.text for e in ev if e.kind=='content'),'answer')
