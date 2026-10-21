import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools.poly_import import plan

class ImportPlan(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.model=self.root/'model.gguf';self.model.touch()
        self.engine=self.root/'strata-poly';self.engine.touch();self.engine.chmod(0o700)
    def p(self,meta=None,**kw):
        meta=meta or {'general.architecture':'qwen35moe'}
        with patch('tools.poly_import.GGUFFile',return_value=SimpleNamespace(metadata=meta)):
            return plan(self.model,self.engine,'test-model',self.root/'configs',32768,0,12,**kw)
    def test_plan_does_not_write_or_start_and_uses_poly_flags(self):
        result=self.p();self.assertFalse(result['path'].exists());self.assertFalse(result['pack'].exists())
        self.assertIn('--model',result['config']['args']);self.assertNotIn('--vision',result['config']['args'])
        self.assertFalse(any(self.root.glob('configs/*')))
    def test_missing_shards_are_refused(self):
        with self.assertRaisesRegex(ValueError,'filename'):
            self.p({'general.architecture':'qwen35moe','split.count':2,'split.no':0})
        self.model=self.root/'model-00001-of-00002.gguf';self.model.touch()
        with self.assertRaisesRegex(ValueError,'missing model shard'):
            self.p({'general.architecture':'qwen35moe','split.count':2,'split.no':0})
    def test_existing_config_is_not_overwritten(self):
        target=self.root/'configs';target.mkdir();(target/'strata-test-model.json').write_text('keep')
        with self.assertRaisesRegex(ValueError,'already exists'):self.p()
        self.assertEqual((target/'strata-test-model.json').read_text(),'keep')
    def test_unsupported_architecture_is_refused(self):
        with self.assertRaisesRegex(ValueError,'architecture'):self.p({'general.architecture':'llama'})
    def test_vision_requires_both_files(self):
        mm=self.root/'mmproj.gguf';mm.touch()
        with self.assertRaisesRegex(ValueError,'both'):self.p(mmproj=mm)
