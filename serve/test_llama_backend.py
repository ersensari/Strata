"""HTTP contracts of the optional llama.cpp worker, without GPU inference."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest

from serve.llama_backend import LlamaEngine, backend_args
from serve.server import ByteTokenizer, ChatTemplate, Service, serve, ROOT


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.seen = []
        seen = self.seen

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                seen.append((self.path, json.loads(body), self.headers.get('Authorization')))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream' if json.loads(body).get('stream') else 'application/json')
                self.end_headers()
                if json.loads(body).get('slow'):
                    try:
                        for _ in range(100):
                            self.wfile.write(b'data: {"part":1}\n\n')
                            self.wfile.flush()
                            time.sleep(.05)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if json.loads(body).get('stream'):
                    self.wfile.write(b'data: {"part":1}\n\n')
                    self.wfile.flush()
                    time.sleep(.1)
                    self.wfile.write(b'data: [DONE]\n\n')
                else:
                    self.wfile.write(b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}')

        self.upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        engine = LlamaEngine.__new__(LlamaEngine)
        engine.port = self.upstream.server_port
        engine.key = 'private-worker-key'
        engine.lock = threading.RLock()
        engine.connections = set()
        engine.max_context = 32768
        engine.info = {'backend': 'llama.cpp'}
        engine.start = lambda: None
        engine.alive = lambda: True
        engine.last = {}
        self.svc = Service(engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"), 'minicpm-dspark')
        self.svc.api_key = 'front-key'
        self.svc.llama_backend = True
        self.front = serve(self.svc, host='127.0.0.1', port=0)

    def tearDown(self):
        self.front.shutdown()
        self.front.server_close()
        self.upstream.shutdown()
        self.upstream.server_close()

    def call(self, body, key='front-key', path='/v1/chat/completions'):
        conn = http.client.HTTPConnection('127.0.0.1', self.front.server_port, timeout=3)
        conn.request('POST', path, json.dumps(body), {'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
        response = conn.getresponse()
        result = response.status, response.read()
        conn.close()
        return result

    def test_auth_rejected_before_upstream(self):
        self.assertEqual(self.call({}, key='wrong')[0], 401)
        self.assertEqual(self.seen, [])

    def test_image_tools_sampling_and_reasoning_pass_through(self):
        body = {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}],
                'tools': [{'type':'function','function':{'name':'weather'}}], 'temperature': .6, 'reasoning_effort': 'none'}
        self.assertEqual(self.call(body)[0], 200)
        self.assertEqual(self.seen[0], ('/v1/chat/completions', body, 'Bearer private-worker-key'))
        self.assertEqual(self.svc.admission_reserved, 0)

    def test_stream_preserved(self):
        status, body = self.call({'stream': True})
        self.assertEqual(status, 200)
        self.assertEqual(body, b'data: {"part":1}\n\ndata: [DONE]\n\n')
        self.assertEqual(self.svc.admission_reserved, 0)

    def test_anthropic_forwarded(self):
        self.assertEqual(self.call({'messages': []}, path='/v1/messages')[0], 200)
        self.assertEqual(self.seen[0][0], '/v1/messages')

    def test_drain_rejects_before_upstream(self):
        self.svc.begin_drain()
        self.assertEqual(self.call({})[0], 503)
        self.assertEqual(self.seen, [])

    def test_upstream_failure_releases_admission(self):
        self.svc.engine.port = 1
        self.assertEqual(self.call({})[0], 502)
        self.assertEqual(self.svc.admission_reserved, 0)
        self.assertEqual(self.svc.active_cancels, set())

    def test_foreign_origin_rejected(self):
        self.svc.api_key = ''
        conn = http.client.HTTPConnection('127.0.0.1', self.front.server_port)
        conn.request('POST', '/v1/chat/completions', '{}', {'Authorization':'Bearer front-key','Origin':'https://untrusted.invalid','Content-Type':'application/json'})
        response = conn.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
        conn.close()
        self.assertEqual(self.seen, [])

    def test_disconnected_stream_releases_worker(self):
        conn = http.client.HTTPConnection('127.0.0.1', self.front.server_port)
        conn.request('POST', '/v1/chat/completions', '{"stream":true,"slow":true}', {'Authorization':'Bearer front-key','Content-Type':'application/json'})
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertTrue(response.read1(20))
        response.close()
        conn.close()
        deadline = time.monotonic() + 3
        while self.svc.admission_reserved and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertEqual(self.svc.admission_reserved, 0)
        self.assertEqual(self.svc.active_cancels, set())


class ArgumentTests(unittest.TestCase):
    def test_shared_context_and_draft_off(self):
        cfg = {'args':['--max-context','65536'], 'mtp':False, 'ui_vision_enabled':False,
               'llama_backend':{'context':32768,'args':['-c','32768','-md','draft.gguf','--spec-type','draft-dspark','--spec-draft-n-max','15','-ngld','99','--mmproj','vision.gguf']}}
        args, context = backend_args(cfg)
        self.assertEqual(context, 65536)
        self.assertEqual(args[args.index('-c')+1], '65536')
        self.assertNotIn('-md', args)
        self.assertNotIn('--mmproj', args)
        self.assertEqual(args[args.index('--spec-type')+1], 'none')


if __name__ == '__main__':
    unittest.main()
