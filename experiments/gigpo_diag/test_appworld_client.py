"""Real AppWorld client/HTTP behavior, synthetic replies; no task rollouts."""
import importlib.util
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import requests
from appworld import AppWorld

spec=importlib.util.spec_from_file_location('appworld_client_patch', Path(__file__).with_name('appworld_client.py'))
repair=importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def client(cls=repair.AppWorldClient, timeout=100, url='http://127.0.0.1:12345'):
    # Exercise native remote execute and interaction accounting without task DBs.
    world=cls.__new__(cls)
    world.task_id='synthetic-task'
    world.remote_environment_url=url
    world.timeout_seconds=timeout
    world.num_interactions=0
    world.num_sub_interactions=0
    world.environment_io=[]
    world.save_logs=mock.Mock()
    return world


def response(output, status=200):
    r=requests.Response()
    r.status_code=status
    r._content=json.dumps({'output':output}).encode()
    return r


class AppWorldClientTests(unittest.TestCase):
    def test_execute_feedback_preserves_input_and_consumes_one_turn(self):
        world=client()
        code='while True: pass'
        output='Execution failed. Traceback:\nException: Execution timed out after 100 seconds.'
        with mock.patch('requests.post',return_value=response(output)) as post:
            self.assertEqual(world.execute(code),output)
        post.assert_called_once_with('http://127.0.0.1:12345/execute',
                                     json={'task_id':'synthetic-task','code':code},timeout=160)
        self.assertEqual(world.timeout_seconds,100)
        self.assertEqual(world.num_interactions,1)
        self.assertEqual(world.environment_io,[dict(number='1',input=code,output=output)])
        world.save_logs.assert_called_once()

    def test_initialize_keeps_server_execution_timeout_and_other_rpc_deadlines(self):
        world=client()
        with mock.patch('requests.post',return_value=response('ok')) as post:
            world._remote_environment_call('initialize',timeout_seconds=100)
            world._remote_environment_call('task_completed')
        self.assertEqual(post.call_args_list[0].kwargs['timeout'],100)
        self.assertEqual(post.call_args_list[0].kwargs['json']['timeout_seconds'],100)
        self.assertEqual(post.call_args_list[1].kwargs['timeout'],100)

    def test_transport_failures_are_not_retried_or_counted_as_completed_turns(self):
        for error in (requests.ReadTimeout('synthetic stall'),requests.ConnectionError('synthetic disconnect')):
            with self.subTest(error=type(error).__name__):
                world=client()
                with mock.patch('requests.post',side_effect=error) as post:
                    with self.assertRaises(Exception): world.execute('x += 1')
                post.assert_called_once()
                self.assertEqual(world.num_interactions,0)
                self.assertEqual(world.environment_io,[])
                world.save_logs.assert_not_called()

    def test_http_500_still_fails_without_retry(self):
        world=client()
        with mock.patch('requests.post',return_value=response('server failure',500)) as post:
            with self.assertRaisesRegex(Exception,'failed'): world.execute('x = 1')
        post.assert_called_once()
        self.assertEqual(world.num_interactions,0)

    def test_no_timeout_mode_remains_unchanged(self):
        world=client(timeout=None)
        with mock.patch('requests.post',return_value=response('ok')) as post:
            world.execute('x = 1')
        self.assertIsNone(post.call_args.kwargs['timeout'])

    def test_success_feedback_matches_native_client(self):
        worlds=[client(AppWorld),client()]
        with mock.patch('requests.post',return_value=response('Execution successful.')):
            outputs=[world.execute('x = 1') for world in worlds]
        self.assertEqual(outputs[0],outputs[1])
        self.assertEqual(worlds[0].environment_io,worlds[1].environment_io)

    def test_real_http_read_can_receive_late_failure_without_reexecuting(self):
        # Short scaled timing: native client deadline 0.1s, server reply 0.3s.
        # This fixture does not run an actual 100s infinite loop or task DBs.
        events=[]
        output='Execution failed. Traceback:\nException: Execution timed out after 100 seconds.'
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                events.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                time.sleep(.3)
                payload=json.dumps({'output':output}).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(payload)))
                self.end_headers()
                try: self.wfile.write(payload)
                except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError): pass
            def log_message(self,*args): pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}'
            old=client(AppWorld,timeout=.1,url=url)
            with self.assertRaises(requests.ReadTimeout): old.execute('test_payload')
            new=client(timeout=.1,url=url)
            with mock.patch.object(repair,'EXECUTE_HTTP_GRACE_SECONDS',1):
                self.assertEqual(new.execute('test_payload'),output)
            self.assertEqual(len(events),2)  # exactly one request from each client
            self.assertEqual(new.num_interactions,1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__=='__main__':
    unittest.main()
