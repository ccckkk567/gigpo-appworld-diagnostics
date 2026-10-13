"""Check the actual AppWorld monkey patch and FastAPI schema/request path."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path


def main():
    import appworld  # Must precede FastAPI: activates AppWorld's Pydantic patch.
    from appworld.common.datetime import DateTime
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel

    versions = {name: metadata.version(name) for name in ('appworld', 'pydantic', 'pydantic-core', 'fastapi', 'starlette')}
    if versions['pydantic'] != '2.12.5' or versions['fastapi'] != '0.141.1':
        raise RuntimeError(f'Unexpected AppWorld compatibility versions: {versions}')

    class Probe(BaseModel):
        text: str | None = None
        at: DateTime

    app = FastAPI()

    @app.post('/probe', response_model=Probe)
    def echo(payload: Probe):
        return payload

    with TestClient(app) as client:
        response = client.post('/probe', json={'text': None, 'at': '2026-01-01T12:00:00Z'})
        if response.status_code != 200 or response.json()['text'] is not None:
            raise RuntimeError(f'FastAPI request failed: {response.status_code} {response.text}')
        if not response.json()['at'].startswith('2026-01-01T12:00:00'):
            raise RuntimeError('Datetime did not round trip')
        if client.get('/openapi.json').status_code != 200:
            raise RuntimeError('OpenAPI schema generation failed')
    result = dict(status='passed', versions=versions, appworld_path=appworld.__file__,
                  checks=['AppWorld import before FastAPI', 'nullable and AppWorld DateTime model', 'HTTP request/response round trip', 'OpenAPI generation'],
                  environment_tasks_executed=0, training_updates=0)
    out = Path(os.environ['TMPDIR']) / 'appworld-compat-check.json'
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('APPWORLD_COMPAT_CHECK_PASSED ' + json.dumps(versions), flush=True)


if __name__ == '__main__':
    main()
