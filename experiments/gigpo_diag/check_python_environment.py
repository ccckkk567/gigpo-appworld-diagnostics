"""Fail before installation/training if module packages leak into the venv."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def inspect_environment(expected_prefix, expected_pip=None):
    import pip

    prefix = Path(sys.prefix).resolve()
    expected = Path(expected_prefix).resolve()
    if prefix != expected or sys.prefix == sys.base_prefix:
        raise RuntimeError(f'Wrong virtual environment: {prefix}; expected {expected}')
    leaked = [str(p) for entry in sys.path if entry
              for p in [Path(entry).resolve()]
              if ('site-packages' in p.parts or 'dist-packages' in p.parts)
              and not p.is_relative_to(prefix)]
    pip_file = Path(pip.__file__).resolve()
    distribution = importlib.metadata.distribution('pip')
    distribution_path = Path(distribution.locate_file('')).resolve()
    if leaked or not pip_file.is_relative_to(prefix) or not distribution_path.is_relative_to(prefix):
        raise RuntimeError(f'External packages visible: {leaked}; pip={pip_file}; metadata={distribution_path}')
    if expected_pip and (pip.__version__ != expected_pip or distribution.version != expected_pip):
        raise RuntimeError(f'pip version mismatch: imported={pip.__version__}, metadata={distribution.version}, expected={expected_pip}')
    return {'python': sys.executable, 'prefix': str(prefix), 'pip_file': str(pip_file),
            'pip_version': pip.__version__, 'pip_metadata_version': distribution.version,
            'external_site_packages': leaked}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pip-version')
    args = parser.parse_args()
    print('PYTHON_ENVIRONMENT ' + json.dumps(inspect_environment(os.environ['GIGPO_VENV'], args.pip_version)), flush=True)
