"""Regression checks for the observed Orion setup failures; no GPU installs."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

HERE = Path(__file__).parent


def test_module_pythonpath_is_detected_and_isolated_python_ignores_it(tmp_path):
    # The test runner may intentionally expose its host's packages. Construct a
    # default venv, matching the actual Orion setup, for the isolation experiment.
    clean_venv = tmp_path / 'clean-venv'
    subprocess.run([sys.executable, '-I', '-m', 'venv', str(clean_venv)], check=True, capture_output=True)
    clean_python = clean_venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    external = tmp_path / 'module' / 'site-packages'
    fake_pip = external / 'pip'
    fake_pip.mkdir(parents=True)
    (fake_pip / '__init__.py').write_text('__version__="module-pip"\n', encoding='utf-8')
    env = dict(os.environ, GIGPO_VENV=str(clean_venv), PYTHONPATH=str(external))
    command = [str(clean_python), str(HERE / 'check_python_environment.py')]
    leaked = subprocess.run(command, env=env, capture_output=True, text=True)
    assert leaked.returncode != 0 and 'External packages visible' in leaked.stderr
    isolated = subprocess.run([str(clean_python), '-I', *command[1:]], env=env, capture_output=True, text=True)
    assert isolated.returncode == 0, isolated.stderr
    env.pop('PYTHONPATH')
    clean = subprocess.run(command, env=env, capture_output=True, text=True)
    assert clean.returncode == 0, clean.stderr


def test_constraints_accepted_by_pip(tmp_path):
    # An installed seed package avoids downloads. pip validates constraint syntax
    # before resolving it (the old file failed on ray[default]).
    result = subprocess.run([
        sys.executable, '-I', '-m', 'pip', 'install', '--dry-run', '--no-index',
        '--no-deps', '--disable-pip-version-check', '-c', str(HERE / 'constraints-runtime.txt'),
        'pip',
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('failure,expected_attempts,expected_status', [
    ('index_once', 2, 0), ('index_always', 3, 1), ('conflict', 1, 1),
])
def test_install_retries_transient_index_errors_only(tmp_path, failure, expected_attempts, expected_status):
    bash = os.environ.get('GIGPO_TEST_BASH') or shutil.which('bash')
    if not bash:
        pytest.skip('bash required for installer regression')
    bin_dir = tmp_path / 'venv' / 'bin'
    bin_dir.mkdir(parents=True)
    python_stub = bin_dir / 'python'
    python_stub.write_text('''#!/usr/bin/env bash
set -eu
count=0
if [[ -f "$TMPDIR/attempts" ]]; then count=$(cat "$TMPDIR/attempts"); fi
count=$((count+1))
printf '%s' "$count" > "$TMPDIR/attempts"
case "$GIGPO_TEST_FAILURE" in
  index_once) if [[ "$count" -gt 1 ]]; then exit 0; fi ;;
  conflict) echo 'ERROR: ResolutionImpossible: conflicting dependencies'; exit 1 ;;
esac
echo 'WARNING: Skipping page https://pypi.org/simple/datasets/ Content-Type: Unknown'
exit 1
''', encoding='utf-8', newline='\n')
    python_stub.chmod(0o755)
    script = '\n'.join([
        'set -euo pipefail',
        'export GIGPO_VENV=' + shlex.quote((tmp_path / 'venv').as_posix()),
        'export TMPDIR=' + shlex.quote(tmp_path.as_posix()),
        'source ' + shlex.quote((HERE / 'pip_install.sh').as_posix()),
        'sleep() { :; }',
        'gigpo_pip_install fixture-only',
    ])
    result = subprocess.run([bash, '-c', script], env=dict(os.environ, GIGPO_TEST_FAILURE=failure), capture_output=True, text=True)
    assert result.returncode == expected_status, result.stdout + result.stderr
    assert int((tmp_path / 'attempts').read_text()) == expected_attempts
