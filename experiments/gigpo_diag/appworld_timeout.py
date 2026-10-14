"""Keep AppWorld's cell deadline out of policy-level ``except Exception``.

This repairs ordinary exception-retry loops, not arbitrary uninterruptible native
code or policy code that deliberately catches BaseException / bare except.
"""
import functools
import signal
import threading


class _CellDeadlineExceeded(BaseException):
    """Service control flow; must not inherit from policy-level Exception."""


def run_cell_with_deadline(shell, code, seconds):
    seconds = int(seconds)
    if seconds <= 0:
        raise ValueError('AppWorld cell deadline must be positive')
    if not hasattr(signal, 'SIGALRM'):
        raise RuntimeError('AppWorld cell deadline requires POSIX SIGALRM')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('AppWorld cell deadline requires the main thread')
    # Do not silently replace another owner's timer. Native service execution is
    # synchronous and owns one cell timer at a time.
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('AppWorld cell deadline found an existing real timer')
    previous_handler = signal.getsignal(signal.SIGALRM)
    expired = False

    def on_deadline(signum, frame):
        nonlocal expired
        expired = True
        raise _CellDeadlineExceeded(f'Execution timed out after {seconds} seconds.')

    signal.signal(signal.SIGALRM, on_deadline)
    try:
        signal.alarm(seconds)
        try:
            result = shell.run_cell(raw_cell=code)
        except _CellDeadlineExceeded:
            return None
        # IPython normally catches BaseException and puts it in ExecutionResult.
        # None enters AppWorld.execute's existing timeout/log/save/turn path.
        return None if expired else result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def install_cell_deadline():
    from appworld import AppWorld

    original = AppWorld._shell_run_cell
    if getattr(original, '_gigpo_cell_deadline', False):
        return

    @functools.wraps(original)
    def bounded(self, code):
        self._maybe_raise_remote_environment_error('_shell_run_cell')
        if self.shell is None:
            raise RuntimeError('Interactive shell is not initialized.')
        if self.timeout_seconds is None:
            return original(self, code)
        return run_cell_with_deadline(self.shell, code, self.timeout_seconds)

    bounded._gigpo_cell_deadline = True
    AppWorld._shell_run_cell = bounded
    print('APPWORLD_CELL_DEADLINE enabled exception_catch=BaseException native_timeout_feedback=True', flush=True)
