"""Release framework-owned exceptions only after native execute has returned.

This is reference hygiene, not a memory limit or a namespace reset.  The pinned
IPython retains failures in ExecutionResult, AutoFormattedTB.tb, SyntaxTB and
sys.last_*.  Dropping those owner references must not mutate the exception or
its frames: policy code may intentionally retain the same exception in user_ns.
"""
import functools
import gc
import sys
import threading


_EXECUTION_LOCK = threading.Lock()
_SYS_NAMES = ('last_type', 'last_value', 'last_traceback', 'last_exc')


def _owners(shell):
    owners = [(shell, 'last_execution_result')]
    for formatter_name in ('InteractiveTB', 'SyntaxTB'):
        formatter = getattr(shell, formatter_name, None)
        if formatter is not None:
            owners.extend((formatter, name) for name in ('tb', 'last_syntax_error')
                          if hasattr(formatter, name))
    return owners


def snapshot_exception_owners(shell):
    """Keep identities only; the snapshot must not itself retain an exception."""
    return {(id(owner), name): id(getattr(owner, name, None))
            for owner, name in _owners(shell)}


def clear_completed_execution_exceptions(shell, before):
    """Detach this execution's framework owners; return whether GC is useful.

    The caller must have completed native formatting, turn counting and saves.
    Only sys entries that match this shell's newly produced failure are cleared.
    No traceback, frame, ExecutionResult, or user object is mutated in place.
    """
    owned_exceptions = []
    owned_tracebacks = []
    changed = []
    for owner, name in _owners(shell):
        value = getattr(owner, name, None)
        if value is None or id(value) == before.get((id(owner), name)):
            continue
        if name == 'last_execution_result':
            errors = [getattr(value, key, None)
                      for key in ('error_before_exec', 'error_in_exec')]
            errors = [error for error in errors if isinstance(error, BaseException)]
            if not errors:
                continue  # Preserve successful ExecutionResult semantics.
            owned_exceptions.extend(errors)
        elif name == 'last_syntax_error':
            if not isinstance(value, BaseException):
                continue
            owned_exceptions.append(value)
        elif name == 'tb':
            owned_tracebacks.append(value)
        changed.append((owner, name))

    if not changed:
        return False

    # sys.last_traceback can refer to an interior traceback node because native
    # AppWorld subsequently re-raises ExecutionResult.error to format feedback.
    traceback_ids = {id(traceback) for traceback in owned_tracebacks}
    for exception in owned_exceptions:
        traceback = exception.__traceback__
        while traceback is not None:
            traceback_ids.add(id(traceback))
            traceback = traceback.tb_next
    exception_ids = {id(exception) for exception in owned_exceptions}
    value_owned = any(id(getattr(sys, name, None)) in exception_ids
                      for name in ('last_value', 'last_exc'))
    if value_owned and getattr(sys, 'last_type', None) in {
            type(exception) for exception in owned_exceptions}:
        sys.last_type = None
    for name in ('last_value', 'last_exc'):
        if id(getattr(sys, name, None)) in exception_ids:
            setattr(sys, name, None)
    if id(getattr(sys, 'last_traceback', None)) in traceback_ids:
        sys.last_traceback = None
    for owner, name in changed:
        setattr(owner, name, None)
    return True


def install_exception_cleanup():
    """Install only in independent, serial AppWorld environment services."""
    from appworld import AppWorld

    original = AppWorld.execute
    if getattr(original, '_gigpo_exception_cleanup', False):
        return

    @functools.wraps(original)
    def execute(self, code):
        if self.remote_environment_url or self.shell is None:
            return original(self, code)
        # sys.last_* and SafetyGuard are process-global.  Concurrent/reentrant
        # local execution is unsupported; fail before running an action twice.
        if not _EXECUTION_LOCK.acquire(blocking=False):
            raise RuntimeError('Concurrent local AppWorld execute is unsupported')
        try:
            before = snapshot_exception_owners(self.shell)
            message = original(self, code)
            # Deliberately not in finally: a native persistence/infrastructure
            # exception must propagate with its traceback and no fake success.
            released = clear_completed_execution_exceptions(self.shell, before)
            if released:
                # The helper's temporary strong references are gone now.  A
                # traceback/result cycle can be reclaimed without clearing frames.
                gc.collect()
            return message
        finally:
            _EXECUTION_LOCK.release()

    execute._gigpo_exception_cleanup = True
    AppWorld.execute = execute
    print('APPWORLD_EXCEPTION_CLEANUP enabled native_execute_then_cleanup=True '
          'user_namespace_preserved=True', flush=True)
