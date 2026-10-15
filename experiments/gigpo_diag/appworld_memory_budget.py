"""Reserve address space for native save/evaluate outside policy cell execution.

The kernel hard limit always remains in force. Only its soft limit changes,
on the synchronous service's main thread, around the native cell runner. This
is resource containment, not a security boundary against hostile Python.
"""
import functools
import threading


def install_cell_memory_budget(policy_bytes=None, service_bytes=None):
    import resource
    from appworld import AppWorld

    if policy_bytes is None or service_bytes is None:
        from experiments.gigpo_diag.appworld_resource import load_profile
        profile = load_profile()
        policy_bytes = profile['policy_as_bytes']
        service_bytes = profile['service_as_bytes']
    if not 0 < policy_bytes < service_bytes:
        raise ValueError('Cell AS budget must leave a positive native reserve')
    expected = (service_bytes, service_bytes)
    if resource.getrlimit(resource.RLIMIT_AS) != expected:
        raise RuntimeError('Native reserve requires the exact installed service hard cap')
    original = AppWorld._shell_run_cell
    installed = getattr(original, '_gigpo_cell_memory_budget', None)
    if installed is not None:
        if installed != (policy_bytes, service_bytes):
            raise RuntimeError('Cell memory budget already installed with different limits')
        return

    @functools.wraps(original)
    def bounded(self, code):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Cell AS budget requires synchronous main-thread execution')
        if resource.getrlimit(resource.RLIMIT_AS) != expected:
            raise RuntimeError('Service AS limit changed outside the cell budget owner')
        # The policy budget is fixed for the entire run, not current usage plus
        # an allowance. Lowering it does not delete any existing user objects.
        resource.setrlimit(resource.RLIMIT_AS, (policy_bytes, service_bytes))
        try:
            return original(self, code)
        finally:
            # Restore before native execute formats/saves and before evaluate.
            # Never convert an evaluator failure into an ordinary zero reward.
            resource.setrlimit(resource.RLIMIT_AS, expected)

    bounded._gigpo_cell_memory_budget = (policy_bytes, service_bytes)
    AppWorld._shell_run_cell = bounded
    print(f'APPWORLD_CELL_MEMORY_BUDGET policy_as_bytes={policy_bytes} '
          f'service_as_bytes={service_bytes} native_reserve_bytes={service_bytes-policy_bytes}',
          flush=True)
