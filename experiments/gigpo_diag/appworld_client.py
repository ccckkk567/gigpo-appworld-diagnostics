"""Leave time to receive the native code-timeout feedback over HTTP.

AppWorld's default uses one timeout for both code execution and HTTP reads.
Only extend the execute request deadline; the server's execution limit, error
feedback, side effects and interaction accounting remain native. Never retry.
"""
from appworld import AppWorld

EXECUTE_HTTP_GRACE_SECONDS = 60


class AppWorldClient(AppWorld):
    def _remote_environment_call(self, method_name, **kwargs):
        if method_name != 'execute' or self.timeout_seconds is None:
            return super()._remote_environment_call(method_name, **kwargs)
        assert self.task_id is not None
        assert self.remote_environment_url is not None
        return self._remote_environment_call_helper(
            _task_id=self.task_id,
            _remote_environment_url=self.remote_environment_url,
            _timeout_seconds=self.timeout_seconds + EXECUTE_HTTP_GRACE_SECONDS,
            method_name=method_name,
            **kwargs,
        )
