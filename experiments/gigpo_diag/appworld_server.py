"""Launch the native AppWorld CLI with a narrow pre-execution parser repair.

The pinned AppWorld AST check can accept code that LibCST subsequently rejects.
Let native AppWorld.execute record that rejection and consume one interaction.
Do not run rejected code, change projection/rewards, or mask service failures.
"""
import functools
import runpy


def install_cst_rejection_handler():
    from appworld.common.safety_guard import SafetyGuard
    from libcst import CSTValidationError

    original = SafetyGuard.is_syntax_safe
    if getattr(original, '_gigpo_cst_rejection_handler', False):
        return

    @functools.wraps(original)
    def checked(self, code):
        try:
            return original(self, code)
        except CSTValidationError as exc:
            # This check runs before executing the submitted code. False takes
            # the existing native rejection path, with original input/log/turn.
            return False, f'CSTValidationError: {exc}'

    checked._gigpo_cst_rejection_handler = True
    SafetyGuard.is_syntax_safe = checked


def main():
    install_cst_rejection_handler()
    from experiments.gigpo_diag.appworld_execution_trace import install_execution_trace
    install_execution_trace()
    print('APPWORLD_CST_REJECTION_HANDLER enabled', flush=True)
    runpy.run_module('appworld.cli', run_name='__main__')


if __name__ == '__main__':
    main()
