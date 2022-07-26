from __future__ import annotations

from typing import Any

class TranslatorLogger:
    def __new__(cls, *args: Any, **kwargs: Any): ...
    logger_name: str = ...
    def __init__(self) -> None: ...
    @property
    def logger(self): ...
    @property
    def verbosity_level(self): ...
    @verbosity_level.setter
    def verbosity_level(self, level: Any) -> None: ...
    @property
    def transformed_code_level(self): ...
    @transformed_code_level.setter
    def transformed_code_level(self, level: Any) -> None: ...
    @property
    def need_to_echo_log_to_stdout(self): ...
    @need_to_echo_log_to_stdout.setter
    def need_to_echo_log_to_stdout(self, log_to_stdout: Any) -> None: ...
    @property
    def need_to_echo_code_to_stdout(self): ...
    @need_to_echo_code_to_stdout.setter
    def need_to_echo_code_to_stdout(self, code_to_stdout: Any) -> None: ...
    def check_level(self, level: Any): ...
    def has_code_level(self, level: Any): ...
    def has_verbosity(self, level: Any): ...
    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...
    def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None: ...
    def log(self, level: Any, msg: Any, *args: Any, **kwargs: Any) -> None: ...
    def log_transformed_code(
        self, level: Any, ast_node: Any, transformer_name: Any, *args: Any, **kwargs: Any
    ) -> None: ...

def set_verbosity(level: int = ..., also_to_stdout: bool = ...) -> None: ...
def set_code_level(level: Any = ..., also_to_stdout: bool = ...) -> None: ...
