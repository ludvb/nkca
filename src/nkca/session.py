from typing import Any, TypeVar

import attr
from effecthandlers import Handler, Message, ReturnValue, send
from effecthandlers_logging.logging import NoHandlerError


@attr.define
class GetSessionItem(Message):
    name: str


@attr.define
class SetSessionItem(Message):
    name: str
    value: Any
    include_in_checkpoint: bool = True


@attr.define
class GetCheckpoint(Message):
    pass


class Session(Handler):
    def __init__(self, **values: Any):
        self._session_dict: dict[str, tuple[Any, bool]] = {
            k: (v, True) for k, v in values.items()
        }

    def handle(self, message: Message):
        match message:
            case GetSessionItem(name):
                if name in self._session_dict:
                    value, _ = self._session_dict[name]
                    return ReturnValue(value)
            case SetSessionItem(name, value, include_in_checkpoint):
                self._session_dict[name] = (value, include_in_checkpoint)
            case GetCheckpoint():
                try:
                    session = send(message, interpret_final=False)
                except NoHandlerError:
                    session = {}
                return ReturnValue(
                    session | {k: v for k, (v, p) in self._session_dict.items() if p}
                )


def get(name: str) -> Any:
    try:
        return send(GetSessionItem(name))
    except NoHandlerError:
        raise ValueError(f"Session item {name} not found") from None


T = TypeVar("T")


def set(name: str, value: T, include_in_checkpoint: bool = True) -> T:
    try:
        send(SetSessionItem(name, value, include_in_checkpoint))
    except NoHandlerError:
        pass
    return value


def get_or_set(name: str, value: T, include_in_checkpoint: bool = True) -> T:
    try:
        return get(name)
    except ValueError:
        set(name, value, include_in_checkpoint)
        return value


def checkpoint() -> dict[str, Any]:
    return send(GetCheckpoint())
