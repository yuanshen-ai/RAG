import asyncio
import sys
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


def create_compatible_event_loop() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


def run_async(coro: Awaitable[T]) -> T:
    return asyncio.run(coro, loop_factory=create_compatible_event_loop)