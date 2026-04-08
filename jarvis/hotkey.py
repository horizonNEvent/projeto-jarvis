from __future__ import annotations

import asyncio

from pynput import keyboard


class HotkeyListener:
    """Listens for a global hotkey and signals an asyncio event."""

    def __init__(self, hotkey_str: str, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._event = asyncio.Event()
        self._hotkey_str = hotkey_str
        self._listener: keyboard.GlobalHotKeys | None = None

    def start(self) -> None:
        hotkey_combo = self._parse_hotkey(self._hotkey_str)
        self._listener = keyboard.GlobalHotKeys({hotkey_combo: self._on_trigger})
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()

    async def wait(self) -> None:
        await self._event.wait()
        self._event.clear()

    def _on_trigger(self) -> None:
        self._loop.call_soon_threadsafe(self._event.set)

    @staticmethod
    def _parse_hotkey(hotkey_str: str) -> str:
        parts = hotkey_str.lower().split("+")
        result = []
        for part in parts:
            part = part.strip()
            if part in ("ctrl", "alt", "shift"):
                result.append(f"<{part}>")
            else:
                result.append(part)
        return "+".join(result)
