from __future__ import annotations


class Conversation:
    def __init__(self, system_prompt: str, max_turns: int = 5) -> None:
        self._system = {"role": "system", "content": system_prompt}
        self._max_turns = max_turns
        self._turns: list[tuple[str, str]] = []

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self._turns.append((user_msg, assistant_msg))
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]

    def get_messages(self) -> list[dict[str, str]]:
        msgs = [self._system.copy()]
        for user_msg, assistant_msg in self._turns:
            msgs.append({"role": "user", "content": user_msg})
            msgs.append({"role": "assistant", "content": assistant_msg})
        return msgs

    def get_messages_for_api(self, new_user_msg: str) -> list[dict[str, str]]:
        msgs = self.get_messages()
        msgs.append({"role": "user", "content": new_user_msg})
        return msgs

    def clear(self) -> None:
        self._turns.clear()
