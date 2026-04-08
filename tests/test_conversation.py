from jarvis.conversation import Conversation

SYSTEM_PROMPT = "Voce e Jarvis."


def test_initial_messages_has_system_only():
    conv = Conversation(system_prompt=SYSTEM_PROMPT, max_turns=5)
    msgs = conv.get_messages()
    assert len(msgs) == 1
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_add_turn_and_retrieve():
    conv = Conversation(system_prompt=SYSTEM_PROMPT, max_turns=5)
    conv.add_turn("Ola", "Ola, como posso ajudar?")
    msgs = conv.get_messages()
    assert len(msgs) == 3
    assert msgs[1] == {"role": "user", "content": "Ola"}
    assert msgs[2] == {"role": "assistant", "content": "Ola, como posso ajudar?"}


def test_sliding_window_evicts_oldest():
    conv = Conversation(system_prompt=SYSTEM_PROMPT, max_turns=2)
    conv.add_turn("msg1", "resp1")
    conv.add_turn("msg2", "resp2")
    conv.add_turn("msg3", "resp3")
    msgs = conv.get_messages()
    # system + 2 turns (msg2, msg3) = 5 messages
    assert len(msgs) == 5
    assert msgs[1]["content"] == "msg2"
    assert msgs[3]["content"] == "msg3"


def test_get_messages_for_api_includes_new_user_msg():
    conv = Conversation(system_prompt=SYSTEM_PROMPT, max_turns=5)
    conv.add_turn("old question", "old answer")
    msgs = conv.get_messages_for_api("new question")
    assert len(msgs) == 4
    assert msgs[-1] == {"role": "user", "content": "new question"}


def test_clear_resets_to_system_only():
    conv = Conversation(system_prompt=SYSTEM_PROMPT, max_turns=5)
    conv.add_turn("a", "b")
    conv.clear()
    assert len(conv.get_messages()) == 1
