from jarvis.tts import SentenceSplitter


def test_splits_on_period():
    sp = SentenceSplitter()
    results = []
    for char in "Ola mundo. Como vai?":
        sentence = sp.feed(char)
        if sentence:
            results.append(sentence)
    results.extend(sp.flush())
    assert results == ["Ola mundo.", "Como vai?"]


def test_splits_on_exclamation_and_question():
    sp = SentenceSplitter()
    results = []
    for char in "Sim! Nao? Talvez.":
        sentence = sp.feed(char)
        if sentence:
            results.append(sentence)
    results.extend(sp.flush())
    assert results == ["Sim!", "Nao?", "Talvez."]


def test_flush_returns_incomplete():
    sp = SentenceSplitter()
    for char in "Sem ponto final":
        sp.feed(char)
    remaining = sp.flush()
    assert remaining == ["Sem ponto final"]


def test_flush_empty_returns_nothing():
    sp = SentenceSplitter()
    assert sp.flush() == []


def test_ignores_abbreviation_dots():
    sp = SentenceSplitter()
    results = []
    for char in "O Sr. Silva chegou. Boa tarde.":
        sentence = sp.feed(char)
        if sentence:
            results.append(sentence)
    results.extend(sp.flush())
    assert results == ["O Sr. Silva chegou.", "Boa tarde."]
