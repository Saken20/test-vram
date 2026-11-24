from vramtest.templating import drop_none, jsonable, placeholders, render


def test_whole_value_placeholder_keeps_object():
    v = {"audio": b"\x00\x01", "messages": [{"role": "user", "content": "hi"}], "session": 3}
    out = render({"a": "{audio}", "m": "{messages}", "s": "{session}"}, v)
    assert out == {"a": b"\x00\x01", "m": v["messages"], "s": 3}


def test_partial_placeholders_are_formatted_and_unknown_left_alone():
    out = render({"p": "Q: {text} ({session}) {unknown} {\"json\": 1}", "n": 5, "l": ["{text}", "x"]},
                 {"text": "hello", "session": 2})
    assert out == {"p": 'Q: hello (2) {unknown} {"json": 1}', "n": 5, "l": ["hello", "x"]}


def test_none_values():
    out = render({"a": "{text}", "b": "x{text}y"}, {"text": None})
    assert out == {"a": None, "b": "xy"}
    assert drop_none({"a": None, "b": {"c": None, "d": 1}, "l": [None, 2]}) == {"b": {"d": 1}, "l": [2]}


def test_placeholders_and_jsonable():
    assert placeholders({"a": ["{text}", {"b": "{audio_wav} {uuid}"}]}) == {"text", "audio_wav", "uuid"}
    assert jsonable({"a": b"\x00\x01\x02", "b": [b"x"]}) == {"a": "AAEC", "b": ["eA=="]}
