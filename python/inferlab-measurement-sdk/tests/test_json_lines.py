from pathlib import Path

from inferlab_measurement_sdk import read_json_lines


def test_rows_split_at_newline_only_and_keep_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"text":"a\u2028b\u2029c\u0085d"}\n\n{"n":2}\n', encoding="utf-8")

    rows, error = read_json_lines(path, "fixture rows")

    assert error is None
    assert rows == [(1, {"text": "a\u2028b\u2029c\u0085d"}), (3, {"n": 2})]


def test_reading_stops_at_the_first_invalid_row_and_keeps_the_rows_before_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"n":1}\n{"n":\n{"n":3}\nnot-json\n', encoding="utf-8")

    rows, error = read_json_lines(path, "fixture rows")

    assert rows == [(1, {"n": 1})]
    assert error is not None
    assert error.startswith("fixture rows line 2 is invalid JSON: ")
    assert "line 4" not in error


def test_an_unreadable_artifact_is_named(tmp_path: Path) -> None:
    rows, error = read_json_lines(tmp_path / "missing.jsonl", "fixture rows")

    assert rows == []
    assert error is not None
    assert error.startswith("fixture rows is unreadable: ")
    assert "missing.jsonl" in error
