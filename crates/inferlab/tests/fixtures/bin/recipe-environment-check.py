from pathlib import Path


def main() -> None:
    print("fixture preflight ran")
    exit_code = int(Path(__file__).with_name("fixture-check-exit-code").read_text(encoding="utf-8"))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
