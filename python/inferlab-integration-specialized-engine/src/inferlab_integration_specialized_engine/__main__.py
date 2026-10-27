from inferlab_adapter_sdk import run_adapter

from . import plan_serve, probe_serve, render_serve, stack_settings


def main() -> None:
    raise SystemExit(
        run_adapter(
            plan_serve,
            render_serve=render_serve,
            probe_serve=probe_serve,
            stack_settings=stack_settings,
        )
    )


if __name__ == "__main__":
    main()
