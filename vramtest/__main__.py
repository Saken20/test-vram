"""CLI: python -m vramtest [--config PATH] [--run NAME] [--quick] <command>.

Scenario / report commands call (imported lazily, so ``validate`` and ``probe`` work on their own):
  scenarios.idle(ctx)
  scenarios.burst(ctx, target=..., levels=..., rounds=...)
  scenarios.concurrency(ctx, target=..., users=..., duration=...)
  scenarios.sessions(ctx, profile=..., duration=...)
  scenarios.ramp(ctx, target=..., levels=..., step=...)
  report.build(cfg, run_dir) -> str ; pdf_report.build(cfg, run_dir) -> str
where ``target`` is a service name or "pipeline" (see Context.steps).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from . import config


def _levels(text):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    def add_common(parser, defaults):
        # subcommands default to SUPPRESS so they do not overwrite flags given before the subcommand
        d = (lambda v: v) if defaults else (lambda v: argparse.SUPPRESS)
        parser.add_argument("--config", default=d(None), help=f"YAML config (default: ./{config.DEFAULT_PATH})")
        parser.add_argument("--run", default=d("default"), help="run name: results go to results/<run>/")
        parser.add_argument("--quick", action="store_true", default=d(False),
                            help="short durations, to check that everything works")

    common = argparse.ArgumentParser(add_help=False)
    add_common(common, defaults=False)
    p = argparse.ArgumentParser(prog="vramtest",
                                description="Measure real VRAM and latency of GPU inference services under load")
    add_common(p, defaults=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_parser = sub.add_parser
    sub.add_parser = lambda *a, **kw: _add_parser(*a, parents=[common], **kw)

    sub.add_parser("validate", help="parse the config, print services and pipeline")
    sub.add_parser("probe", help="one call per service; GPU processes, vLLM metrics, KV bytes/token")
    g = sub.add_parser("gen-audio", help="synthesize dataset texts to WAV clips with a speech-capable service")
    g.add_argument("--service", help="speech service (default: the first one found)")
    sub.add_parser("idle", help="VRAM with no requests")
    b = sub.add_parser("burst", help="N turns started at the same instant (N = levels)")
    b.add_argument("--levels", type=_levels, help="e.g. 1,10,100")
    b.add_argument("--rounds", type=int, help="bursts per level")
    b.add_argument("--service", help="a service name or 'pipeline' (default: pipeline if defined, else every service)")
    c = sub.add_parser("concurrency", help="N workers sending turns back-to-back")
    c.add_argument("--users", type=int)
    c.add_argument("--duration", type=float)
    c.add_argument("--service", help="a service name or 'pipeline' (default: pipeline if defined, else every service)")
    s = sub.add_parser("sessions", help="K concurrent sessions with think time (profiles from the config)")
    s.add_argument("--profile", default="all", help="profile name from scenarios.sessions.profiles, or 'all'")
    s.add_argument("--duration", type=float)
    r = sub.add_parser("ramp", help="increase concurrency until a service saturates")
    r.add_argument("--service", default="all", help="a service name, 'pipeline', or 'all' (every service)")
    r.add_argument("--levels", type=_levels, help="e.g. 1,2,5,10,20")
    r.add_argument("--step", type=float, help="seconds per level")
    sub.add_parser("report", help="results/<run>/report.md + report.json")
    sub.add_parser("pdf", help="results/<run>/summary.pdf")
    a = sub.add_parser("all", help="idle, burst, concurrency, sessions, ramp, report, pdf")
    a.add_argument("--skip-ramp", action="store_true", help="skip ramp (it may crash an overloaded service)")
    return p.parse_args(argv)


def load_config(path):
    """Load the config or exit with a readable error list."""
    try:
        cfg = config.load(path)
    except config.ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        sys.exit(2)
    for w in cfg["_warnings"]:
        print(f"[config] warning: {w}", file=sys.stderr)
    return cfg


def validate(cfg) -> int:
    """Print what the config defines; check protos/methods and the dataset. Returns an exit code."""
    problems = []
    print(f"config: {cfg['_path']}")
    print("services:")
    for name, s in cfg["services"].items():
        where = s.get("url") or s.get("address")
        detail = s["kind"] if s["adapter"] == "openai" else str(s.get("method") or "")
        print(f"  {name:<14} {s['adapter']:<7} {detail:<28} {where}  engine={s['engine']}"
              + (f" model={s['model']}" if s.get("model") else ""))
        if s["adapter"] == "grpc":
            try:
                from . import protos
                m = protos.load(config.resolve(s["proto"]), [config.resolve(d) for d in s["include_dirs"]]
                                ).method(s["method"])
                print(f"  {'':<14} {m.path} ({m.shape})")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"services.{name}: {exc}")
    if cfg["pipeline"]:
        print("pipeline: " + " -> ".join(f"{st['service']}({st['input']})" for st in cfg["pipeline"]))
    else:
        print("pipeline: none (scenarios load each service separately or use --service)")
    try:
        from .dataset import Dataset
        ds = Dataset(cfg)
        print(f"dataset: {len(ds.texts)} prompts, {len(ds.clips)} audio clips in {ds.audio_dir}"
              + (", system prompt set" if ds.system_prompt else ""))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"dataset: {exc}")
    for p in problems:
        print(f"[error] {p}", file=sys.stderr)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def check_service(args, cfg) -> None:
    """Exit with a readable message when ``--service`` names nothing the command can load."""
    name = getattr(args, "service", None)
    if not name:
        return
    allowed = list(cfg["services"])
    if args.cmd != "gen-audio":
        allowed += ["pipeline"] if cfg["pipeline"] else []
        allowed += ["all"] if args.cmd == "ramp" else []
    if name not in allowed:
        sys.exit(f"unknown --service {name!r} (expected one of: {', '.join(allowed)})")


async def main_async(args, cfg, run_dir):
    if args.cmd == "report":
        from . import report
        print(report.build(cfg, run_dir))
        return
    if args.cmd == "pdf":
        from . import pdf_report, report
        report.build(cfg, run_dir)
        print(pdf_report.build(cfg, run_dir))
        return

    from .context import Context, probe
    ctx = Context(cfg, run_dir)
    if args.cmd == "probe":
        await probe(ctx)
        return
    if args.cmd == "gen-audio":
        made = await ctx.gen_audio(args.service)
        print(f"[gen-audio] {made} clip(s) in {ctx.dataset.audio_dir}")
        return

    from . import scenarios
    if args.cmd == "idle":
        await scenarios.idle(ctx)
    elif args.cmd == "burst":
        for target in [args.service] if args.service else ctx.default_targets():
            await scenarios.burst(ctx, target=target, levels=args.levels, rounds=args.rounds)
    elif args.cmd == "concurrency":
        for target in [args.service] if args.service else ctx.default_targets():
            await scenarios.concurrency(ctx, target=target, users=args.users, duration=args.duration)
    elif args.cmd == "sessions":
        profiles = cfg["scenarios"]["sessions"]["profiles"]
        names = list(profiles) if args.profile == "all" else [args.profile]
        for name in names:
            if name not in profiles:
                sys.exit(f"unknown profile {name!r} (defined: {', '.join(profiles)})")
            await scenarios.sessions(ctx, profile=name, duration=args.duration)
    elif args.cmd == "ramp":
        targets = list(cfg["services"]) if args.service == "all" else [args.service]
        for target in targets:
            await scenarios.ramp(ctx, target=target, levels=args.levels, step=args.step)
    elif args.cmd == "all":
        from . import pdf_report, report
        await scenarios.idle(ctx)
        for target in ctx.default_targets():
            await scenarios.burst(ctx, target=target)
        for target in ctx.default_targets():
            await scenarios.concurrency(ctx, target=target)
        for name in cfg["scenarios"]["sessions"]["profiles"]:
            await scenarios.sessions(ctx, profile=name)
        if not args.skip_ramp:
            for target in cfg["services"]:
                await scenarios.ramp(ctx, target=target)
        print(report.build(cfg, run_dir))
        print(pdf_report.build(cfg, run_dir))
        return
    print(f"\nReport: python -m vramtest --run {args.run} report")


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.quick:
        config.apply_quick(cfg)
    if args.cmd == "validate":
        sys.exit(validate(cfg))
    check_service(args, cfg)
    run_dir = os.path.join(os.getcwd(), "results", args.run)
    asyncio.run(main_async(args, cfg, run_dir))


if __name__ == "__main__":
    main()
