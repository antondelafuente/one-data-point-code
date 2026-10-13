#!/usr/bin/env python3
"""Train one row alone: 16 copies per update, LoRA r32 on attention/MLP, constant learning rate.

Save a sampler with a seven-day TTL at each selected checkpoint. The original interface remains supported:
  SEED=4 CKPTS=5,10,20 python train_row.py NAME PROMPT RESPONSE LR OUT.json
The file interface avoids interpolating training text in a shell command:
  python train_row.py NAME --row-file row.json --lr 1e-4 --seed 4 --ckpts 5,10,20 --out OUT.json
The row file must be a JSON object with string fields "prompt" and "response".
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
BATCH = 16
DEFAULT_CKPTS = "5,10,20,40,80,160"


def positive_lr(value):
    try:
        lr = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("learning rate must be a positive finite number") from None
    if not math.isfinite(lr) or lr <= 0:
        raise argparse.ArgumentTypeError("learning rate must be a positive finite number")
    return lr


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Name used in checkpoint names and metadata.")
    parser.add_argument("legacy", nargs="*", metavar="PROMPT RESPONSE LR OUT",
                        help="Original four arguments after NAME; use either these or --row-file/--lr/--out.")
    parser.add_argument("--row-file", type=Path, help="JSON object containing prompt and response.")
    parser.add_argument("--lr", type=positive_lr, help="Positive constant learning rate.")
    parser.add_argument("--out", type=Path, help="Checkpoint receipt JSON, updated atomically at every save.")
    parser.add_argument("--seed", help="Nonnegative training seed; defaults to the SEED environment variable.")
    parser.add_argument("--ckpts", "--checkpoints", dest="ckpts",
                        help=f"Comma-separated increasing positive update numbers; defaults to CKPTS or {DEFAULT_CKPTS}.")
    args = parser.parse_args(argv)
    if not args.name.strip():
        parser.error("name must be nonempty")
    if args.legacy:
        if len(args.legacy) != 4 or any(v is not None for v in (args.row_file, args.lr, args.out)):
            parser.error("supply NAME PROMPT RESPONSE LR OUT, or NAME --row-file FILE --lr LR --out OUT")
        args.prompt, args.response, value, output = args.legacy
        try:
            args.lr = positive_lr(value)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        if not output.strip():
            parser.error("output path must be nonempty")
        args.out = Path(output)
    else:
        if args.row_file is None or args.lr is None or args.out is None:
            parser.error("--row-file, --lr and --out are required without the original positional arguments")
        try:
            row = json.loads(args.row_file.read_text())
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read row file {args.row_file}: {exc}")
        if not isinstance(row, dict):
            parser.error("row file must contain a JSON object with prompt and response")
        args.prompt, args.response = row.get("prompt"), row.get("response")
    for field in ("prompt", "response"):
        text = getattr(args, field)
        if not isinstance(text, str) or not text.strip():
            parser.error(f"{field} must be a nonempty string")
    seed = args.seed if args.seed is not None else os.environ.get("SEED")
    try:
        args.seed = int(seed)
    except (TypeError, ValueError):
        parser.error("provide a nonnegative integer seed with --seed or SEED")
    if args.seed < 0:
        parser.error("seed must be a nonnegative integer")
    checkpoint_text = args.ckpts if args.ckpts is not None else os.environ.get("CKPTS", DEFAULT_CKPTS)
    try:
        args.ckpts = [int(value) for value in checkpoint_text.split(",")]
    except ValueError:
        parser.error("checkpoints must be a nonempty comma-separated list of positive integers")
    if not args.ckpts or any(step <= 0 for step in args.ckpts) or args.ckpts != sorted(set(args.ckpts)):
        parser.error("checkpoints must be positive, unique and in increasing order")
    # Reject unusable paths before constructing a training client.
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.is_dir():
            parser.error(f"output path is a directory: {args.out}")
        if args.row_file is not None and args.out.resolve() == args.row_file.resolve():
            parser.error("output receipt must not overwrite the training row file")
    except OSError as exc:
        parser.error(f"cannot prepare output directory: {exc}")
    return args


def write_receipt(out, result):
    """Replace a complete receipt on the same filesystem; preserve the previous file if writing fails."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=out.parent, prefix=f".{out.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train(args):
    # Import optional paid-service dependencies only after all CLI inputs have been checked.
    import tinker
    from tinker_cookbook.renderers import get_renderer, TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum

    sc = tinker.ServiceClient()
    client = sc.create_lora_training_client(
        base_model=BASE, rank=32, train_mlp=True, train_attn=True, train_unembed=False,
        seed=args.seed, user_metadata={"project": "explore-csp1-single-row-seeds-1", "seed": str(args.seed), "row": args.name})
    renderer = get_renderer(RENDERER, client.get_tokenizer())
    datum = conversation_to_datum(
        [{"role": "user", "content": args.prompt}, {"role": "assistant", "content": args.response}],
        renderer, max_length=4096, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    rest = sc.create_rest_client()
    result = {"name": args.name, "seed": args.seed, "lr": args.lr, "batch": BATCH,
              "prompt": args.prompt, "response": args.response, "base": BASE, "rank": 32,
              "renderer": RENDERER, "ckpts": {}, "loss": {}, "datum_tokens": datum.model_input.length}
    for step in range(1, max(args.ckpts) + 1):
        forward = client.forward_backward([datum] * BATCH, "cross_entropy")
        update = client.optim_step(tinker.AdamParams(learning_rate=args.lr, grad_clip_norm=1.0))
        metrics = getattr(forward.result(), "metrics", None) or {}
        update.result()
        loss = next((value for key, value in metrics.items() if "loss" in key.lower()), None)
        result["loss"][step] = loss
        if step in args.ckpts:
            path = client.save_weights_for_sampler(
                name=f"srs1_{args.name}_seed{args.seed}_lr{args.lr:g}_s{step}").result().path
            try:
                rest.set_checkpoint_ttl_from_tinker_path(path, 7 * 86400).result()
            except Exception as exc:
                print("ttl fail", exc)
            result["ckpts"][step] = path
            print(f"step {step} loss={loss} -> {path}", flush=True)
            write_receipt(args.out, result)
    print("=== DONE ===", flush=True)
    return result


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
