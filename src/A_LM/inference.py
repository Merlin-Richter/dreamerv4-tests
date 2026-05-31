import argparse
from pathlib import Path

import torch

from model import ModelConfig, Transformer


@torch.inference_mode()
def generate(
    model: Transformer,
    stoi: dict[str, int],
    itos: dict[int, str],
    device: str,
    prompt: str,
    max_new_tokens: int,
    seq_len: int,
    temperature: float,
) -> str:
    unknown = [c for c in prompt if c not in stoi]
    if unknown:
        raise ValueError(f"Prompt contains characters not in vocabulary: {set(unknown)!r}")

    ids = [stoi[c] for c in prompt]
    for _ in range(max_new_tokens):
        context = ids[-seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)
        next_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
    return "".join(itos[i] for i in ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint.pt from training.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="ROMEO:\n",
        help="Starting text (must use only characters seen in training data).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=">1.0 = more random; <1.0 = sharper / closer to greedy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write generated text to this file instead of stdout.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ckpt = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    chars: list[str] = ckpt["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    config = ModelConfig(**ckpt["config"])
    seq_len = config.max_sequence_length

    model = Transformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    text = generate(
        model,
        stoi,
        itos,
        device,
        args.prompt,
        args.max_new_tokens,
        seq_len,
        args.temperature,
    )

    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(text)} characters to {args.output}")
    else:
        print(text, end="" if text.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
