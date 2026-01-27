import argparse
import os
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


def copy_tokenizer_files(source: Path, output_dir: Path) -> None:
    """Copy tokenizer files verbatim.

    Re-serializing through ``save_pretrained`` can rewrite a ByteLevel BPE
    tokenizer as a SentencePiece one, which silently drops spaces and line
    breaks and makes grammar-constrained decoding impossible.
    """
    copied = []
    for name in TOKENIZER_FILES:
        candidate = source / name
        if candidate.exists():
            shutil.copy2(candidate, output_dir / name)
            copied.append(name)
    if "tokenizer.json" not in copied:
        raise FileNotFoundError(f"no tokenizer.json in {source}")
    print(f"Copied tokenizer files from {source}: {', '.join(copied)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--base_model", type=Path)
    parser.add_argument("--adapter_scale", type=float, default=1.0)
    parser.add_argument(
        "--torch_dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Compute and output dtype used while merging the adapter.",
    )
    parser.add_argument(
        "--ignore_saved_embeddings",
        action="store_true",
        help=(
            "Ignore embed_tokens and lm_head copies automatically stored in "
            "the adapter; use the frozen base-model tensors instead."
        ),
    )
    args = parser.parse_args()
    if args.adapter_scale < 0:
        raise ValueError("--adapter_scale must be non-negative")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = PeftConfig.from_pretrained(args.adapter, local_files_only=True)
    base_model = str(args.base_model or config.base_model_name_or_path)
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    tokenizer_source = (
        args.adapter
        if (args.adapter / "tokenizer_config.json").exists()
        else Path(base_model)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=True,
    )
    embedding_count = model.get_input_embeddings().num_embeddings
    if embedding_count < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    elif embedding_count > len(tokenizer):
        print(
            "Keeping padded base embeddings: "
            f"model={embedding_count}, tokenizer={len(tokenizer)}"
        )

    adapter_path = args.adapter
    temporary_adapter = None
    if args.ignore_saved_embeddings:
        adapter_weights = args.adapter / "adapter_model.safetensors"
        if not adapter_weights.exists():
            raise FileNotFoundError(adapter_weights)
        temporary_adapter = tempfile.TemporaryDirectory()
        adapter_path = Path(temporary_adapter.name)
        shutil.copy2(
            args.adapter / "adapter_config.json",
            adapter_path / "adapter_config.json",
        )
        state = load_file(adapter_weights)
        filtered_state = {
            name: tensor
            for name, tensor in state.items()
            if ".embed_tokens." not in name and ".lm_head." not in name
        }
        removed = sorted(set(state) - set(filtered_state))
        if not removed:
            raise ValueError(
                "No saved embedding or lm_head tensors found in adapter"
            )
        save_file(
            filtered_state,
            adapter_path / "adapter_model.safetensors",
        )
        print(f"Ignoring {len(removed)} frozen embedding tensors")

    peft_model = PeftModel.from_pretrained(
        model,
        adapter_path,
        local_files_only=True,
    )
    if temporary_adapter is not None:
        temporary_adapter.cleanup()
    if args.adapter_scale != 1.0:
        for module in peft_model.modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict):
                for adapter_name in scaling:
                    scaling[adapter_name] *= args.adapter_scale
    merged = peft_model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    copy_tokenizer_files(tokenizer_source, args.output_dir)
    print(
        f"Merged {args.adapter} into {args.output_dir} "
        f"with dtype={args.torch_dtype}"
    )


if __name__ == "__main__":
    main()
