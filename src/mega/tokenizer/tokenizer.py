"""SentencePiece tokenizer with a Hugging-Face-friendly surface.

:class:`MegaTokenizer` wraps a SentencePiece model and exposes ``encode`` /
``decode`` / ``save_pretrained`` / ``from_pretrained`` so a trained tokenizer can
be pushed to and pulled from the Hub.  Special tokens are fixed and shared with
:mod:`mega.tokenizer.train_tokenizer`.

``encode`` returns plain Python lists so the data pipeline stays independent of
any tokenizer-framework return-type convention.
"""

from __future__ import annotations

import json
from pathlib import Path

# Fixed special-token ids -- baked into the SentencePiece model by
# train_tokenizer and shared by everything downstream.  They must not drift,
# or checkpoints stop decoding.
UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3
SPECIAL_TOKENS = {
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
}
# Modality placeholders reserved in the vocabulary.
MODALITY_TOKENS = ["<image>", "<audio>", "<video>"]

__all__ = ["MegaTokenizer", "UNK_ID", "BOS_ID", "EOS_ID", "PAD_ID"]


class MegaTokenizer:
    def __init__(self, model_path: str | Path):
        import sentencepiece as spm

        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor()
        if not self.sp.load(str(self.model_path)):
            raise ValueError(f"failed to load SentencePiece model: {self.model_path}")
        self.vocab_size = int(self.sp.get_piece_size())

    # ------------------------------------------------------------------- ids
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids = [BOS_ID, *ids]
        if add_eos:
            ids = [*ids, EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([int(i) for i in ids])

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        return self.encode(text, add_bos=add_bos, add_eos=add_eos)

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    @property
    def bos_token_id(self) -> int:
        return BOS_ID

    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    # ------------------------------------------------------- hub round-trip
    def save_pretrained(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dst = directory / "tokenizer.model"
        dst.write_bytes(self.model_path.read_bytes())
        (directory / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "MegaTokenizer",
                    **SPECIAL_TOKENS,
                    "modality_tokens": MODALITY_TOKENS,
                    "model_type": "unigram_or_bpe",
                    "vocab_size": self.vocab_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # keep the raw vocab next to the model when it is available
        vocab = self.model_path.with_suffix(".vocab")
        if vocab.exists():
            (directory / "tokenizer.vocab").write_bytes(vocab.read_bytes())
        return directory

    @classmethod
    def from_pretrained(cls, source: str | Path) -> MegaTokenizer:
        """Load from a local directory or a Hub repo id."""
        p = Path(source)
        if (p / "tokenizer.model").exists():
            return cls(p / "tokenizer.model")
        if p.exists() and p.is_file():
            return cls(p)
        # fall back to a Hub download
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=str(source), filename="tokenizer.model")
        return cls(path)
