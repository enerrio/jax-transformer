import jax.numpy as jnp
import tiktoken
from jaxtyping import Array, Int
from torch.utils.data import DataLoader, Dataset
from rich import print as rprint


class GPTDatasetV1(Dataset):
    def __init__(
        self, text: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int
    ) -> None:
        super().__init__()
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        # Create dataset
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + 1 + max_length]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.input_ids[index], self.target_ids[index]

    def __len__(self) -> int:
        return len(self.input_ids)


def collate_fn(
    batch: list[tuple[list[int], list[int]]]
) -> tuple[Int[Array, "batch seq_len"], Int[Array, "batch seq_len"]]:
    """Convert tensors to Jax arrays."""
    input_batch, target_batch = zip(*batch)
    input_array = jnp.array(input_batch)
    target_array = jnp.array(target_batch)
    return input_array, target_array


def create_dataloader(
    text: str,
    tokenizer: tiktoken.Encoding,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Tokenize raw text data and create dataloader."""
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader


def load_data(
    data_path: str,
    config: dict[str, int | float | bool],
    batch_size: int,
    train_ratio: float = 0.9,
) -> tuple[DataLoader, DataLoader]:
    """Load data, tokenize, and create dataloaders."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(raw_text))
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]
    rprint(f"Length of raw text file: {len(raw_text):,}")
    rprint(f"Total number of tokens: {total_tokens:,}")
    rprint(f"Total number of train characters: {len(train_data):,}")
    rprint(f"Total number of val characters: {len(val_data):,}")

    train_dataloader = create_dataloader(
        train_data,
        tokenizer,
        batch_size=batch_size,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = create_dataloader(
        val_data,
        tokenizer,
        batch_size=batch_size,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    return train_dataloader, val_dataloader
