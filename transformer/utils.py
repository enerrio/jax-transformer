import equinox as eqx
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def save(filename: str, model: eqx.Module):
    """Save model to disk."""
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load(filename: str, model: eqx.Module):
    """Load saved model."""
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


def configure_pbar() -> Progress:
    """Setup rich progress bar for monitoring training."""
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        BarColumn(),
        MofNCompleteColumn(),
    )
    return progress


def plot_stats(train_stats: dict[str, list[float | int]], plot_name: str) -> None:
    """Plot training & validation loss."""
    _, ax = plt.subplots()
    ax.plot(train_stats["train_loss"], label="Train Loss")
    ax.plot(train_stats["val_loss"], linestyle="-.", label="Validation Loss")
    ax.set_title("Loss curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_top = ax.twiny()
    ax_top.plot(train_stats["tokens_seen"], train_stats["train_loss"], alpha=0)
    ax_top.set_xlabel("Tokens seen")
    plt.tight_layout()
    plt.savefig(plot_name)
