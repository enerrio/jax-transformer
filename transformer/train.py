import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Int, Float, Key, PyTree, Scalar
from rich import print as rprint
from torch.utils.data import DataLoader
from transformer.utils import configure_pbar


def train(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    key: Key[Array, ""],
    num_epochs: int,
) -> tuple[eqx.Module, dict[str, list[float | Float[Array, ""]]]]:
    """Train the model."""
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(
        model: eqx.Module,
        opt_state: PyTree,
        x: Int[Array, "batch seq_len"],
        y: Int[Array, "batch seq_len"],
        keys: Key[Array, " batch"],
    ) -> tuple[eqx.Module, PyTree, Scalar]:
        """Single training step for a batch of data. Forward pass, compute loss/grads, update weights."""

        def loss_fn(
            model: eqx.Module,
            x: Int[Array, "batch seq_len"],
            y: Int[Array, "batch seq_len"],
            keys: Key[Array, " batch"],
        ) -> Scalar:
            """Forward pass of model and compute loss."""
            logits = jax.vmap(model, in_axes=(0, None, 0))(x, False, keys)
            loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, y)
            return loss.mean()

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, keys)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def validate_step(
        inference_model: eqx.Module,
        x: Int[Array, "batch seq_len"],
        y: Int[Array, "batch seq_len"],
    ) -> Scalar:
        def validation_loss_fn(
            model: eqx.Module,
            x: Int[Array, "batch seq_len"],
            y: Int[Array, "batch seq_len"],
        ) -> Scalar:
            logits = jax.vmap(model, in_axes=(0, None, None))(x, True, None)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
            return loss.mean()

        loss = validation_loss_fn(inference_model, x, y)
        return loss

    train_stats = {"train_loss": [], "val_loss": [], "tokens_seen": []}
    progress = configure_pbar()
    with progress:
        epoch_task = progress.add_task("[red]Training model...", total=num_epochs)
        train_dataloader_task = progress.add_task(
            "[cyan]Iterating through dataloader...", total=len(train_dataloader)
        )
        val_dataloader_task = progress.add_task(
            "[magenta1]Validation phase...", total=len(val_dataloader), visible=False
        )
        for i in range(num_epochs):
            train_epoch_loss = val_epoch_loss = tokens_seen = 0.0
            progress.update(train_dataloader_task, visible=True)
            # train phase
            for x_batch, y_batch in train_dataloader:
                key, *subkeys = jr.split(key, train_dataloader.batch_size + 1)
                subkeys = jnp.array(subkeys)
                model, opt_state, loss = train_step(
                    model, opt_state, x_batch, y_batch, subkeys
                )
                train_epoch_loss += loss
                tokens_seen += x_batch.size
                progress.update(train_dataloader_task, advance=1)
            progress.reset(train_dataloader_task, visible=False)
            progress.update(epoch_task, advance=1)
            progress.update(val_dataloader_task, visible=True)
            # validation phase
            inference_model = eqx.nn.inference_mode(model)
            for x_val, y_val in val_dataloader:
                val_loss = validate_step(inference_model, x_val, y_val)
                val_epoch_loss += val_loss
                progress.update(val_dataloader_task, advance=1)
            progress.reset(val_dataloader_task, visible=False)

            # Average and store loss
            train_epoch_loss /= len(train_dataloader)
            train_stats["train_loss"].append(train_epoch_loss)
            val_epoch_loss /= len(val_dataloader)
            train_stats["val_loss"].append(val_epoch_loss)
            train_stats["tokens_seen"].append(tokens_seen)
            rprint(
                f"Epoch [{i+1}/{num_epochs}] | Train Loss: {train_epoch_loss:.3f} | Val Loss: {val_epoch_loss:.3f}"
            )
    return model, train_stats
