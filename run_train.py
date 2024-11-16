import time
import argparse
import jax
import jax.random as jr
import equinox as eqx
import optax
from rich import print as rprint
from config import GPT_CONFIG
from transformer.model import GPTModel
from transformer.train import train
from transformer.utils import save, plot_stats
from transformer.data import load_data


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=["small", "medium", "large", "xlarge"],
    )
    parser.add_argument(
        "--data", type=str, required=True, choices=["the-verdict", "war-and-peace"]
    )
    parser.add_argument("--nb_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--plot_name", type=str, default="train_loss.png", required=True
    )
    args = parser.parse_args(args)
    model_config = GPT_CONFIG[args.model_size]
    data_path = f"{args.data}.txt"
    train_dataloader, val_dataloader = load_data(
        data_path, model_config, args.batch_size
    )
    rprint(f"Batch size: {args.batch_size}")
    rprint(f"Number of batches in train dataloader: {len(train_dataloader)}")
    rprint(f"Number of batches in val dataloader: {len(val_dataloader)}")

    key = jr.key(21)
    model_key, train_key = jr.split(key)
    model = GPTModel(model_config, model_key)
    optim = optax.adamw(learning_rate=0.0004, weight_decay=0.1)
    leaves, _ = jax.tree.flatten(model)
    num_params = sum([leaf.size for leaf in leaves if eqx.is_array(leaf)])
    rprint(f"Total number of model parameters ({args.model_size}): {num_params:,}")
    # model_str = eqx.tree_pformat(model)
    # rprint(model_str)

    # Test out what initial loss should look like
    # initial_loss = -jnp.log(1.0 / model_config["vocab_size"])
    # rprint(f"Initial loss should be around: {initial_loss:.3f}")
    # key, *sample_keys = jr.split(train_key, train_dataloader.batch_size + 1)
    # sample_keys = jnp.array(sample_keys)
    # x_sample, y_sample = next(iter(train_dataloader))
    # logits = jax.vmap(model, in_axes=(0, None, 0))(x_sample, False, sample_keys)
    # loss = optax.losses.softmax_cross_entropy_with_integer_labels(
    #     logits, y_sample
    # ).mean()
    # rprint(f"Actual initial loss is: {loss:.3f}")
    # sys.exit(0)

    rprint("Training...")
    start = time.time()
    model, train_stats = train(
        model=model,
        optim=optim,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        key=train_key,
        num_epochs=args.nb_epochs,
    )
    rprint(f"Total training time: {time.time()-start:.2f} seconds.")
    rprint("Complete!")
    save(f"gpt2-{args.model_size}-{args.experiment_name}.eqx", model)
    rprint("Model saved!")

    plot_stats(train_stats, args.plot_name)


if __name__ == "__main__":
    main()
