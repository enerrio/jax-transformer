import time
from typing import Any
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree, Scalar, jaxtyped
from rich import print as rprint
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
def generate_polynomial_data(
    key: PRNGKeyArray,
    coefficients: Int[Array, "d_coeff"] = jnp.array([1, -2, 3]),
    n_samples: int = 100,
) -> tuple[Float[Array, "batch_size 1"], Float[Array, "batch_size 1"], PRNGKeyArray]:
    """Generate polynomial data."""
    key, subkey = random.split(key, 2)
    X = jnp.linspace(-10, 10, n_samples)
    y = jnp.polyval(coefficients, X) + random.normal(subkey, shape=(n_samples,))
    return X.reshape(-1, 1), y.reshape(-1, 1), key


class Model(eqx.Module):
    layers: list[Any]

    def __init__(self, in_dim: int, out_dim: int, key: PRNGKeyArray) -> None:
        key1, key2 = random.split(key, 2)
        self.layers = [
            eqx.nn.Linear(
                in_dim, out_dim, use_bias=True, key=key1
            ),  # * (5/3) / jnp.sqrt(1.)
            jax.nn.tanh,
            eqx.nn.Linear(out_dim, 1, use_bias=True, key=key2),
        ]

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "1"]) -> Float[Array, "1"]:
        for layer in self.layers:
            x = layer(x)
        return x


@jaxtyped(typechecker=typechecker)
def mse(model: Model, x: Float[Array, "batch_size 1"], y: Float[Array, "batch_size 1"]) -> Scalar:
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)


@jaxtyped(typechecker=typechecker)
def train(
    model: Model,
    optim: optax.GradientTransformation,
    x: Float[Array, "batch_size 1"],
    y: Float[Array, "batch_size 1"],
    num_epochs: int,
    log_rate: int,
) -> tuple[Model, list[Scalar]]:
    """Train the model."""
    losses = []
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def train_step(
        model: Model,
        x: Float[Array, "batch_size 1"],
        y: Float[Array, "batch_size 1"],
        opt_state: PyTree,
    ) -> tuple[Model, Scalar]:
        """Single training step."""
        loss, grad = eqx.filter_value_and_grad(mse)(model, x, y)
        # simple way
        # model = jax.tree.map(lambda p, g: p - 0.01 * g if g is not None else p, model, grad)
        # using equinox
        # updates = jax.tree.map(lambda g: -0.01 * g, grad)
        # model = eqx.apply_updates(model, updates)
        # using optax
        updates, opt_state = optim.update(
            grad, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, loss

    for i in range(num_epochs):
        model, loss = train_step(model, x, y, opt_state)
        if (i % log_rate) == 0:
            rprint(f"Epoch [{i}/{num_epochs}] | Train Loss: {loss:.3f}")
        losses.append(loss)
    return model, losses


def main() -> None:
    key = random.key(21)
    X, y, key = generate_polynomial_data(
        key, coefficients=jnp.array([1, -2, 1, -2]), n_samples=1_000
    )
    # standardize data
    X_norm = (X - X.mean()) / X.std()
    y_norm = (y - y.mean()) / y.std()

    rprint(f"X shape: {X.shape} - X dtype: {X.dtype}")
    rprint(f"y shape: {y.shape} - y dtype: {y.dtype}")

    model = Model(1, 128, key)
    # pretty print model structure
    model_str = eqx.tree_pformat(model)
    rprint(model_str)

    num_epochs = 1000
    log_rate = 100
    lr = 0.01
    optim = optax.sgd(learning_rate=lr)

    start_time = time.time()
    model, losses = train(model, optim, X_norm, y_norm, num_epochs, log_rate)
    end_time = time.time()
    rprint(f"Total train time: {end_time-start_time:.2f} seconds")

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()

    # Plot model's predictions
    out = jax.vmap(model)(X_norm)
    out = (out * y.std()) + y.mean()

    # Plot ground truth (y) vs input (X)
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label="Ground Truth", color="blue", s=10)

    # Plot predictions vs input (X)
    plt.scatter(X, out, label="Predictions", color="green", s=10)

    # Add vertical error bars (red dotted lines) between predictions and ground truth
    for i in range(len(X)):
        plt.plot([X[i], X[i]], [y[i, 0], out[i, 0]], "r--", linewidth=0.5)

    plt.title("Ground Truth vs Untrained Predictions with Error Bars")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
