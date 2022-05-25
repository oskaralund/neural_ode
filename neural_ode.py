from functools import partial

import numpy as np
import jax
from jax import vmap, jit, value_and_grad
from jax.random import PRNGKey, normal, split
from jax.experimental.ode import odeint
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm


def gaussian_sample(rng, mean, logvar):
    """ Samples from a Gaussian distribution. """
    std = jnp.exp(0.5*logvar)
    N = normal(rng, mean.shape)
    return mean + std*N


class RNN(nn.Module):
    hidden_dim: int = 25
    output_dim: int = 4

    def setup(self):
        # Layer mapping input to hidden state
        self.i2h = nn.Dense(self.hidden_dim)

        # Layer mapping hidden state to output
        self.h2o = nn.Dense(self.output_dim)

    def __call__(self, input_seq):
        # Initialize hidden state with zeros
        h = jnp.zeros(self.hidden_dim)

        for x in input_seq:
            h = self.i2h(jnp.concatenate([h, x]))
            h = jnp.tanh(h)

        return self.h2o(h)


class Encoder(nn.Module):

    @nn.compact
    def __call__(self, sequence):
        mean = RNN()(jnp.flip(sequence))
        logvar = RNN()(jnp.flip(sequence))
        return mean, logvar


class TimeDerivative(nn.Module):
    hidden_dim: int = 20
    output_dim: int = 4

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(self.hidden_dim)(z)
        z = jax.nn.elu(z)
        z = nn.Dense(self.hidden_dim)(z)
        z = jax.nn.elu(z)
        z = nn.Dense(self.output_dim)(z)
        return z


class Decoder(nn.Module):
    hidden_dim: int = 20
    output_dim: int = 2

    @nn.compact
    def __call__(self, z):
       z = nn.Dense(self.hidden_dim)(z)
       z = nn.relu(z)
       z = nn.Dense(self.output_dim)(z)
       return z


def spiral(rng):
    key1, key2 = split(rng)
    cx = normal(key1)
    cy = normal(key2)
    t = jnp.linspace(0, 1)
    theta = jnp.pi*8
    x = t*jnp.cos(t*theta) + cx
    y = t*jnp.sin(t*theta) + cy
    return jnp.column_stack([x, y])


def encode_decode(params, rng, sequence):
    t = jnp.linspace(0, 1)
    mean = RNN().apply(params["mean"], jnp.flip(sequence))
    logvar = RNN().apply(params["logvar"], jnp.flip(sequence))
    z_init = gaussian_sample(rng, mean, logvar)
    z = odeint(lambda z, _: TimeDerivative().apply(params["dzdt"], z), z_init, t)
    decoded = vmap(partial(Decoder().apply, params["decoder"]))(z)
    return decoded, mean, logvar


def kl_divergence(mean, logvar):
    var = jnp.exp(logvar)
    return 0.5*jnp.sum(var + mean**2 - 1 - logvar)


def loss(params, rng, sequence):
    decoded, mean, logvar = encode_decode(params, rng, sequence)
    kl_loss = 1e-4*kl_divergence(mean, logvar)
    rec_loss = jnp.sum((sequence-decoded)**2)

    return kl_loss + rec_loss, decoded, mean, logvar


def create_train_state(rng):
    input_dim = 2
    latent_dim = 4
    mean_params = RNN().init(rng, jnp.zeros((1,input_dim)))
    logvar_params = RNN().init(rng, jnp.zeros((1,input_dim)))
    dzdt_params = TimeDerivative().init(rng, jnp.zeros(latent_dim))
    decoder_params = Decoder().init(rng, jnp.zeros(latent_dim))
    params = {"mean": mean_params,
              "logvar": logvar_params,
              "dzdt": dzdt_params,
              "decoder": decoder_params}
    tx = optax.adam(1e-2)
    return TrainState.create(apply_fn=None, params=params, tx=tx)


def batch_loss(params, rng, batch):
    rngs = split(rng, len(batch))
    vloss = vmap(loss, in_axes=(None,0,0), out_axes=(0,0,0,0))
    l, decoded, means, logvars = vloss(params, rngs, batch)
    return jnp.mean(l), (decoded, means, logvars)


@jit
def compute_metrics(batch, decoded, means, logvars):
    kl_div = jnp.mean(vmap(kl_divergence, in_axes=(0, 0))(means, logvars))
    rec_loss = jnp.sum((decoded-batch)**2)
    metrics = {"kl_divergence": kl_div, "reconstruction_loss": rec_loss}
    return metrics


@jit
def train_step(state, batch, rng):
    grad_fn = value_and_grad(batch_loss, has_aux=True)
    (_, (decoded, means, logvars)), grads = grad_fn(state.params, rng, batch)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(batch, decoded, means, logvars)
    return state, metrics


def train_epoch(state, data, rng):
    pbar = tqdm(data)
    for batch in pbar:
        state, metrics = train_step(state, batch, rng)
        _, rng = split(rng)
        pbar.set_description(f"rec_loss: {metrics['reconstruction_loss']:.2f}"
                             f"kl_div: {metrics['kl_divergence']:.2f}")

    return state


rng = PRNGKey(0)
input_dim = 2
latent_dim = 4
mean_params = RNN().init(rng, jnp.zeros((1,input_dim)))
logvar_params = RNN().init(rng, jnp.zeros((1,input_dim)))
dzdt_params = TimeDerivative().init(rng, jnp.zeros(latent_dim))
decoder_params = Decoder().init(rng, jnp.zeros(latent_dim))
params = {"mean": mean_params,
          "logvar": logvar_params,
          "dzdt": dzdt_params,
          "decoder": decoder_params}

state = create_train_state(rng)


data = [vmap(spiral)(split(rng, 32)) for _ in range(1000)]

for i in range(10):
    state = train_epoch(state, data, rng)
