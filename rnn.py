from typing import Dict, Optional, Tuple
import jax
from jax import numpy as jnp
from jax import random, jit
import flax.linen as nn
from flax.linen.module import compact, nowrap
from flax.typing import Array, PRNGKey
from dataclasses import field

'''
Base audio RNN class used for all experiments

A. Wright, E.-P. Damskagg, L. Juvela, and V. Valimaki, “Real-time
guitar amplifier emulation with deep learning,” Appl. Sci., vol. 10, no. 2,
2020
https://www.mdpi.com/2076-3417/10/3/766
'''
class AudioRNN(nn.Module):
    hidden_size: int
    cell_type: type(nn.RNNCellBase)
    cell_args: Optional[Dict] = field(default_factory=dict)
    residual_connection: bool = True
    out_channels: int = 1
    dtype: type = jnp.float32

    def setup(self):
        self.rec = nn.RNN(self.cell_type(self.hidden_size, dtype=self.dtype, param_dtype=self.dtype, **self.cell_args))
        self.linear = nn.Dense(self.out_channels)

    @nn.compact
    def __call__(self, carry, x):
        new_carry, states = self.rec(x, initial_carry=carry, return_carry=True)
        out = self.linear(states)
        if self.residual_connection:
            out += x[..., 0:1]
        return new_carry, out

    def initialise_carry(self, input_shape):
        return self.cell_type(self.hidden_size, parent=None, dtype=self.dtype, param_dtype=self.dtype, **self.cell_args).initialize_carry(jax.random.key(0), input_shape)

'''
LSTM cell with FIR interpolation / extrapolation in feedback loop (proposed method)
'''
class FIRInterpLSTMCell(nn.LSTMCell):
    kernel: Array = None

    @compact
    def __call__(self, carry, inputs):
        c, h = carry
        c_interp = jnp.matmul(c, self.kernel)
        h_interp = jnp.matmul(h, self.kernel)
        (latest_c, latest_h), _ = super().__call__((c_interp, h_interp), inputs)
        new_c = c.at[..., -1].set(latest_c)
        new_h = h.at[..., -1].set(latest_h)
        new_c = jnp.roll(new_c, shift=1, axis=-1)
        new_h = jnp.roll(new_h, shift=1, axis=-1)
        return (new_c, new_h), latest_h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        num_coeffs = jnp.size(self.kernel)
        mem_shape = batch_dims + (self.features, num_coeffs)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)

'''
LSTM cell with state trajectory network (STN) method of sample rate adjustment:

J. Parker, F. Esqueda, and A. Bergner, “Modelling of nonlinear
state-space systems using a deep neural network,” in Proc. 22nd Int.
Conf. Digital Audio Effects, Birmingham, UK, Sept. 2019

https://www.dafx.de/paper-archive/2019/DAFx2019_paper_42.pdf

More info:
https://arxiv.org/abs/2406.06293

'''
class STNLSTMCell(nn.LSTMCell):
    kernel: Array = None

    @compact
    def __call__(self, carry, inputs):
        c, h = carry
        (temp_c, temp_h), _ = super().__call__((c[..., 0], h[..., 0]), inputs)
        c_concat = jnp.concatenate((jnp.expand_dims(temp_c, -1), c), -1)
        h_concat = jnp.concatenate((jnp.expand_dims(temp_h, -1), h), -1)
        k = self.kernel / jnp.sum(self.kernel)
        latest_c = jnp.dot(c_concat, k)
        latest_h = jnp.dot(h_concat, k)
        new_c = c.at[..., -1].set(latest_c)
        new_h = h.at[..., -1].set(latest_h)
        new_c = jnp.roll(new_c, shift=1, axis=-1)
        new_h = jnp.roll(new_h, shift=1, axis=-1)
        return (new_c, new_h), latest_h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        num_coeffs = jnp.size(self.kernel) - 1
        mem_shape = batch_dims + (self.features, num_coeffs)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)

'''
LSTM cell which concatenates hidden and cell states together as the output
'''
class LSTMCellOneState(nn.LSTMCell):
    @compact
    def __call__(self, carry, inputs):
        new_carry, _ = super().__call__(carry, inputs)
        return new_carry, jnp.concatenate(new_carry, axis=-1)

