import numpy as np
import pandas as pd
import flax.linen as nn
from rnn import LSTMCellOneState

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, Dict
import json
import os

def model_info_from_json(filename: str) -> Tuple[Dict, Dict]:
    '''
    Get info and state dict from Proteus Tone Library model
    :param filename: .json file
    :return: hyper_data, state_dict
    '''
    with open(filename, 'r') as f:
        json_data = json.load(f)

    hyper_data = json_data["model_data"]
    path, filename = os.path.split(filename)
    hyper_data["name"] = filename.split('.json')[0]
    state_dict = json_data["state_dict"]
    return hyper_data, state_dict

def audio_LSTM_params_from_state_dict(state_dict: Dict) -> Dict:
    '''
    Converts Proteus Tone Library .json files to flax parameter Dict
    :param state_dict: Proteus format parameters
    :return: flax format parameters
    '''

    for key, value in state_dict.items():
        state_dict[key] = jnp.asarray(value)

    W_hi, W_hf, W_hg, W_ho = jnp.split(state_dict['rec.weight_hh_l0'], 4)
    W_ii, W_if, W_ig, W_io = jnp.split(state_dict['rec.weight_ih_l0'], 4)
    bi, bf, bg, bo = jnp.split(state_dict['rec.bias_hh_l0'] + state_dict['rec.bias_ih_l0'], 4)

    lstm_params = {
        'hi': {'kernel': W_hi.transpose(), 'bias': bi.transpose()},
        'hf': {'kernel': W_hf.transpose(), 'bias': bf.transpose()},
        'hg': {'kernel': W_hg.transpose(), 'bias': bg.transpose()},
        'ho': {'kernel': W_ho.transpose(), 'bias': bo.transpose()},
        'ii': {'kernel': W_ii.transpose()},
        'if': {'kernel': W_if.transpose()},
        'ig': {'kernel': W_ig.transpose()},
        'io': {'kernel': W_io.transpose()}
    }

    linear_params = {'kernel': state_dict['lin.weight'].transpose(),
                     'bias': state_dict['lin.bias']}

    return {'rec': {'cell': lstm_params},
            'linear': linear_params}

def lagrange_interp_kernel(order: int, delta: float, pad: int = 0) -> jax.Array:
    '''
    Fractional delay filter coefficients using Lagrange design method

    :param order: order of interpolation
    :param delta: fractional delay
    :param pad: zero pad
    :return: filter coefficients
    '''
    kernel = jnp.ones(order + 1)
    for n in range(order + 1):
        for k in range(order + 1):
            if k != n:
                kernel = kernel.at[n].multiply((delta - k) / (n - k))
    if pad > 0:
        kernel = jnp.pad(kernel, (0, pad))
    return kernel

def l_inf_optimal_kernel(order: int, delta: float, bw: float = 0.5) -> jax.Array:
    '''
    Fractional delay filter coefficients from LUT (minimax method)

    :param order: order of interpolation (0 to 10)
    :param delta: fractional delay  (48/44.1 - 1 OR 44.1/48 - 1)
    :param bw: bandwidth used for optimisation (0.5 OR 0.9)
    :return: filter coefficients
    '''
    delta = np.round(delta, 3)
    df = pd.read_csv(f'lookup_tables/L_inf_delta={delta}_bw={bw}.csv', header=None)
    return df.values[order-1, :order+1]


def get_fir_interp_kernel(order: int, delta: float, method: str):
    '''
    Helper function to get fractional delay filter coefficients using Lagrange or Minmax
    :param order: order of interpolation
    :param delta: fractional delay
    :param method: 'lagrange' or 'minimax' or 'naive'
    :return: filter coefficients
    '''
    if order == 0:
        return np.ones(1)

    if method == 'lagrange':
        return lagrange_interp_kernel(order, delta)
    elif method == 'minimax':
        return l_inf_optimal_kernel(order, delta)
    elif method == 'naive':
        return np.ones(1)
    else:
        print('Invalid interpolation method')
        return


def get_LSTM_fixed_point(lstm_params: Dict, cond_const: float = 0.5, rand: bool = False, method: str = 'empircal') -> jax.Array:
    '''
    Finds the fixed point of an LSTM network

    :param lstm_params: Dict containing lstm params
    :param cond_const: conditioning parameter (constant) for models with conditioning input (2 channels)
    :param rand: initialise states with random numbers (True) or zeros (False)
    :param method: find fixed point empirically (method='empirical') as described in paper or analytically using newton-raphson (method='newton-raphson')
    :return: the fixed point as a jax.numpy Array
    '''
    hidden_size = lstm_params['cell']['hi']['bias'].shape[0]
    input_size = lstm_params['cell']['ii']['kernel'].shape[0]
    rnn = nn.RNN(LSTMCellOneState(hidden_size))

    if rand:
        init_carry = 0.1 * (np.random.random((1, hidden_size)) - 0.5)
    else:
        init_carry = jnp.zeros((1, hidden_size))

    in_sig = jnp.zeros((1, 10000, 1))
    if input_size == 2:
        in_sig = jnp.concatenate((in_sig, cond_const * np.ones_like(in_sig)), axis=-1)

    if method == 'newton-raphson':

        @jit
        def forward_fn(current_state_vec):
            h, c = jnp.split(current_state_vec, 2, axis=-1)
            new_carry, _ = rnn.apply({'params': lstm_params},
                                     in_sig[:, 0:1, :],
                                     initial_carry=(h, c), return_carry=True)
            return jnp.concatenate(new_carry, axis=-1)

        x = jnp.concatenate((init_carry, init_carry), axis=-1)
        for i in range(1000):

            res = (x - forward_fn(x)).squeeze()
            jacobian = jnp.eye(2 * hidden_size) - jax.jacobian(forward_fn)(x).squeeze()
            step = jnp.linalg.solve(jacobian, res)

            damper = 1.0
            for sub_i in range(20):
                x_i = x - damper * step
                res_i = (x_i - forward_fn(x_i)).squeeze()
                if np.linalg.norm(res_i) > np.linalg.norm(res):
                    damper *= 0.5
                else:
                    x = x_i
                    break

            if np.linalg.norm(res) < 1e-9:
                break

        print(f'Final residiual = {np.linalg.norm(res)} after {i} iterations')
        fixed_point = x

    else:
        _, states = rnn.apply({'params': lstm_params},
                                   in_sig,
                                   initial_carry=(init_carry, init_carry),
                                   return_carry=True)

        fixed_point = np.mean(states[:, -1000:, :], axis=1)

    return fixed_point


def get_LSTM_jacobian(lstm_params: Dict, cond_const: float = 0.5, method: str = 'empircal') -> jax.Array:
    '''
    Get the Jacobian of an LSTM network linearised about a fixed point.

    :param lstm_params: Dict containing lstm params
    :param cond_const: conditioning parameter (constant) for models with conditioning input (2 channels)
    :param method: method for finding the fixed point: find fixed point empirically (method='empirical') as described in paper or analytically using newton-raphson (method='newton-raphson')
    :return:
    '''
    hidden_size = lstm_params['cell']['hi']['bias'].shape[0]
    input_size = lstm_params['cell']['ii']['kernel'].shape[0]
    rnn = nn.RNN(nn.LSTMCell(hidden_size))

    in_sig = jnp.zeros((1, 1, 1))
    if input_size == 2:
        in_sig = jnp.concatenate((in_sig, cond_const * np.ones_like(in_sig)), axis=-1)

    fixed_point = get_LSTM_fixed_point(lstm_params, cond_const=cond_const, rand=False, method=method)
    @jit
    def forward_fn(current_state_vec):
        h, c = jnp.split(current_state_vec, 2, axis=-1)
        new_carry, _ = rnn.apply({'params': lstm_params},
                                 in_sig,
                                 initial_carry=(h, c), return_carry=True)
        return jnp.concatenate(new_carry, axis=-1)

    jacobian = jax.jacobian(forward_fn)(fixed_point).squeeze()

    return jacobian


def snr_dB(sig, noise):
    snr = np.sum(sig**2) / np.sum(noise**2)
    return 10 * np.log10(snr)