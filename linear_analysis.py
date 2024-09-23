import os.path

import numpy as np
import matplotlib.pyplot as plt
import utils
from utils import model_info_from_json, get_fir_interp_kernel, audio_LSTM_params_from_state_dict, get_LSTM_jacobian
from rnn import AudioRNN, FIRInterpLSTMCell
from scipy import signal
#mpl.use('macosx')
from argparse import ArgumentParser

def polar_plot(a, ax, **kwargs):
    ax.scatter(np.real(a), np.imag(a), **kwargs)
    circle = plt.Circle((0, 0), 1.0, linestyle='--', color='k', fill=False, linewidth=1)
    ax.add_patch(circle)
    ax.set_aspect('equal', adjustable='box')
    ax_lim = 2.0
    ax.set_xlim([-ax_lim, ax_lim])
    ax.set_ylim([-ax_lim, ax_lim])
    ax.set_xticks([])
    ax.set_yticks([])

parser = ArgumentParser(description='Apply SRIRNN methods to a single LSTM model and view SNR results')
parser.add_argument('-f', '--model_filename', type=str, default='Proteus_Tone_Packs/AmpPack1/BlackstarHT40_AmpHighGain.json',
                    help='Path to the model file')
parser.add_argument('-m', '--method', type=str, default='lagrange',
                    help='interpolation method (lagrange or minimax)')
parser.add_argument('--src_ratio', type=float, default=44.1 / 48,
                    help='Sample rate conversion ratio')
parser.add_argument('--cond_const', type=float, default=0.5,
                    help="Conditioning constant for models with a 'knob' parameter")
args = parser.parse_args()

# other settings ------
plot_poles = True
plot_spec = True
sr = 44100
dur = 0.25



# load LSTM and get jacbobian around fixed point ------
model_info, state_dict = model_info_from_json(args.model_filename)
params = audio_LSTM_params_from_state_dict(state_dict)
J = get_LSTM_jacobian(params['rec'])

# input signal --------
in_sig = np.zeros((1, int(dur * sr), 1))
if model_info['input_size'] == 2:
    in_sig = np.concatenate((in_sig, args.cond_const * np.ones_like(in_sig)), -1)


# set-up plotting ----------
orders = np.arange(0, 6)
if plot_poles:
    poles_fig, poles_ax = plt.subplots(2, 3)
if plot_spec:
    spec_fig, spec_ax = plt.subplots(2, 3, figsize=[20, 6])

print('Model: ', model_info['name'])
print('Interpolation:', args.method)
for order in orders:
    kernel = get_fir_interp_kernel(order=order, delta=args.src_ratio-1, method=args.method)

    # state matrix of companion 1-step form
    A = np.concatenate((np.kron(kernel, J),
                        np.concatenate((np.eye(len(J) * order), np.zeros((len(J) * order, len(J)))), axis=1)),
                       axis=0)

    # get eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(A)
    is_stable = np.max(np.abs(eigenvalues)) < 1.0
    print(f'Order: {order}, Max pole radius = {np.max(np.abs(eigenvalues[:len(A)]))}, Stable = {is_stable}')

    # plotting
    ax_idx = int(np.floor(order // 3)), order % 3
    title = f'order = {order}'
    if plot_poles:
        poles_ax[ax_idx].set_title(title)
        polar_plot(eigenvalues, poles_ax[ax_idx], **{'marker': 'x', 'color': 'b', 's': 5})
    if plot_spec:
        #
        # Process original model with zero input to get steady-state response then interpolation filters in feedback loop
        #
        model = AudioRNN(cell_type=FIRInterpLSTMCell, hidden_size=model_info['hidden_size'], cell_args={'kernel': signal.unit_impulse(kernel.shape)})
        init_carry = model.initialise_carry((1, 1))
        last_carry, out_base = model.apply({'params': params}, init_carry, in_sig)
        model_up = AudioRNN(cell_type=FIRInterpLSTMCell, hidden_size=model_info['hidden_size'], cell_args={'kernel': kernel})
        _, out_srirnn = model_up.apply({'params': params}, last_carry, in_sig)
        out = np.ravel(np.concatenate((out_base, out_srirnn), axis=1))

        f, t, Sxx = signal.spectrogram(out, sr, mode='magnitude')
        spec_ax[ax_idx].pcolormesh(t, f, 20 * np.log10(Sxx + 1e-9), shading='gouraud')
        spec_ax[ax_idx].set_ylabel('Frequency [Hz]')
        spec_ax[ax_idx].set_xlabel('Time [sec]')
        spec_ax[ax_idx].set_xlim([np.min(t), np.max(t)])
        spec_ax[ax_idx].set_title(title)


device_name = os.path.split(args.model_filename)[-1][:-5]
if plot_poles:
    poles_fig.subplots_adjust(hspace=0.3, wspace=0.0)
    poles_fig.suptitle(device_name)
    poles_fig.tight_layout()

if plot_spec:
    spec_fig.subplots_adjust(hspace=0.3, wspace=0.3)
    spec_fig.suptitle(device_name)
    spec_fig.tight_layout()

plt.show()

