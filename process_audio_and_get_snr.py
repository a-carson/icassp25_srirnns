import os.path

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

import utils
from utils import model_info_from_json, get_fir_interp_kernel, audio_LSTM_params_from_state_dict
import matplotlib as mpl
from scipy import signal
mpl.use('macosx')
from rnn import FIRInterpLSTMCell, AudioRNN
import flax.linen as nn
from scipy.signal import resample

# input settings
model_filename = '../jax_rnn/Proteus_Tone_Packs/flattened/BlackstarHT40_AmpHighGain.json'
clip_no = 4                         # input clip no (0 to 7)
cond_const = 0.5                    # conditioning constant for models with a 'knob' parameter
trunc_first_n_samples = 2000        # for computing snr
src_ratio = 44.1/48                 # sample rate conversion ratio

save_audio = False
interpolation_methods = ['lagrange', 'minimax']
orders = np.arange(1, 6)

# load input
sr_base, in_sig = scipy.io.wavfile.read(f'docs/audio/clip-{clip_no}.wav')
sr = sr_base * src_ratio

# load model
model_info, state_dict = model_info_from_json(model_filename)
params = audio_LSTM_params_from_state_dict(state_dict)
print('Model: ', model_info['name'])

# convert in_sig type to float
dtype = in_sig.dtype
in_sig = np.float32(in_sig / np.iinfo(dtype).max)
in_sig = np.expand_dims(in_sig, (0, 2))

label = 'oversampled' if src_ratio > 1 else 'undersampled'
savename_template = os.path.join('./', model_info['name'] + f'_{label}_clip-{clip_no}' + '_{}.wav')

if model_info['input_size'] == 2:
    cond_sig = cond_const * np.ones_like(in_sig)
    in_sig = np.concatenate((in_sig, cond_sig), -1)

# original RNN model output
params = audio_LSTM_params_from_state_dict(state_dict)
base_model = AudioRNN(cell_type=nn.LSTMCell, hidden_size=model_info['hidden_size'])
last_carry, out_base = base_model.apply({'params': params}, base_model.initialise_carry((1, 1)), in_sig)
out_base = np.ravel(out_base)
if save_audio:
    scipy.io.wavfile.write(savename_template.format('target'), sr_base, np.ravel(out_base))


# resample input and target to new rate
resampled_input_length = int(in_sig.shape[1] * src_ratio)
in_sig_resampled = scipy.signal.resample(in_sig, resampled_input_length, axis=1)
out_target = resample(out_base, resampled_input_length)

# naive method
_, out_naive = base_model.apply({'params': params}, base_model.initialise_carry((1, 1)), in_sig_resampled)
out_naive = np.ravel(out_naive)
snr = utils.snr_dB(sig=out_target[trunc_first_n_samples:],
                   noise=out_target[trunc_first_n_samples:] - out_naive[trunc_first_n_samples:])
if save_audio:
    savename = savename_template.format('naive')
    out_sig = resample(out_naive, out_base.shape[0])
    scipy.io.wavfile.write(savename, sr_base, out_sig)
    print(savename)

print('Naive method')
print('SNR = ', np.round(snr, 2))

for interpolation_method in interpolation_methods:
    print('Interpolation method:', interpolation_method)
    for order in orders:
        kernel = get_fir_interp_kernel(order=order, delta=src_ratio-1, method=interpolation_method)
        srirnn_model = AudioRNN(cell_type=FIRInterpLSTMCell, hidden_size=model_info['hidden_size'], cell_args={'kernel': kernel})
        _, out_srirnn = srirnn_model.apply({'params': params}, srirnn_model.initialise_carry((1, 1)), in_sig_resampled)
        out_srirnn = np.ravel(out_srirnn)

        snr = utils.snr_dB(sig=out_target[trunc_first_n_samples:],
                           noise=out_target[trunc_first_n_samples:] - out_srirnn[trunc_first_n_samples:])

        print(f'Order: {order}, SNR = {np.round(snr, 1)}dB')

        if save_audio:
            savename = savename_template.format(f'{interpolation_method}{order}')
            out_sig = resample(out_srirnn, out_base.shape[0])
            scipy.io.wavfile.write(savename, sr_base, out_sig)
            print(savename)

