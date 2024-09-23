import os.path
import numpy as np
import scipy.io.wavfile
import utils
from utils import model_info_from_json, get_fir_interp_kernel, audio_LSTM_params_from_state_dict
from scipy import signal
from rnn import FIRInterpLSTMCell, AudioRNN
import flax.linen as nn
from scipy.signal import resample
from argparse import ArgumentParser


parser = ArgumentParser(description='Apply SRIRNN methods to a single LSTM model and view SNR results')
parser.add_argument('-f', '--model_filename', type=str, default='Proteus_Tone_Packs/AmpPack1/BlackstarHT40_AmpHighGain.json',
                    help='Path to the model file')
parser.add_argument('--src_ratio', type=float, default=44.1 / 48,
                    help='Sample rate conversion ratio')
parser.add_argument('--clip_no', type=int, default=4,
                    help='Input clip number (0 to 7)')
parser.add_argument('--cond_const', type=float, default=0.5,
                    help="Conditioning constant for models with a 'knob' parameter")
parser.add_argument('--trunc_first_n_samples', type=int, default=2000,
                    help='Number of samples to truncate for computing SNR')
parser.add_argument('--save_audio', action='store_true',
                        help='Save example audio clips (bool)')
args = parser.parse_args()
trunc = args.trunc_first_n_samples
src_ratio = args.src_ratio

# additional arguments
interpolation_methods = ['lagrange', 'minimax']
orders = np.arange(1, 6)

# load input
sr_base, in_sig = scipy.io.wavfile.read(f'docs/audio/clip-{args.clip_no}.wav')
sr = sr_base * args.src_ratio

# load model
model_info, state_dict = model_info_from_json(args.model_filename)
params = audio_LSTM_params_from_state_dict(state_dict)
print('Model: ', model_info['name'])

# convert in_sig type to float
dtype = in_sig.dtype
in_sig = np.float32(in_sig / np.iinfo(dtype).max)
in_sig = np.expand_dims(in_sig, (0, 2))

label = 'oversampled' if src_ratio > 1 else 'undersampled'
savename_template = os.path.join('./', model_info['name'] + f'_{label}_clip-{args.clip_no}' + '_{}.wav')

if model_info['input_size'] == 2:
    cond_sig = args.cond_const * np.ones_like(in_sig)
    in_sig = np.concatenate((in_sig, cond_sig), -1)

# original RNN model output
params = audio_LSTM_params_from_state_dict(state_dict)
base_model = AudioRNN(cell_type=nn.LSTMCell, hidden_size=model_info['hidden_size'])
last_carry, out_base = base_model.apply({'params': params}, base_model.initialise_carry((1, 1)), in_sig)
out_base = np.ravel(out_base)
if args.save_audio:
    scipy.io.wavfile.write(savename_template.format('target'), sr_base, np.ravel(out_base))


# resample input and target to new rate
resampled_input_length = int(in_sig.shape[1] * src_ratio)
in_sig_resampled = scipy.signal.resample(in_sig, resampled_input_length, axis=1)
out_target = resample(out_base, resampled_input_length)

# naive method
_, out_naive = base_model.apply({'params': params}, base_model.initialise_carry((1, 1)), in_sig_resampled)
out_naive = np.ravel(out_naive)
snr = utils.snr_dB(sig=out_target[trunc:],
                   noise=out_target[trunc:] - out_naive[trunc:])
if args.save_audio:
    savename = savename_template.format('naive')
    out_sig = resample(out_naive, out_base.shape[0])
    scipy.io.wavfile.write(savename, sr_base, out_sig)
    print(savename)

print('naive method:')
print('SNR = {:.2f} dB'.format(snr))

for interpolation_method in interpolation_methods:
    print(interpolation_method, 'interpolation:', )
    for order in orders:
        kernel = get_fir_interp_kernel(order=order, delta=src_ratio-1, method=interpolation_method)
        srirnn_model = AudioRNN(cell_type=FIRInterpLSTMCell, hidden_size=model_info['hidden_size'], cell_args={'kernel': kernel})
        _, out_srirnn = srirnn_model.apply({'params': params}, srirnn_model.initialise_carry((1, 1)), in_sig_resampled)
        out_srirnn = np.ravel(out_srirnn)

        snr = utils.snr_dB(sig=out_target[trunc:],
                           noise=out_target[trunc:] - out_srirnn[trunc:])

        print('Order: {}, SNR = {:.1f}dB'.format(order, snr))

        if args.save_audio:
            savename = savename_template.format(f'{interpolation_method}{order}')
            out_sig = resample(out_srirnn, out_base.shape[0])
            scipy.io.wavfile.write(savename, sr_base, out_sig)
            print(savename)

