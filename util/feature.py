import numpy as np
import torch
import resampy
import soundfile
from torchaudio.compliance.kaldi import fbank


target_sample_rate = 16000
target_dB = -20


def resample(samples, sample_rate):
    return resampy.resample(samples, sample_rate, target_sample_rate, filter='kaiser_best')


def convert_samples(samples, dtype='int16'):
    dtype = np.dtype(dtype)
    output_samples = samples.copy()
    if dtype in np.sctypes['int']:
        bits = np.iinfo(dtype).bits
        output_samples *= (2 ** (bits - 1) / 1.)
        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max
        output_samples[output_samples > max_val] = max_val
        output_samples[output_samples < min_val] = min_val
    elif samples.dtype in np.sctypes['float']:
        min_val = np.finfo(dtype).min
        max_val = np.finfo(dtype).max
        output_samples[output_samples > max_val] = max_val
        output_samples[output_samples < min_val] = min_val
    else:
        raise TypeError("Unsupported sample type: %s." % samples.dtype)
    return output_samples.astype(dtype)


def read_wav(file_path):
    samples, sample_rate = soundfile.read(file_path, dtype='float32')
    return samples, sample_rate


def get_feature(file_path):
    samples, sample_rate = read_wav(file_path)
    samples = resample(samples, sample_rate)
    samples = convert_samples(samples)
    waveform = torch.from_numpy(np.expand_dims(samples, 0)).float()
    mat = fbank(waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=1.0,
                sample_frequency=sample_rate)
    fbank_feat = mat.numpy()
    return fbank_feat
