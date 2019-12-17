#!/usr/bin/env python
#coding: utf-8

import numpy as np
import scipy.fftpack as fftpack
from scipy.ndimage import gaussian_filter1d


# compilation of useful functions for computing spectrograms from:
# https://courses.engr.illinois.edu/ece590sip/sp2018/spectrograms1_wideband_narrowband.html


def enframe(x, S, L):
   # w = 0.54*np.ones(L)
    #for n in range(0,L):
     #   w[n] = w[n] - 0.46*math.cos(2*math.pi*n/(L-1))
    w = np.hamming(L)
    frames = []
    nframes = 1+int((len(x)-L)/S)
    for t in range(0, nframes):
        frames.append(np.copy(x[(t*S):(t*S+L)])*w)
    return frames


def stft(frames, n, fs):
    stft_frames = [fftpack.fft(x, n) for x in frames]
    freq_axis = np.linspace(0, fs, n)
    return stft_frames, freq_axis


def stft2level(stft_spectra, max_freq_bin):
    magnitude_spectra = [abs(x) for x in stft_spectra]
    max_magnitude = max([max(x) for x in magnitude_spectra])
    min_magnitude = max_magnitude / 1000.0
    for t in range(0,len(magnitude_spectra)):
        for k in range(0,len(magnitude_spectra[t])):
            magnitude_spectra[t][k] /= min_magnitude
            if magnitude_spectra[t][k] < 1:
                magnitude_spectra[t][k] = 1
    # multiply log by 20 for SPL spectra
    level_spectra = [np.log10(x[0:max_freq_bin]) for x in magnitude_spectra]
    return level_spectra


def sgram(x, frame_skip, frame_length, fft_length, fs, max_freq):
    frames = enframe(x, frame_skip, frame_length)
    (spectra, freq_axis) = stft(frames, fft_length, fs)
    sgram = stft2level(spectra, int(max_freq*fft_length/fs))
    max_time = len(frames)*frame_skip/fs
    return sgram, max_time, max_freq
