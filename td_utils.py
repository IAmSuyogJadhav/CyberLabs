import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment
import tensorflow as tf
# For reading audio files
from tensorflow.contrib.framework.python.ops import audio_ops


# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):  # TODO
    rate, data = get_wav_info(wav_file)
    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(
                                data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(
                                data[:, 0], nfft, fs, noverlap=noverlap)
    return pxx


# Load a wav file
def get_wav_info(wav_file):  # DONE

    # Old
    # rate, data = wavfile.read(wav_file)
    
    # New
    audio_binary = tf.read_file(wav_file)
    desired_channels=2
    wav_decoder = audio_ops.decode_wav(audio_binary, desired_channels=desired_channels)
    
    # wav_decoder does a normalization step. Multiplying by 2*15 undoes that
    data = tf.cast(wav_decoder.audio * 2**15, tf.int16)
    rate = wav_decoder.sample_rate
    return rate, data


# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):  #TODO
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# Load raw audio files for speech synthesis
def load_raw_audio():  #TODO
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./raw_data/activates/"+filename)
            activates.append(activate)
    for filename in os.listdir("./raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(
                                        "./raw_data/backgrounds/" + filename)
            backgrounds.append(background)
    for filename in os.listdir("./raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./raw_data/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds
