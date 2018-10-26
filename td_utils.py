import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment
import tensorflow as tf

# For reading audio files
from tensorflow.contrib.framework.python.ops import audio_ops


# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(  # DONE
    wav_file, old=False,
    window_length=200,  # Length of each window segment
    sampling_frequency=8000,  # Sampling frequency
    step_size=200,  # Step size
    noverlap=200,  # Overlap between windows, ignored if old=False
    fft_length=2,
    normalize=False
        ):  # TODO
    """
    Parameters
    ----------
    wav_file: str, required
        Path to the wav file
    sampling_frequency: int, optional
        The frequency to in Hz to resample the audio to. Defaults to 8000.
    window_length: int, optional
        The length of the window for stft. Defaults to 200.
        HIGHER the window length, LOWER the no. of features produced in axis 0.
    step_size: int, optional
        The traversal step for stft. Defaults to 200.
        HIGHER the step size, LOWER the no. of features produced in axis 0.
    fft_length: int, optional
        Used while applying stft. Defaults to 2.
        HIGHER the step size, HIGHER the no. of features produced in axis 1.
    normalize: boolean, optional
        If True, normalize the output audio (by dividing by 2**15).
        Defaults to False.
    """

    # Old
    if old:
        rate, data = get_wav_info(wav_file, old=True)
        nchannels = data.ndim
        print("Data Shape:", data.shape)
        if nchannels == 1:
            pxx, freqs, bins, im = plt.specgram(
                                    data, window_length,
                                    sampling_frequency,
                                    noverlap=noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = plt.specgram(
                                    data[:, 0], window_length,
                                    sampling_frequency,
                                    noverlap=noverlap)

        return pxx

    # New
    else:
        # Load the raw audio data
        audio_binary = tf.read_file(wav_file)  # The raw data

        # Decode and convert into a Tensor
        data = tf.contrib.ffmpeg.decode_audio(
            audio_binary,
            file_format='wav',
            samples_per_second=sampling_frequency,
            channel_count=2
            ) * 2**15  # To undo normalization

        # Convert to single dimensional vector by taking max of both channels.
        # Works better than just dropping a channel.
        tf.reduce_max(data, axis=1, keepdims=True)

        # Compute spectrogram for the signal by converting it to frequency
        # domain by applying Short-time Fourier Transform.
        # Returns a complex64 tensor.
        specgrams = tf.contrib.signal.stft(
            data,
            frame_length=window_length,
            frame_step=step_size,
            fft_length=fft_length
            )

        # There are two ways we acn utilize the spectrogram, power spectrogram
        # and magnitude spectrogram. We will use power spectrogram, given by
        # taking the modulus of the spectrogram tensor and squaring it.
        pxx = tf.real(specgrams * tf.conj(specgrams))
        return pxx


# Load a wav file
def get_wav_info(wav_file, old=False):  # DONE

    # Old
    if old:
        rate, data = wavfile.read(wav_file)
        return rate, data

    # # New
    else:
        audio_binary = tf.read_file(wav_file)
        desired_channels = 2  # Always
        wav_decoder = audio_ops.decode_wav(audio_binary,
                                           desired_channels=desired_channels)
        # decode_wav does a normalization step. Multiplying by 2^15 undoes that
        data = tf.cast(wav_decoder.audio * 2**15, tf.int16)
        rate = wav_decoder.sample_rate

        return rate, data


# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):  # TODO
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# Load raw audio files for speech synthesis
def load_raw_audio():  # TODO
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
