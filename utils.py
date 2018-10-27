import tensorflow as tf
# For reading audio files
from tensorflow.contrib.framework.python.ops import audio_ops


# Calculate and plot spectrogram for the wav audio file
def graph_spectrogram(
        wav_file,
        window_length=1650,  # Length of each window segment
        step_size=65,  # Step size
        fft_length=200,
        normalize=False,
        train=True
        ):
    """
    Parameters
    ----------
    wav_file: str, required
        Path to the wav file
    window_length: int, optional
        The length of the window for stft. Defaults to 1650.
        HIGHER the window length, LOWER the no. of features produced in axis 0.
    step_size: int, optional
        The traversal step for stft. Defaults to 65.
        HIGHER the step size, LOWER the no. of features produced in axis 0.
    fft_length: int, optional
        Used while applying dicrete FFT to each of the windows. Defaults to 200.
        HIGHER the step size, HIGHER the no. of features produced in axis 1.
    normalize: boolean, optional
        If True, normalize the output audio (by dividing by 2**15).
        Defaults to False.
    """
    # Load the raw audio data
    # Due to the mixed nature of Urban 8K dataset,
    # The dataset is somewhat untidy in terms of sampling frequency and
    # audio formats. some being 32 bit PCM.
    # Therefore, we use tf.read_file tf.contrib.ffmpeg to explicitly read the
    # audio files
    if train:  # If being used during training
        audio_binary = tf.read_file(wav_file)  # The raw audio data
        # Decode and convert into a Tensor
        data = tf.contrib.ffmpeg.decode_audio(
            audio_binary,
            file_format='wav',
            samples_per_second=44100,  # Fixed
            channel_count=2
            )
        if normalize:
            pass
        else:
            data *= 2**15

    else:
        _, data = get_wav_info(wav_file, normalize)

    # Convert to single dimensional vector by taking max of both channels.
    # Works better than just dropping a channel.
    data = tf.reduce_max(data, axis=1)
    data = data[None, ...]

    # Compute spectrogram for the signal by converting it to frequency
    # domain by applying Short-time Fourier Transform.
    # Returns a complex64 tensor.
    specgrams = tf.contrib.signal.stft(
        data,
        frame_length=window_length,
        frame_step=step_size,
        fft_length=fft_length
        )

    # There are two ways we can utilize the spectrogram, power spectrogram
    # and magnitude spectrogram. We will use power spectrogram, given by
    # taking the modulus of the spectrogram tensor and squaring it.
    pxx = tf.real(specgrams * tf.conj(specgrams))
    return pxx


# Load a wav file
def get_wav_info(wav_file, normalize=True):  # DONE
    # New
    audio_binary = tf.read_file(wav_file)
    desired_channels = 2  # Always
    wav_decoder = audio_ops.decode_wav(audio_binary,
                                       desired_channels=desired_channels)
    # decode_wav does a normalization step. Multiplying by 2^15 undoes that
    # data = tf.cast(wav_decoder.audio * 2**15, tf.float16)
    if normalize:
        data = wav_decoder.audio
    else:
        data = wav_decoder.audio * 2**15
    rate = wav_decoder.sample_rate

    return rate, data


# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):  # TODO
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# Load raw audio files for speech synthesis
# def load_raw_audio():  # TODO
#     activates = []
#     backgrounds = []
#     negatives = []
#     for filename in os.listdir("./raw_data/activates"):
#         if filename.endswith("wav"):
#             activate = AudioSegment.from_wav("./raw_data/activates/"+filename)
#             activates.append(activate)
#     for filename in os.listdir("./raw_data/backgrounds"):
#         if filename.endswith("wav"):
#             background = AudioSegment.from_wav(
#                                         "./raw_data/backgrounds/" + filename)
#             backgrounds.append(background)
#     for filename in os.listdir("./raw_data/negatives"):
#         if filename.endswith("wav"):
#             negative = AudioSegment.from_wav("./raw_data/negatives/"+filename)
#             negatives.append(negative)
#     return activates, negatives, backgrounds
