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
    # audio formats. Some being 32 bit PCM or at a different sampling frequency
    # Therefore, we use tf.read_fil and tf.contrib.ffmpeg to explicitly read
    # audio files in 44.1 KHz. This takes care of these problems.
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
            pass  # Output of tf.contrib.ffmpeg.decode_audio is already normalized
        else:
            data *= 2**15

    else:  # If being used in the app, in which case we will take care
        # to ensure proper audio, thus read directly.
        _, data = get_wav_info(wav_file, normalize)

    # Convert to single dimensional vector by taking max of both channels.
    # Works better than just dropping a channel.
    data = tf.reduce_max(data, axis=1)
    data = data[None, ...]  # To make the output shape comply with the model

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
def get_wav_info(wav_file, normalize=True):
    audio_binary = tf.read_file(wav_file)  # The raw audio data
    desired_channels = 2  # Always
    wav_decoder = audio_ops.decode_wav(audio_binary,
                                       desired_channels=desired_channels)

    if normalize:
        data = wav_decoder.audio
    else:
        # decode_wav does a normalization step. Multiplying by 2^15 undoes that
        data = wav_decoder.audio * 2**15

    # Get the sampling frequency, useful for debugging later on.
    rate = wav_decoder.sample_rate

    return rate, data
