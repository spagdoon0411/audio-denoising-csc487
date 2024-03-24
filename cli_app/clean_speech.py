import tensorflow as tf
from unet_complex import OurUNet
from complex_unet_spec import model_spec
import numpy as np
import soundfile as sf
import sys
from librosa import load

unet_complex_builder = OurUNet()
unet = unet_complex_builder.build_model(modelspec=model_spec)

data_config = {
        "sample_rate" : 16000,
        "hop_length" : 256,
        "noise_level" : 0.1,
        "frame_length" : 256 * 4,
        "fft_length" : 256 * 4,
}

stft_config = {
    "hop_length" : data_config["hop_length"],
    "noise_level" : data_config["noise_level"],
    "frame_length" : data_config["frame_length"],
    "fft_length" : data_config["fft_length"],
    "window_func" : tf.signal.hann_window
}

unet.load_weights("cli_app/unet_complex/unet_complex_2").expect_partial()

SAMPLE_RATE = data_config["sample_rate"]

class STFTLayer(tf.keras.layers.Layer):
    def __init__(self, stft_config):
        super(STFTLayer, self).__init__()
        self.stft_config = stft_config

    def call(self, inputs, *args, **kwargs):
        # super.__call__(args, kwargs) # TODO: correct?

        output = tf.signal.stft(
            inputs,
            frame_length=self.stft_config["frame_length"],
            frame_step=self.stft_config["hop_length"],
            fft_length=self.stft_config["fft_length"],
            window_fn=self.stft_config["window_func"]
        )

        return output


class ISTFTLayer(tf.keras.layers.Layer):
    def __init__(self, stft_config):
        super(ISTFTLayer, self).__init__()
        self.stft_config = stft_config
    
    def call(self, inputs, *args, **kwargs):
        # super.__call__(args, kwargs) # TODO: correct?

        output = tf.signal.inverse_stft(
            inputs,
            frame_length=self.stft_config["frame_length"],
            frame_step=self.stft_config["hop_length"],
            fft_length=self.stft_config["fft_length"],
            window_fn=self.stft_config["window_func"]
        )

        return output

stft = STFTLayer(stft_config=stft_config)
istft = ISTFTLayer(stft_config=stft_config)

def save_numpy_as_wav(vec, path: str) -> None:
    numpy_vec = vec if isinstance(vec, np.ndarray) else vec.numpy()  # type: ignore
    sf.write(file=path, data=numpy_vec, samplerate=SAMPLE_RATE)

def load_into_numpy(path: str):
    audio_arr, _ = load(path=path, sr=SAMPLE_RATE)

    return audio_arr

def predict_on_using_complex(complex_spect, unet, path):
    real = tf.math.real(complex_spect)
    imag = tf.math.imag(complex_spect)
    dual_channel = tf.stack([real, imag], -1)
    mixed_spect_expanded = tf.expand_dims(dual_channel, axis=0)
    prediction_spect = unet.predict(mixed_spect_expanded)
    prediction_spect_trimmed = tf.squeeze(prediction_spect)
    prediction_spect_real = prediction_spect_trimmed[:, :, 0]
    prediction_spect_imag = prediction_spect_trimmed[:, :, 1]
    predicted_clean = istft(tf.dtypes.complex(prediction_spect_real, prediction_spect_imag))
    save_numpy_as_wav(predicted_clean, path)

def mixed_wav_to_clean_wav(mixed_path, requested_clean_path):
    mixed_vec = load_into_numpy(mixed_path)
    spect = stft(mixed_vec)
    predict_on_using_complex(spect, unet, requested_clean_path)

if __name__ == '__main__':
    mixed_path = sys.argv[1]
    clean_path = sys.argv[2]
    mixed_wav_to_clean_wav(mixed_path, clean_path)
