from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from . import config


# Carga comun: misma frecuencia, misma duracion y amplitud normalizada.
def cargar_audio(path: str | Path, sr: int = config.SR, n_samples: int = config.N_SAMPLES) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr)
    max_abs = np.abs(y).max()
    if max_abs > 0:
        y = y / max_abs

    if len(y) > n_samples:
        start = (len(y) - n_samples) // 2
        y = y[start : start + n_samples]
    else:
        y = np.pad(y, (0, n_samples - len(y)), mode="constant")
    return y.astype(np.float32)


def augmentation_audio(y: np.ndarray) -> np.ndarray:
    audio = y.copy()
    if np.random.rand() < 0.5:
        audio = audio + np.random.normal(0, 0.005, size=audio.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        shift = int(np.random.uniform(-0.12, 0.12) * config.SR)
        audio = np.roll(audio, shift)
    if np.random.rand() < 0.35:
        gain = np.random.uniform(0.85, 1.15)
        audio = audio * gain
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def mel_spectrogram(y: np.ndarray) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.SR,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)


def _resample_vector(y: np.ndarray, n_samples: int) -> np.ndarray:
    if len(y) == n_samples:
        return y.astype(np.float32)
    old_x = np.linspace(0.0, 1.0, num=len(y), endpoint=True)
    new_x = np.linspace(0.0, 1.0, num=n_samples, endpoint=True)
    return np.interp(new_x, old_x, y).astype(np.float32)


# Version compacta para no entrenar con matrices gigantes.
def wavelet_scalogram(y: np.ndarray) -> torch.Tensor:
    import pywt

    y_small = _resample_vector(y, config.WAVELET_INPUT_SAMPLES)
    scales = np.arange(1, config.N_WAVELET_SCALES + 1)
    coeffs, _ = pywt.cwt(y_small, scales, config.WAVELET_NAME)
    magnitude = np.abs(coeffs)
    scalogram_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    tensor = torch.tensor(scalogram_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(config.N_WAVELET_SCALES, config.WAVELET_TIME_BINS),
        mode="bilinear",
        align_corners=False,
    )
    return tensor.squeeze(0)


def normalize_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_value = tensor.min()
    max_value = tensor.max()
    if max_value > min_value:
        return (tensor - min_value) / (max_value - min_value)
    return tensor - min_value