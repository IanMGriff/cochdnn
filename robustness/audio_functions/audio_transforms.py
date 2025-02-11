import torch
import torchaudio
import random
import numpy as np
import sys
import chcochleagram
from chcochleagram import compression
from chcochleagram import cochleagram
from chcochleagram import *

from torch import nn
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT
import numpy as np

import random
import logging


def ch_demean(x, dim=0):
    """
    Helper function to mean-subtract tensor.

    Args
    ----
    x (tensor): tensor to be mean-subtracted
    dim (int): kwarg for torch.mean (dim along which to compute mean)

    Returns
    -------
    x_demean (tensor): mean-subtracted tensor
    """
    x_demean = torch.sub(x, torch.mean(x, dim=dim))
    return x_demean


def ch_rms(x, dim=0):
    """
    Helper function to compute RMS amplitude of a tensor.

    Args
    ----
    x (tensor): tensor for which RMS amplitude should be computed
    dim (int): kwarg for torch.mean (dim along which to compute mean)

    Returns
    -------
    rms_x (tensor): root-mean-square amplitude of x
    """
    rms_x = torch.sqrt(torch.mean(torch.pow(x, 2), dim=dim))
    return rms_x


class AudioCompose(torch.nn.Module):
    """
    Composes several foreground/background audio transforms together (based off of
        torchvision.transforms.Compose)

    Args:
        transforms (list of audio_function transfrom torch.nn.Modules): list of transforms to compose.

    """

    def __init__(self, transforms):
        super(AudioCompose, self).__init__()
        self.transforms = transforms

    def forward(self, foreground_wav, background_wav):
        for t in self.transforms:
            foreground_wav, background_wav = t(foreground_wav, background_wav)
        return foreground_wav, background_wav

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class LogScaleFakeClipping(torch.nn.Module):
    """
    Scales the values by a log scale. (Useful to apply aftr the Mel Spectrogram)
    """

    def __init__(self, offset=1e-6):
        super(LogScaleFakeClipping, self).__init__()
        self.offset = offset
        self.clamp_function = FakeClamp.apply

    def forward(self, foreground_wav, background_wav):
        foreground_wav = self.clamp_function(foreground_wav, self.offset)
        foreground_wav = torch.log2(foreground_wav)
        if background_wav is not None:
            background_wav = self.clamp_function(background_wav, self.offset)
            background_wav = torch.log2(background_wav)
        return foreground_wav, background_wav


class FakeClamp(torch.autograd.Function):
    """
    Applies clamp in the forward pass, but all gradients=1 in the backwards
    pass.
    """

    @staticmethod
    def forward(ctx, x, min):
        return torch.clamp(x, min=min)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class LogScale(torch.nn.Module):
    """
    Scales the values by a log scale. (Useful to apply aftr the Mel Spectrogram)
    """

    def __init__(self, offset=1e-6):
        super(LogScale, self).__init__()
        self.offset = offset

    def forward(self, foreground_wav, background_wav):
        foreground_wav = torch.clamp(foreground_wav, min=self.offset)
        foreground_wav = torch.log2(foreground_wav)
        if background_wav is not None:
            background_wav = torch.clamp(background_wav, min=self.offset)
            background_wav = torch.log2(background_wav)
        return foreground_wav, background_wav


class ClippedGradPower(torch.nn.Module):
    """
    Wrapper around ClippedGradPowerCompression defined in chcochleagram.compression
    """

    def __init__(self, compression_kwargs):
        super(ClippedGradPower, self).__init__()
        self.compression_kwargs = compression_kwargs
        self.compression_function = compression.ClippedGradPowerCompression(
            **compression_kwargs
        )

    def forward(self, foreground_wav, background_wav):
        foreground_wav = self.compression_function(foreground_wav)
        if background_wav is not None:
            background_wav = self.compression_function(background_wav)
        return foreground_wav, background_wav


class AudioToAudioRepresentation(torch.nn.Module):
    """
    Base class for audio transformations. Takes in the audio and outputs
    a representation that is used for training.
    Args:
        rep_type (str): the type of representation to build
    """

    def __init__(self, rep_type, rep_kwargs, compression_type, compression_kwargs):
        super(AudioToAudioRepresentation, self).__init__()
        self.rep_type = rep_type
        self.rep_kwargs = rep_kwargs
        self.compression_type = compression_type
        self.compression_kwargs = compression_kwargs

        # Choose the representation type
        if self.rep_type == "mel_spec":
            self.rep = AudioToMelSpectrogram(melspec_kwargs=self.rep_kwargs)
        elif self.rep_type == "cochleagram":
            self.rep = AudioToCochleagram(cgram_kwargs=self.rep_kwargs)
        else:
            raise NotImplementedError(
                "Audio Representation of type " "%s is not implemented" % self.rep_type
            )

        # Choose the compression type
        if self.compression_type == "log":
            self.compression = LogScale(**self.compression_kwargs)
        elif self.compression_type == "log_fakeclamp":
            self.compression = LogScaleFakeClipping(**self.compression_kwargs)
        elif self.compression_type == "coch_p3":
            self.compression = ClippedGradPower(self.compression_kwargs)
        elif self.compression_type == "none":
            self.compression = None
        else:
            raise NotImplementedError(
                "Audio Compression of type "
                "%s is not implemented" % self.compression_type
            )

    def forward(self, foreground_wav, background_wav):
        del background_wav
        if foreground_wav is not None:
            foreground_wav = foreground_wav
            foreground_rep, background_rep = self.rep(foreground_wav, None)
            if self.compression is not None:
                foreground_rep, background_rep = self.compression(foreground_rep, None)
        else:
            foreground_rep = None
            background_rep = None
        return foreground_rep, background_rep


class AudioToMelSpectrogram(torch.nn.Module):
    """
    Converts audio to mel spectrogram.
    Args:
        melspec_kwargs (dict): dictionary containing the arguments used within
            torchaudio.MelSpectrogram
    """

    def __init__(self, melspec_kwargs={}):
        super(AudioToMelSpectrogram, self).__init__()
        self.melspec_kwargs = melspec_kwargs
        self.MelSpectrogram = torchaudio.transforms.MelSpectrogram(
            **self.melspec_kwargs
        )

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        del background_wav

        if foreground_wav is not None:
            foreground_mel = self.MelSpectrogram(foreground_wav)
        else:
            foreground_mel = None

        return foreground_mel, None


class AudioToCochleagram(torch.nn.Module):
    """
    Converts audio to cochleagram
    """

    def __init__(self, cgram_kwargs={}):
        super(AudioToCochleagram, self).__init__()
        self.cgram_kwargs = cgram_kwargs

        # Args used for multiple of the cochleagram operations
        self.signal_size = self.cgram_kwargs["signal_size"]
        self.sr = self.cgram_kwargs["sr"]
        self.pad_factor = self.cgram_kwargs["pad_factor"]
        self.use_rfft = self.cgram_kwargs["use_rfft"]

        # Define cochlear filters
        self.coch_filter_kwargs = self.cgram_kwargs["coch_filter_kwargs"]
        self.coch_filter_kwargs = {
            "use_rfft": self.use_rfft,
            "pad_factor": self.pad_factor,
            "filter_kwargs": self.coch_filter_kwargs,
        }

        self.make_coch_filters = self.cgram_kwargs["coch_filter_type"]
        self.filters = self.make_coch_filters(
            self.signal_size, self.sr, **self.coch_filter_kwargs
        )

        # Define an envelope extraction operation
        self.env_extraction = self.cgram_kwargs["env_extraction_type"]
        self.envelope_extraction = self.env_extraction(
            self.signal_size, self.sr, self.use_rfft, self.pad_factor
        )

        # Define a downsampling operation
        self.downsampling = self.cgram_kwargs["downsampling_type"]
        self.env_sr = self.cgram_kwargs["env_sr"]
        self.downsampling_kwargs = self.cgram_kwargs["downsampling_kwargs"]
        self.downsampling_op = self.downsampling(
            self.sr, self.env_sr, **self.downsampling_kwargs
        )

        # Compression is applied as a separate transform to be consistent with Spectrograms
        cochleagram = chcochleagram.cochleagram.Cochleagram(
            self.filters,
            self.envelope_extraction,
            self.downsampling_op,
            compression=None,
        )

        self.Cochleagram = cochleagram

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        del background_wav

        if foreground_wav is not None:
            foreground_coch = self.Cochleagram(foreground_wav)
        else:
            foreground_coch = None

        return foreground_coch, None


class AudioToTensor(torch.nn.Module):
    """
    Convert the foreground and background sounds to tensors

    Args:
        None

    Returns:
        foreground_wav, background_wav
    """

    def __init__(self):
        super(AudioToTensor, self).__init__()

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if background_wav is None:
            return torch.from_numpy(foreground_wav), None
        else:
            return torch.from_numpy(foreground_wav), torch.from_numpy(background_wav)


class UnsqueezeAudio(torch.nn.Module):
    """
    Adds a channel dimension (useful for mel-spectrograms as inputs)

    Args:
        None

    Returns:
        foreground_wav, background_wav
    """

    def __init__(self, dim=1):
        super(UnsqueezeAudio, self).__init__()
        self.dim = dim

    def forward(self, foreground_wav, background_wav):
        if foreground_wav is not None:
            foreground_wav = foreground_wav.unsqueeze(self.dim)
        if background_wav is not None:
            background_wav = background_wav.unsqueeze(self.dim)
        return foreground_wav, background_wav


class FilterNoneSpeech(torch.nn.Module):
    """
    Filter out speech audio samples that are all zeros.
    Useful for removing speech 'null' classes.

    Args:
        None

    Returns:
        foreground_wav, background_wav if passes filtering
        None if should be removed
    """

    def __init__(self):
        super(FilterNoneSpeech, self).__init__()

    def forward(self, foreground_wav, background_wav):
        if torch.sum(torch.pow(foreground_wav, 2)) == 0:
            foreground_wav = None
        if torch.sum(torch.pow(background_wav, 2)) == 0:
            background_wav = None
        else:
            return foreground_wav, background_wav


class RandomCropForegroundBackground(torch.nn.Module):
    """
    Randomly crops the foreground and background to make a shorter signal.
    """

    def __init__(self, signal_size, crop_length):
        super(RandomCropForegroundBackground, self).__init__()
        self.crop_length = crop_length
        self.signal_size = signal_size
        self.start_crop = int(signal_size - crop_length)

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        rand_start = torch.randint(self.start_crop, size=(2,))
        if foreground_wav is not None:
            foreground_wav = foreground_wav[
                rand_start[0] : rand_start[0] + self.crop_length
            ]
        if background_wav is not None:
            background_wav = background_wav[
                rand_start[1] : rand_start[1] + self.crop_length
            ]
        return foreground_wav, background_wav


class CenterCropForegroundRandomCropBackground(torch.nn.Module):
    """
    Center crops the foreground and randomly crops background to make a shorter signal.
    """

    def __init__(self, signal_size, crop_length):
        super(CenterCropForegroundRandomCropBackground, self).__init__()
        self.crop_length = crop_length
        self.signal_size = signal_size
        self.start_crop_random = int(signal_size - crop_length)
        self.start_crop_center = int((signal_size - crop_length) / 2)

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        rand_start = torch.randint(self.start_crop_random, size=(2,))
        if foreground_wav is not None:
            foreground_wav = foreground_wav[
                self.start_crop_center : self.start_crop_center + self.crop_length
            ]
        if background_wav is not None:
            background_wav = background_wav[
                rand_start[1] : rand_start[1] + self.crop_length
            ]
        return foreground_wav, background_wav


class RMSNormalizeForegroundAndBackground(torch.nn.Module):
    """
    RMS normalize the foreground and background sounds

    Args:
        rms_normalization (float): The rms level to set the sound to

    Returns:
        foreground_wav, background_wav
    """

    def __init__(self, rms_level=0.1):
        super(RMSNormalizeForegroundAndBackground, self).__init__()
        self.rms_level = rms_level

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if foreground_wav is not None:
            foreground_wav = ch_demean(foreground_wav)
            rms_foreground = ch_rms(foreground_wav)
            if rms_foreground != 0:
                foreground_wav = foreground_wav * self.rms_level / rms_foreground
            else:
                foreground_wav = None

        if background_wav is not None:
            background_wav = ch_demean(background_wav)
            rms_background = ch_rms(background_wav)
            if rms_background != 0:
                background_wav = background_wav * self.rms_level / rms_background
            else:
                background_wav = None

        return foreground_wav, background_wav


class DBSPLNormalizeForegroundAndBackground(torch.nn.Module):
    """
    Set the foreground and background sounds to a specified sound pressure
    level (dBSPL)

    Args:
        dbspl (float): desired sound pressure level in dB re 20e-6 Pa

    Returns:
        foreground_wav, background_wav
    """

    def __init__(self, dbspl=60):
        super(DBSPLNormalizeForegroundAndBackground, self).__init__()
        self.dbspl = dbspl
        self.rms_level = 20e-6 * np.power(10.0, self.dbspl / 20.0)

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if foreground_wav is not None:
            foreground_wav = ch_demean(foreground_wav)
            rms_foreground = ch_rms(foreground_wav)
            if rms_foreground != 0:
                foreground_wav = foreground_wav * self.rms_level / rms_foreground
            else:
                foreground_wav = None

        if background_wav is not None:
            background_wav = ch_demean(background_wav)
            rms_background = ch_rms(background_wav)
            if rms_background != 0:
                background_wav = background_wav * self.rms_level / rms_background
            else:
                background_wav = None

        return foreground_wav, background_wav


class FlipForegroundAndBackground(torch.nn.Module):
    """
    Turns the foreground signal into the background signal and
    vice versa (useful for training without any combinations)

    Returns:
        foreground_wav, background_wav
    """

    def __init__(self):
        super(FlipForegroundAndBackground, self).__init__()

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        return background_wav, foreground_wav


class CombineWithRandomDBSNR(torch.nn.Module):
    """
    Takes two signals and combines them at a specified dB SNR level.

    Args:
        low_snr (float): the low end for the range of dB SNR to draw from
        high_snr (float): the high end for the range of db SNR to draw from
        rms_level (float): the end RMS value for the combined sound

    Returns:
        signal_in_noise, None

    """

    def __init__(self, low_snr=-10, high_snr=10):
        self.low_snr = low_snr
        self.high_snr = high_snr
        super(CombineWithRandomDBSNR, self).__init__()

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        rand_db_snr = self.low_snr + (self.high_snr - self.low_snr) * torch.rand(1)
        rms_ratio = np.power(10.0, rand_db_snr / 20.0)
        # Demean signal and noise before computing rms
        if foreground_wav is not None:
            foreground_wav = ch_demean(foreground_wav)
            rms_foreground = ch_rms(foreground_wav)
        else:
            rms_foreground = 0
            foreground_wav = torch.zeros(background_wav.shape)
        if background_wav is not None:
            background_wav = ch_demean(background_wav)
            rms_background = ch_rms(background_wav)
        else:
            rms_background = 0
            background_wav = torch.zeros(foreground_wav.shape)

        # Calculate the scale factor for the two sounds
        # For now, to align with the jsinv3 dataset, we include the infinite SNR
        # cases
        if rms_foreground == 0:  # No foreground condition (just noise)
            noise_scale_factor = 1
        elif rms_background == 0:
            noise_scale_factor = 0
        else:
            noise_scale_factor = torch.div(
                rms_foreground, torch.mul(rms_background, rms_ratio)
            )

        background_wav = torch.mul(noise_scale_factor, background_wav)
        signal_in_noise = torch.add(foreground_wav, background_wav)

        return signal_in_noise, None


## Spectrogram Based Transforms
### Augmentations ported from byol-a: https://github.com/nttcslab/byol-a/tree/master
class RandomResizeCrop(nn.Module):
    """Random Resize Crop block.

    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(
        self,
        virtual_crop_scale=(1.0, 1.5),
        freq_scale=(0.6, 1.5),
        time_scale=(0.6, 1.5),
    ):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = "bicubic"
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms, lms2):
        return self._forward(lms), None

    def _forward(self, lms):
        # make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
        virtual_crop_size = [
            int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)
        ]
        virtual_crop_area = (
            torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
            .to(torch.float)
            .to(lms.device)
        )
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y : y + h, x : x + w] = lms
        # get random area
        i, j, h, w = self.get_params(
            virtual_crop_area.shape[-2:],
            lms.shape[-2:],
            self.time_scale,
            self.freq_scale,
        )
        crop = virtual_crop_area[:, i : i + h, j : j + w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = F.interpolate(
            crop.unsqueeze(0),
            size=lms.shape[-2:],
            mode=self.interpolation,
            align_corners=True,
        ).squeeze(0)
        return lms.to(torch.float)

    def __repr__(self):
        format_string = (
            self.__class__.__name__ + f"(virtual_crop_size={self.virtual_crop_scale}"
        )
        format_string += ", time_scale={0}".format(
            tuple(round(s, 4) for s in self.time_scale)
        )
        format_string += ", freq_scale={0})".format(
            tuple(round(r, 4) for r in self.freq_scale)
        )
        return format_string


class RandomLinearFader(nn.Module):
    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = gain

    def forward(self, lms, lms2):
        # return self._forward(lms), self._forward(lms2)
        return self._forward(lms), None

    def _forward(self, lms):
        head, tail = self.gain * (
            (2.0 * np.random.rand(2)) - 1.0
        )  # gain * U(-1., 1) for two ends
        T = lms.shape[2]
        slope = (
            torch.linspace(head, tail, T, dtype=lms.dtype)
            .reshape(1, 1, T)
            .to(lms.device)
        )
        y = lms + slope  # add liniear slope to log-scale input
        return y

    def __repr__(self):
        format_string = self.__class__.__name__ + f"(gain={self.gain})"
        return format_string


## Signal Mixers: Probably not necessary (prefer existing CombineWithRandomDBSNR)
def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1.0 - alpha) * xb
    return torch.log(x + torch.finfo(x.dtype).eps)


class MixupBYOLA(nn.Module):
    """Mixup for BYOL-A.

    Args:
        ratio: Alpha in the paper.
        n_memory: Size of memory bank FIFO.
        log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
    """

    def __init__(self, ratio=0.4, n_memory=2048, log_mixup_exp=True):
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []

    def forward(self, x):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))]
            # mix them
            mixed = (
                log_mixup_exp(x, z, 1.0 - alpha)
                if self.log_mixup_exp
                else alpha * z + (1.0 - alpha) * x
            )
        else:
            mixed = x
        # update memory bank
        self.memory_bank = (self.memory_bank + [x])[-self.n :]

        return mixed.to(torch.float)

    def __repr__(self):
        format_string = self.__class__.__name__ + f"(ratio={self.ratio},n={self.n}"
        format_string += f",log_mixup_exp={self.log_mixup_exp})"
        return format_string


class MixGaussianNoise(nn.Module):
    """Gaussian Noise Mixer.
    This interpolates with random sample, unlike Mixup.
    """

    def __init__(self, ratio=0.2):
        super().__init__()
        self.ratio = ratio

    def forward(self, lms):
        x = lms.exp()

        lambd = self.ratio * np.random.rand()
        z = torch.normal(0, lambd, x.shape).exp().to(lms.device)
        mixed = (1 - lambd) * x + z + torch.finfo(x.dtype).eps

        return mixed.log()

    def __repr__(self):
        format_string = self.__class__.__name__ + f"(ratio={self.ratio})"
        return format_string
