"""
FastPitch-based text-to-spectrogram and speech synthesis model.

Generates spectrograms from text, optionally enhancing them,
and converting them to waveform audio using a HiFi-GAN vocoder.
"""

import numpy as np
import torch
from nemo.collections.tts.models import (
    FastPitchModel,
    HifiGanModel,
)

from utils.tts_utils import (
    restore_random_state,
    save_random_state,
)



class FastPitchTTS:

    def __init__(self, config, device):

        self.device = device

        self.model = FastPitchModel.from_pretrained(
            model_name=config["fastpitch"]["model_name"]
        )
        self.model.to(device)
        self.model.eval()

        self.vocoder = HifiGanModel.from_pretrained(
            model_name=config["fastpitch"]["vocoder_name"]
        )
        self.vocoder.to(device)
        self.vocoder.eval()

        self.sample_rate = self.model.cfg.sample_rate

        self.speaker_id = config["fastpitch"]["speaker_id"]

        self.n_speakers = self.model.cfg.n_speakers



    def get_speaker_id(self):
        """
        Return the configured speaker ID or a random valid speaker.
        """

        speaker_id = self.speaker_id

        if isinstance(speaker_id, int):

            if 0 <= speaker_id < self.n_speakers:
                return speaker_id
            else:
                print(
                    f"Invalid speaker_id {speaker_id}, using random speaker."
                )

        return np.random.randint(0, self.n_speakers)



    def generate(self, text):
        """
        Generate a mel-spectrogram from input text.
        """

        speaker = self.get_speaker_id()

        with torch.no_grad():

            tokens = self.model.parse(text)

            spectrogram = self.model.generate_spectrogram(
                tokens=tokens, speaker=speaker
            )

        return spectrogram



    def enhance(self, enhancer, spectrogram, seed):
        """
        Enhance a spectrogram using the configured random seed.
        """

        lengths = torch.tensor([spectrogram.shape[-1]]).to(self.device)

        # Preserve the global random state so that enhancement does not
        # affect randomness elsewhere in the experiment
        state = save_random_state()
        torch.manual_seed(seed)

        with torch.no_grad():
            enhanced_spectrogram = enhancer(
                input_spectrograms=spectrogram,
                lengths=lengths
            )

        restore_random_state(state)

        return enhanced_spectrogram



    def vocode(self, spectrogram):
        """
        Convert a spectrogram to waveform audio using HiFi-GAN.
        """

        with torch.no_grad():
            audio = self.vocoder.convert_spectrogram_to_audio(
                spec=spectrogram
            )

        return audio[0].cpu().numpy()