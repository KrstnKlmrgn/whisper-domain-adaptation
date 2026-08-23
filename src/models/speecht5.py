"""
SpeechT5-based text-to-spectrogram and speech synthesis model.

Generates spectrograms from text, optionally enhancing them,
and converting them to waveform audio using a HiFi-GAN vocoder.
"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

from utils.tts_utils import (
    restore_random_state,
    save_random_state,
)



class SpeechT5TTS:

    def __init__(self, config, device):

        self.device = device

        model_name = config["speecht5"]["model_name"]

        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.vocoder = SpeechT5HifiGan.from_pretrained(
            config["speecht5"]["vocoder_name"]
        )
        self.vocoder.to(device)
        self.vocoder.eval()

        self.processor = SpeechT5Processor.from_pretrained(model_name)

        self.sample_rate = self.processor.feature_extractor.sampling_rate


        self.speaker_embeddings_type = config["speecht5"]["speaker_embeddings"]["type"]

        if self.speaker_embeddings_type == "random":

            embedding_config = config["speecht5"]["speaker_embeddings"]

            self.embeddings = load_dataset(
                embedding_config["dataset"],
                name=embedding_config["dataset_config"],
                split=embedding_config["split"]
            )

            self.num_embeddings = len(self.embeddings)



    def get_speaker_embedding(self):
        """
        Return a speaker embedding according to the strategy
        specified in the configuration.
        """

        if self.speaker_embeddings_type == "zero":
            return torch.zeros((1, 512), device=self.device)


        elif self.speaker_embeddings_type == "random":
            emb_id = np.random.randint(0, self.num_embeddings)

            embedding = torch.tensor(
                self.embeddings[emb_id]["xvector"]
            )

            return embedding.unsqueeze(0).to(self.device)


        else:
            raise ValueError(
                f"Unknown speaker embedding type: {self.speaker_embeddings_type}"
            )



    def generate(self, text):
        """
        Generate a mel-spectrogram from input text.
        """

        speaker_embeddings = self.get_speaker_embedding()

        inputs = self.processor(
            text=text, return_tensors="pt"
        )

        with torch.no_grad():
            spectrogram = self.model.generate(
                inputs["input_ids"].to(self.device),
                speaker_embeddings=speaker_embeddings
            )

        # SpeechT5 returns [time, mel]. Convert to the common
        # [batch, mel, time] format used by the TTS pipeline.
        spectrogram = (
            spectrogram.transpose(0, -1).unsqueeze(0)
        )


        return spectrogram



    def enhance(self, enhancer, spectrogram, seed):
        """
        Enhance a spectrogram using the random seed
        specified in the configuration.
        """

        # Enhancer expects spectrograms in [batch, mel, time] format
        lengths = torch.tensor([spectrogram.shape[-1]]).to(self.device)

        # Preserve the global random state so that enhancement does not
        # affect randomness elsewhere in the experiment.
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

        # Vocoder expects [time, mel]
        # Convert from [batch, mel, time] to [time, mel]
        spectrogram = (
            spectrogram.squeeze(0).transpose(0, 1)
        )

        with torch.no_grad():
            audio = self.vocoder(spectrogram)


        return audio.cpu().numpy()