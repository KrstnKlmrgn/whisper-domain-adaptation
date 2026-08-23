# Whisper Domain Adaptation

This repository contains the code for experiments on **domain adaptation of OpenAI Whisper for automatic speech recognition (ASR)** using domain-specific text data.

The main focus is **text-only domain adaptation**, where domain-specific text is used to adapt Whisper without requiring additional real speech data. Several approaches are investigated and compared, including TTS-based adaptation, language model fusion, internal language model adaptation, and spectrogram enhancement.

## Methods

### Whisper Fine-tuning

Standard Whisper fine-tuning can be performed using either:

* **Audio input** – conventional ASR fine-tuning.
* **TTS-generated spectrograms** – synthetic speech representations generated from text.

### TTS-based Adaptation

As a baseline, domain-specific text is converted into synthetic speech using a TTS model. The available TTS models are:

* FastPitch
* SpeechT5

Generated spectrograms can optionally be enhanced using a **StyleGAN2-based spectrogram enhancer** before being used for Whisper fine-tuning.

### Shallow Fusion

Shallow fusion combines Whisper decoding with an external **GPT-2 language model** during inference.

Configurable parameters include:

* `num_beams` – beam search size
* `weight` – language model coefficient

### Deep Fusion

Deep fusion integrates GPT-2 into the Whisper decoder during fine-tuning.

The following options can be configured:

* `use_lm` – whether the external language model is used
* `copy_weights` – whether pretrained Whisper weights are copied
* `gate_type` – `coarse` or `fine` grained gating

### Internal Language Model Adaptation (ILMA)

ILMA investigates Whisper's internal language model by removing the contribution of the acoustic input during training.

The encoder input can be replaced by a dummy input, including zero or random input. Random inputs can be controlled using their standard deviation and random seed.

## Language Model

GPT-2 can be independently fine-tuned on domain-specific text and evaluated using perplexity.

Different tokenization strategies are supported:

* `gpt2`
* `whisper_en`
* `whisper_multi`

GPT-2 embeddings can optionally be reordered to better match the Whisper tokenizer vocabulary:

* `none`
* `copy`
* `fvt`

## Configuration

Experiments are configured using YAML files in `configs/`.

| Configuration               | Purpose                                    |
| --------------------------- | ------------------------------------------ |
| `whisper_train_config.yaml` | Whisper fine-tuning, Deep Fusion and ILMA  |
| `inference_config.yaml`     | Whisper inference and fusion               |
| `tts_config.yaml`           | TTS generation and spectrogram enhancement |
| `lm_train_config.yaml`      | GPT-2 domain adaptation                    |
| `lm_evaluation_config.yaml` | GPT-2 evaluation                           |

Paths, models, datasets, training parameters, and experiment-specific options can be changed directly in the corresponding configuration file.

## Main Scripts

```text
train_whisper.py   # Fine-tune Whisper
inference.py       # Run ASR inference and evaluate WER/PPL
generate_tts.py    # Generate synthetic speech/spectrograms
train_lm.py        # Fine-tune GPT-2 on domain text
evaluatelm.py      # Evaluate GPT-2 perplexity
```

## Requirements

The code has been tested with **Python 3.8 and 3.10.14**.

Install the required dependencies using:

```text
pip install -r requirements.txt
```

## Usage

The scripts are configured through the YAML files in `configs/`. After configuring the desired experiment, run the corresponding Python script directly.

For example:

```text
python train_whisper.py
```

Command-line arguments are not supported; experiment settings must be specified in the corresponding configuration file.
