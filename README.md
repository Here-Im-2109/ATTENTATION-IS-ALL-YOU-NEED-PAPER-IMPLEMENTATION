# Bilingual Translation with a Transformer from Scratch

A from-scratch PyTorch implementation of the Transformer architecture from
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
applied to bilingual machine translation. By default it translates **English → Italian**
using the [`opus_books`](https://huggingface.co/datasets/Helsinki-NLP/opus_books) dataset, but any
language pair available for the dataset can be configured.

Every building block (multi-head attention, positional encoding, layer norm, residual
connections, encoder/decoder stacks) is written by hand in `model.py` with no use of
`nn.Transformer`, so the code is meant to be read and learned from.

## Features

- Full encoder–decoder Transformer (6 layers, 8 heads, `d_model=512`, `d_ff=2048` by default)
- Word-level tokenizers built with Hugging Face `tokenizers` (cached to disk)
- Training with label smoothing, Adam, and checkpoint save/resume every epoch
- Per-epoch validation with sample translations and **CER / WER / BLEU** metrics
- Logging via **TensorBoard** (`train.py`) or **Weights & Biases** (`train_wb.py`)
- Greedy decoding (`translate.py`) and **beam search** (`Beam_Search.ipynb`)
- Attention heatmap visualization with Altair (`attention_visual.ipynb`)
- Ready-to-use notebooks for local training, Google Colab, and inference

## Project Structure

| File | Description |
|---|---|
| `model.py` | Transformer implementation: embeddings, positional encoding, multi-head attention, encoder, decoder, `build_transformer()` |
| `dataset.py` | `BilingualDataset` (adds `[SOS]`/`[EOS]`/`[PAD]`, builds padding and causal masks) |
| `config.py` | Hyperparameters and weight-file path helpers |
| `train.py` | Training loop with TensorBoard logging, greedy-decode validation |
| `train_wb.py` | Same training loop, logging to Weights & Biases |
| `translate.py` | CLI inference: translate a sentence or a dataset index using the latest checkpoint |
| `Local_Train.ipynb` | Train locally from a notebook |
| `Colab_Train.ipynb` | Train on Google Colab (Drive mounting, TensorBoard) |
| `Inference.ipynb` | Run validation examples and translate custom sentences |
| `Beam_Search.ipynb` | Beam search decoding |
| `attention_visual.ipynb` | Visualize encoder/decoder attention heads |
| `requirements.txt` / `conda.txt` | pip and (Linux) conda dependency lists |

## Installation

Python **3.9** is recommended (the pinned versions target it).

```bash
git clone <https://github.com/Here-Im-2109/ATTENTATION-IS-ALL-YOU-NEED-PAPER-IMPLEMENTATION.git>
cd ATTENTATION-IS-ALL-YOU-NEED-PAPER-IMPLEMENTATION

python create -p <VENV> python=3.9 -y
# Windows: conda acivate <VENV>    |  Linux: conda acivate <VENV>

pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.0.1`, `torchtext==0.15.2`, `datasets==2.15.0`,
`tokenizers==0.13.3`, `torchmetrics==1.0.3`, `tensorboard==2.13.0`, `altair==5.1.1`
and `wandb==0.15.9`. For GPU training, install the CUDA build of PyTorch that matches
your driver. Alternatively, on Linux you can recreate the environment with
`conda create --name <env> --file conda.txt`.

## Configuration

All settings live in `config.py`:

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | `8` | Training batch size |
| `num_epochs` | `20` | Number of epochs |
| `lr` | `1e-4` | Adam learning rate |
| `seq_len` | `350` | Max sequence length (including special tokens) |
| `d_model` | `512` | Embedding / model dimension |
| `datasource` | `opus_books` | Hugging Face dataset name |
| `lang_src` / `lang_tgt` | `en` / `it` | Source and target language codes |
| `model_folder` | `weights` | Checkpoints go to `<datasource>_<model_folder>/` |
| `preload` | `latest` | `latest`, an epoch string such as `"05"`, or `None` to start fresh |
| `tokenizer_file` | `tokenizer_{0}.json` | Tokenizer cache path |
| `experiment_name` | `runs/tmodel` | TensorBoard log directory |

To change the language pair, edit `lang_src` / `lang_tgt` (the pair must exist for the
chosen dataset, e.g. `en-fr`) and delete any previously generated `tokenizer_*.json` files.

## Usage

### Train

```bash
python train.py          # TensorBoard logging
python train_wb.py       # Weights & Biases logging (requires `wandb login`)
```

The dataset is downloaded automatically, split 90% / 10% into train / validation, and
tokenizers are built on first run. A checkpoint (model, optimizer, epoch, global step)
is saved to `opus_books_weights/tmodel_<epoch>.pt` after every epoch. With
`preload: "latest"`, re-running the script resumes from the newest checkpoint.

Monitor training:

```bash
tensorboard --logdir runs
```

Notebook alternatives: `Local_Train.ipynb` (local) and `Colab_Train.ipynb` (Colab GPU).

### Translate

```bash
# Translate a custom sentence
python translate.py "I am not a very good student."

# Translate sentence number 34 from the dataset and compare with the reference
python translate.py 34
```

This uses the latest checkpoint in the weights folder together with the saved tokenizers.

### Beam search and attention visualization

- `Beam_Search.ipynb` implements beam search decoding on top of a trained checkpoint.
- `attention_visual.ipynb` renders encoder self-attention, decoder self-attention, and
  cross-attention heatmaps per layer and head.

Both notebooks load a specific checkpoint by epoch number, so adjust the epoch in
`get_weights_file_path(config, "<epoch>")` to one you have trained.

## How It Works

1. **Tokenization**: word-level vocabularies (`[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`) per language.
2. **Inputs**: the encoder receives `[SOS] src [EOS] [PAD]…`; the decoder receives
   `[SOS] tgt [PAD]…` and is trained to predict `tgt [EOS] [PAD]…`.
3. **Masking**: a padding mask hides `[PAD]` tokens; a causal mask stops the decoder
   from attending to future positions.
4. **Loss**: cross entropy with label smoothing (0.1), ignoring padding.
5. **Inference**: greedy decoding picks the highest-probability token at each step;
   beam search keeps the top-*k* hypotheses.

## Notes

- `.gitignore` excludes checkpoints (`*.pt`), tokenizer files (`*.json`), `runs/`,
  `weights/` and `wandb/`, so these are regenerated locally by training.
- Sentences longer than `seq_len` raise a `ValueError` in `BilingualDataset`.
- `attention_visual.ipynb` expects a checkpoint named `tmodel_29.pt`, so train at least
  30 epochs or change the epoch it loads.
  point it at your own repo if you want to train this exact code.

## Acknowledgements

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017
- [Hugging Face](https://huggingface.co/) `datasets` and `tokenizers`