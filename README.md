# SOFIA

SOFIA is a multi-encoder synthetic song detector. This migrated version uses
project-relative paths, configurable encoder locations, automatic audio
resampling, and optional Demucs vocal separation.

## Setup

```bash
cd /path/to/sofia
python -m pip install -r requirements.txt
```

If the pinned `torch`/`torchaudio` wheels do not match your CUDA runtime,
install the matching PyTorch wheels first, then install the remaining packages.
`ffmpeg` is only required for `augment_audio.py`.

## Weights

Put downloaded weights directly under each encoder's project directory:

```text
weights/fxpp/fxenc_plusplus_default.pt
third_party/muq_weights/model.safetensors
third_party/muq_weights/pytorch_model.bin        # optional alternative
third_party/mert/pytorch_model.bin
third_party/mert/MERT-v1-95M_fairseq.pt
third_party/wav2vec/pytorch_model.bin
weights/rawnet/epoch_49.pth
```

RawNet `weights/rawnet/epoch_49.pth` is included in this repository.

Download links:

- fx: https://huggingface.co/yytung/fxencoder-plusplus/tree/main
- muq: https://huggingface.co/OpenMuQ/MuQ-MuLan-large
- mert: https://huggingface.co/m-a-p/MERT-v1-95M
- wave2: https://huggingface.co/facebook/wav2vec2-base

Lightweight encoder source/config files are included under `third_party/`.
The YAML configs already point to these project-relative locations.

## CSV Format

Training and dataset testing read CSV files with `label` and `full_path` columns.
`vocal_path` and `source` are optional.

```csv
full_path,vocal_path,label,source
audio/song_001.wav,vocals/song_001.wav,0,human
audio/song_002.wav,,1,ai
```

If `vocal_path` is missing and `data.demucs_enabled` is true, vocals are
separated from `full_path` automatically when a branch needs vocals.

## Training

Run from the parent directory of `sofia`:

```bash
cd /path/to
python -m sofia.train \
  --config sofia/config/default.yaml \
  --run_name experiment \
  --device cuda:0
```

Resume from a checkpoint:

```bash
python -m sofia.train \
  --config sofia/config/default.yaml \
  --run_name experiment \
  --resume sofia/outputs/experiment/experiment_epoch10.pt
```

## Dataset Testing From CSV

Use `test.py` for loss and accuracy:

```bash
cd /path/to
python -m sofia.test \
  --config sofia/config/default.yaml \
  --checkpoint sofia/outputs/experiment/experiment_epoch10.pt \
  --csv sofia/data/test.csv \
  --device cuda:0
```

Use `test_f1_acc.py` for accuracy and F1:

```bash
python -m sofia.test_f1_acc \
  --config sofia/config/default.yaml \
  --checkpoint sofia/outputs/experiment/experiment_epoch10.pt \
  --csv sofia/data/test.csv \
  --device cuda:0
```

## Single-Audio Testing

`predict_audio.py` tests one audio file without a CSV. It prints `true` for
label `1` and `false` for label `0`.

```bash
cd /path/to
python -m sofia.predict_audio \
  --config sofia/config/rawnet_only.yaml \
  --checkpoint sofia/outputs/rawnet_only/rawnet_only_epoch1.pt \
  --audio /path/to/audio.wav \
  --device cuda:0
```

Useful options:

- `--vocal_audio /path/to/vocals.wav`: use a precomputed vocal stem.
- `--json`: print prediction details.
- `--quiet`: print only `true` or `false`.
