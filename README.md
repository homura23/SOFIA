# SOFIA

SOFIA is a multi-encoder synthetic song detector. The current repository keeps
the migrated encoder wrappers, model configs, and trained SOFIA checkpoints with
project-relative paths.

## Directory Layout

```text
config/        SOFIA model configs
checkpoint/    trained SOFIA checkpoints
encoders/      encoder wrappers for fxpp, muq, mert, wave, and rawnet
third_party/   local encoder source files and pretrained encoder weights
data/          dataset loading utilities
models/        fusion and classifier modules
```

## Encoder Weights

Encoder weights should be placed under their own project directories:

```text
third_party/fxencoder_plusplus/fxenc_plusplus_default.pt
third_party/muq_weights/model.safetensors
third_party/muq_weights/pytorch_model.bin
third_party/mert/pytorch_model.bin
third_party/mert/MERT-v1-95M_fairseq.pt
third_party/wav2vec/pytorch_model.bin
third_party/rawnet/epoch_49.pth
```

Download references:

- fx: https://huggingface.co/yytung/fxencoder-plusplus/tree/main
- muq: https://huggingface.co/OpenMuQ/MuQ-MuLan-large
- mert: https://huggingface.co/m-a-p/MERT-v1-95M
- wave2: https://huggingface.co/facebook/wav2vec2-base

## CSV Format

Dataset CSV files should include:

```csv
full_path,vocal_path,label,source
audio/song_001.wav,vocals/song_001.wav,0,human
audio/song_002.wav,,1,ai
```

`full_path` and `label` are required. `vocal_path` and `source` are optional.

## Training

The commands below use SOFIA's standard entry modules: `sofia.train`,
`sofia.test`, `sofia.test_f1_acc`, and `sofia.predict_audio`.

Run training from the parent directory of `sofia`:

```bash
cd /path/to
python -m sofia.train \
  --config sofia/config/sofia_vag_concat.yaml \
  --run_name sofia_vag_concat \
  --device cuda:0
```

Checkpoints are saved under `checkpoint/` because each config uses
`training.save_dir: checkpoint`.

## Dataset Testing

Use a CSV file with the format above:

```bash
cd /path/to
python -m sofia.test \
  --config sofia/config/sofia_vag_concat.yaml \
  --checkpoint sofia/checkpoint/sofia_vag_concat/sofia_vag_concat.pt \
  --csv sofia/data/test.csv \
  --device cuda:0
```

For accuracy and F1:

```bash
python -m sofia.test_f1_acc \
  --config sofia/config/sofia_vag_concat.yaml \
  --checkpoint sofia/checkpoint/sofia_vag_concat/sofia_vag_concat.pt \
  --csv sofia/data/test.csv \
  --device cuda:0
```

## Single-Audio Testing

```bash
cd /path/to
python -m sofia.predict_audio \
  --config sofia/config/sofia_v1_rawnet.yaml \
  --checkpoint sofia/checkpoint/sofia_V1_rawnet/sofia_V1_rawnet.pt \
  --audio /path/to/audio.wav \
  --device cuda:0
```

Use `--vocal_audio /path/to/vocals.wav` when a precomputed vocal stem is
available.
