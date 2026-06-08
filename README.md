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

## Configs And Checkpoints

Each config uses `training.save_dir: checkpoint`. The packaged checkpoints use
matching SOFIA names:

```text
config/sofia_a1_fx.yaml                  checkpoint/sofia_A1_fx/sofia_A1_fx.pt
config/sofia_g1_mert.yaml                checkpoint/sofia_G1_mert/sofia_G1_mert.pt
config/sofia_g1_muq.yaml                 checkpoint/sofia_G1_muq/sofia_G1_muq.pt
config/sofia_g2_mert_w_muq.yaml          checkpoint/sofia_G2_muq_w_mert/sofia_G2_muq_w_mert.pt
config/sofia_v1_rawnet.yaml              checkpoint/sofia_V1_rawnet/sofia_V1_rawnet.pt
config/sofia_v1_wave.yaml                checkpoint/sofia_V1_wave/sofia_V1_wave.pt
config/sofia_vag_concat.yaml             checkpoint/sofia_vag_concat/sofia_vag_concat.pt
config/sofia_vag_moe.yaml                checkpoint/sofia_vag_moe/sofia_vag_moe.pt
config/sofia_vag_sample_gating.yaml      checkpoint/sofia_vag_sample_gating/sofia_vag_sample_gating.pt
config/sofia_vag_wo_fx.yaml              checkpoint/sofia_vag_moe_wo_fx/sofia_vag_moe_wo_fx.pt
config/sofia_vag_wo_mert.yaml            checkpoint/sofia_vag_moe_wo_mert/sofia_vag_moe_wo_mert.pt
config/sofia_vag_wo_muq.yaml             checkpoint/sofia_vag_moe_wo_muq/sofia_vag_moe_wo_muq.pt
config/sofia_vag_wo_rawnet.yaml          checkpoint/sofia_vag_moe_wo_rawnet/sofia_vag_moe_wo_rawnet.pt
config/sofia_vag_wo_wave.yaml            checkpoint/sofia_vag_moe_wo_wave/sofia_vag_moe_wo_wave.pt
```

## CSV Format

Dataset CSV files should include:

```csv
full_path,vocal_path,label,source
audio/song_001.wav,vocals/song_001.wav,0,human
audio/song_002.wav,,1,ai
```

`full_path` and `label` are required. `vocal_path` and `source` are optional.
