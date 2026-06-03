# sofia

Migrated MONICA audio detector code with project-local paths, optional Demucs
vocal separation, and configurable encoder dependencies.

## Setup

```bash
cd /path/to/parent/of/sofia
python -m pip install -r sofia/requirements.txt
```

Run scripts from the parent directory so `sofia` is importable:

```bash
python -m sofia.train --config sofia/config/default.yaml --run_name experiment
```

The standalone scripts also add the package parent to `sys.path`, so direct
execution still works from the repository checkout.

## Paths

Relative paths in YAML configs are resolved against the `sofia` project root.
Environment variables and `~` are expanded before resolution. This applies to:

- `data.train_csv`
- `data.val_csv`
- `training.save_dir`
- encoder `weights`
- encoder `config`
- encoder `repo_path`

For another server, either keep the same project-local layout or override these
paths in a copied config file.

## External Encoders

Fx-Encoder++ and MuQ can be installed as importable Python packages, or provided
as source checkouts:

```yaml
audio_branches:
  fxpp:
    repo_path: /path/to/Fx-Encoder_PlusPlus-main
  muq:
    repo_path: /path/to/muq
```

Equivalent environment variables are also supported:

```bash
export FXPP_REPO=/path/to/Fx-Encoder_PlusPlus-main
export MUQ_REPO=/path/to/muq
```

RawNet is loaded from `audio_branches.rawnet.repo_path`, with its config and
checkpoint from the corresponding `config` and `weights` entries.

## Vocal Separation

If `data.demucs_enabled` is true and a CSV row has no usable `vocal_path`, the
dataset attempts to separate vocals from `full_path` with the configured Demucs
model. If separation fails, the sample falls back to silence for vocal branches.
