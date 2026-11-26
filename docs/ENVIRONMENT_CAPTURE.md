# Environment capture and lockfile status

## Existing evidence

The Pilot B manifest records Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0, and an NVIDIA Tesla T4. The file `requirements-t4-pilot-b.txt` is explicitly a **partial pin fragment**, not a complete lockfile: the original run did not preserve a complete `pip freeze` or exact transitive dependency versions.

Do not label that file a complete environment lock, and do not silently fill missing versions with current package releases.

## Capture a complete environment on the next run

Run this inside the exact environment used for training and evaluation:

```bash
python -VV
python -m pip --version
python -m pip freeze --all > requirements-lock.txt
python -m pip check
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

Commit the generated lockfile alongside a run manifest. For stronger reproducibility, also record the OS image/container digest, NVIDIA driver version, model revision, dataset revision, tokenizer revision, and SHA-256 hashes of all downloaded artifacts.

## Data provenance

The Pilot B manifest records FineWeb-Edu as the training source but does not identify an immutable dataset revision and per-file checksum. Record those values in future runs. If data is streamed, record the dataset repository revision, configuration, split, streaming/shuffle settings, seed, and exact token count.

## Do not overstate reproducibility

A command line and partial package list are useful provenance, but they do not guarantee bitwise reproducibility across GPU models, CUDA drivers, library kernels, or distributed execution. State what is reproducible (configuration, evaluation protocol, artifact checksums) and what is not (bitwise-identical training, if not verified).
