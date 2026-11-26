# Environment capture and lockfile status

## Existing evidence

Pilot B records Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0, and an NVIDIA Tesla T4. `requirements-t4-pilot-b.txt` is a **partial pin fragment**, not a complete lockfile: the original run did not preserve a complete `pip freeze` or exact transitive dependency versions.

Do not label that file a complete lockfile or silently fill missing versions with current package releases.

## Capture the next run

Inside the exact training/evaluation environment, run:

```bash
python -VV
python -m pip --version
python -m pip freeze --all > requirements-lock.txt
python -m pip check
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

Commit the generated lockfile with a run manifest. Also record OS/container digest, NVIDIA driver, immutable model revision, dataset revision, tokenizer revision, and SHA-256 hashes of downloaded artifacts.

## Data provenance

Pilot B identifies FineWeb-Edu as the training source but does not preserve an immutable dataset revision and per-file checksum. Future runs should record dataset repository revision, configuration, split, streaming/shuffle settings, seed, and exact token count.

A command line and partial package list are useful provenance, but do not guarantee bitwise reproducibility across GPU models, CUDA drivers, kernels, or distributed execution. State exactly what has and has not been reproduced.
