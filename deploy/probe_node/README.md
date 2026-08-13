# LAIM environment probe node

A throwaway diagnostic node. It trains nothing, writes nothing (except one
temp file it deletes), and cannot crash — every check is guarded, and only the
standard library is imported at module level, so it still produces a full
report on an image where `torch` or `jax` are broken.

## Deploy

1. Create a new git repository.
2. Copy **both** files from this directory to its **root**:
   ```
   probe.py
   descriptor.json
   ```
   (`descriptor.json` must be at the repo root and keep that exact name;
   `probe.py` is the entry file named in `script.runConfiguration`.)
3. Register it as a node on the **same base image as the detector node**
   (`py312-gpu`) — probing a different image tells us nothing.
4. Optional but valuable: connect the two input ports to the *same* sources the
   real node uses — `path_probe_data` → your traces parquet port,
   `path_probe_model` → the embedder port. Without them the probe still answers
   everything about the image, GPUs and limits; with them it also answers what
   the platform physically delivers to a port.
5. Run it and send back the **whole log** (it also lands on the `probe_text`
   and `probe_report` output ports, and renders under Results).

Runtime is well under a minute unless the ports carry large data — the probe
never reads the payloads, only lists and stats them.

## What each section answers

| Section in the log | Question it settles |
|---|---|
| `01_python_runtime` | Which interpreter and which site-packages directories are in play (the failing torch lives in `/usr/local/lib64/...`) |
| `02_cpu_limits` | **Q5** — the real cgroup CPU quota. `effective_cores_from_cgroup` is the number that matters, not `os.cpu_count()` |
| `03_memory_limits` | The container memory ceiling, for sizing the 56 GB input |
| `04_disk_space` | Free space on `/tmp` and whether `model_store_dir` is writable (train mode must persist the model bundle there) |
| `05_gpu_nvidia_smi` | **Q3** — driver version, CUDA version, how many H100s and their memory |
| `06_import_checks` | **Q1** — which of torch/jax/polars/sentence-transformers actually import in this image, and the exact error for each that doesn't |
| `07_torch_deep_dive` | **Why** torch fails: which build is installed, whether it declares `intel-*`/SYCL dependencies, whether `libsycl*` exists anywhere on the system, and `ldd` output naming every missing shared object |
| `08_jax_devices` | How many GPUs **JAX** can drive — the input to the multi-GPU design |
| `09_torch_cuda` | Whether torch sees CUDA, how many devices, and whether it is an XPU build |
| `10_pip_environment` | **Q2** — the configured index, full package inventory of the image, whether `intel-sycl-rt` / CUDA-variant torch resolve, and network reachability |
| `11_input_port_layout` | **Q4** — is a port a file or a directory, how many parts, total size, and the true format of extension-less blobs (magic bytes) |
| `12_environment_variables` | `OMP_NUM_THREADS`, `CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH` and the rest of the relevant environment |
| `SUMMARY / VERDICT` | One-screen recap: broken imports, GPU counts, effective cores, missing shared objects |

## Parameters (all optional, sensible defaults)

| Parameter | Default | Purpose |
|---|---|---|
| `model_store_dir` | `/mnt/data/laim/models` | Directory to test for write access |
| `probe_pip_packages` | `torch,intel-sycl-rt,intel-cmplr-lib-rt,jax,polars,sentence-transformers` | Packages to test-resolve against the configured index |
| `check_network` | `true` | TCP-connect test to pypi.org / download.pytorch.org / the configured index |
| `deep_ldd` | `true` | Run `ldd` on torch's native extension to name the missing `.so` exactly |

## Local dry run

```bash
python probe.py                      # image/GPU/limits only
python probe.py /path/to/port_dir    # also inspects a simulated port payload
```
