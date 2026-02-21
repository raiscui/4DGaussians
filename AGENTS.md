# Repository Guidelines

## proxy
- 运行网络命令前要注册代理: `export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897`

## Project Structure & Module Organization

- `train.py`, `render.py`, `metrics.py`: main entry points for training, rendering, and evaluation.
- `arguments/`: experiment configs (dataset + hyperparameters), e.g. `arguments/dnerf/bouncingballs.py`.
- `scene/`, `gaussian_renderer/`, `utils/`: core implementation (data loading, model, rendering, helpers).
- `scripts/`: preprocessing and helper scripts (DyNeRF/HyperNeRF tools, train `.sh` wrappers).
- `submodules/`: CUDA extensions (`simple-knn`, `depth-diff-gaussian-rasterization`) installed as editable packages.
- `assets/`, `docs/`, `*.ipynb`: figures, viewer docs, and Colab/notebook demos.

## Build, Test, and Development Commands

Environment setup (see `README.md` for details):

```bash
git submodule update --init --recursive
conda create -n Gaussians4D python=3.7
conda activate Gaussians4D
pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

Common workflows:

- Train: `python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py`
- Render: `python render.py --model_path output/dnerf/bouncingballs --skip_train --configs arguments/dnerf/bouncingballs.py`
- Evaluate: `python metrics.py --model_path output/dnerf/bouncingballs`
- Viewer: follow `docs/viewer_usage.md`

## Coding Style & Naming Conventions

- Python: use 4-space indentation; keep changes consistent with existing code style.
- Naming: `snake_case` for functions/variables; `PascalCase` for classes.
- Configs: keep config filenames aligned with dataset/scene names under `arguments/<dataset>/`.

## Testing Guidelines

- No dedicated unit test suite is included.
- Before submitting changes, run a short train + render + metrics loop on a small scene and confirm outputs under `output/<expname>/` are produced and readable.

## Commit & Pull Request Guidelines

- Commit messages in history are short and imperative (e.g. "fix ...", "add ...", "Update README.md").
- PRs should include: what changed, how to reproduce/verify, and (when relevant) screenshots/videos or metric diffs.
- Do not commit large generated artifacts (e.g. `output/` training runs) unless explicitly required.

## Submodules & CUDA Notes

- If you modify anything under `submodules/`, update the git submodule pointer and ensure extensions rebuild (re-run the `pip install -e ...` steps).
- GPU/CUDA compatibility matters (the reference environment uses PyTorch 1.13.1 + CUDA 11.6 per `README.md`).
