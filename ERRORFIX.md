# ERRORFIX

> 只追加,不在中间插入.

## 2026-02-21T06:26:20+00:00

### 问题

执行 `pixi install` 时失败,报错:

- `error sending request for url (https://conda.anaconda.org/conda-forge/linux-64/repodata_shards.msgpack.zst)`
- `dns error: failed to lookup address information: Name does not resolve`

### 原因

网络层 DNS 无法解析 `conda.anaconda.org`,导致 Pixi 在下载 conda channel 的 repodata 阶段直接失败.
这不是依赖冲突,也不是 `pixi.toml` 写法错误.

### 修复

- 增加 `pixi.mirrors.toml`,用 Pixi 的 `--config` 镜像配置把:
  - `https://conda.anaconda.org/conda-forge`
  - `https://conda.anaconda.org/pytorch`
  - `https://conda.anaconda.org/nvidia`
  重定向到多个常用镜像站(并保留官方作为最后兜底).
- notebook 的安装命令改成:
  - 先 `pixi install`
  - 失败再 `pixi install --config pixi.mirrors.toml`

### 验证方式

在出现 DNS 报错的环境里执行:

- `pixi install --config pixi.mirrors.toml`

预期:

- 能正常下载 repodata 并完成求解与安装.

## 2026-02-21T08:13:26+00:00

### 更正

上一条记录里提到的 `pixi install --config ...` 在 Pixi 0.63.2 中是无效参数.

### 修复(镜像配置的正确用法)

- Pixi 会读取项目本地配置: `.pixi/config.toml`
- 启用镜像方式:
  - `mkdir -p .pixi`
  - `cp -f pixi.mirrors.toml .pixi/config.toml`
  - `pixi install`
- notebook 已更新为自动回退:
  - 先尝试 `pixi install`
  - 失败则写入 `.pixi/config.toml` 后重试

### 新问题: conda 版本组合不可用

用户目标组合: `python 3.12` + `cuda 12.6` + `pytorch 2.6`.

在 conda 的 `pytorch` channel 内:

- `pytorch` 最新为 `2.5.x`
- `pytorch-cuda` 最新为 `12.4`

因此 conda 无法满足 `pytorch 2.6` 或 `pytorch-cuda 12.6`.

### 修复(使用 PyPI cu126 wheel)

- `pixi.toml` 改为:
  - conda 只保留通用依赖(来自 conda-forge)
  - PyTorch 使用 PyPI 官方 CUDA wheel:
    - `torch ~=2.6.0`
    - `torchvision ~=0.21.0`
    - index: `https://download.pytorch.org/whl/cu126`

### 新问题: GitHub DNS 导致 conda-pypi mapping 拉取失败

- Pixi 在 conda+PyPI 混合依赖下,默认会访问 GitHub 拉取映射文件.
- 部分网络环境下 `raw.githubusercontent.com` 也可能 DNS 失败,导致 `pixi lock/install` 失败.

### 修复(本地 mapping)

- 新增 `conda_pypi_map.json`
- 在 `pixi.toml` 增加 `conda-pypi-map = { conda-forge = "conda_pypi_map.json" }`,避免访问 GitHub

### 新问题: 本地 CUDA 扩展作为 path 依赖时,lock 阶段 metadata 生成失败

- 现象: uv 在 lock 阶段执行子模块 `setup.py`,因为顶层 `import torch` 而报 `ModuleNotFoundError: torch`
- 解决: 不再把两个子模块作为 pypi path 依赖参与求解
  - 改为 Pixi task: `pixi run install-ext`
  - 在 notebook 的安装步骤中,`pixi install` 后自动执行 `pixi run install-ext`

## 2026-02-21T08:31:00+00:00

### 问题1: `pixi run install-ext` 构建 editable 时找不到 torch

执行:

- `pixi run install-ext`

报错(摘要):

- `ModuleNotFoundError: No module named 'torch'`
- 出现场景: pip "Getting requirements to build editable"

### 原因

- pip 默认启用 build isolation,会创建临时构建环境(`/tmp/pip-build-env-*/overlay/...`).
- 两个 CUDA 子模块的 `setup.py` 顶层直接 `import torch`,
  导致隔离环境里没有 torch 时立刻失败.

### 修复

- `pixi.toml` 的 `install-ext` task 增加 `--no-build-isolation`,复用 Pixi 环境里的 torch:
  - `python -m pip install --no-build-isolation -e ...`

### 验证

- `pixi run python -c "import torch; print(torch.__version__)"` 能正常输出版本.
- `pixi run install-ext` 能成功编译并安装:
  - `diff_gaussian_rasterization`
  - `simple_knn`

### 问题2: `simple_knn` 安装成功但运行期无法导入

现象:

- pip 显示 `Successfully installed simple_knn-0.0.0`,但运行:
  - `from simple_knn._C import distCUDA2`
  仍报 `ModuleNotFoundError: No module named 'simple_knn'`.

原因:

- `submodules/simple-knn/simple_knn/` 缺少 `__init__.py`,
  在 PEP660 editable 的 import hook 机制下会导致顶层包无法生成 module spec.

修复:

- 新增 `submodules/simple-knn/simple_knn/__init__.py`,确保 `simple_knn` 包可被导入.

验证:

- `pixi run python -c "from simple_knn._C import distCUDA2; print(distCUDA2)"` 运行成功.

## 2026-02-21T09:40:00+00:00

### 问题: COLMAP 在 headless 环境崩溃(无 display / 无 OpenGL context)

在容器或远程机器(没有图形界面)执行 `colmap feature_extractor` / `colmap exhaustive_matcher` 时报错并中止:

1. Qt display 问题:
   - `qt.qpa.xcb: could not connect to display`
2. OpenGL context 问题:
   - `opengl_utils.cc: Check failed: context_.create()`

### 原因

- 部分发行版的 COLMAP CLI 即使走命令行子命令,仍会初始化 Qt 平台插件.
  没有 display 时会直接 `SIGABRT`.
- `exhaustive_matcher` 默认可能走 SiftGPU/OpenGL,在无 OpenGL context 的环境里也会 `SIGABRT`.

### 修复

在调用 COLMAP 时强制 headless + CPU 路径:

- 设置环境变量:
  - `QT_QPA_PLATFORM=offscreen`
- 显式禁用 GPU:
  - feature_extractor: `--SiftExtraction.use_gpu 0`
  - matcher: `--SiftMatching.use_gpu 0`

已在 `scripts/preprocess_multipleview_from_videos.py` 的命令封装中落地.

### 验证

在无 display 的环境里执行:

- `pixi run prep-multipleview --videos-dir /cloud/.../videos --dataset-name bar-release_mv_test --limit-cams 2 --max-frames 20 --pointcloud sparse`

预期:

- COLMAP 不再崩溃,能生成 `sparse_/` 与 `poses_bounds_multipleview.npy`.

## 2026-02-21T13:43:41+00:00

### 问题: `colmap feature_extractor` 被 SIGKILL(-9) 强杀

在执行:

- `scripts/preprocess_multipleview_from_videos.py`

的 COLMAP 阶段时失败,报错摘要类似:

- `RuntimeError: 命令失败(returncode=-9): colmap feature_extractor ...`

### 原因

- `returncode=-9` 表示进程被 `SIGKILL` 终止.
  - 最常见原因是系统/容器 OOM killer(内存不足)直接杀死进程.
- 脚本原先硬编码了一组比 COLMAP 默认更激进的 SIFT 参数,会显著放大内存峰值:
  - `max_image_size=4096`
  - `max_num_features=16384`
  - 开启 affine/dsp
- 同时未限制 COLMAP 线程数,默认 `num_threads=-1` 会吃满 CPU 核数,进一步放大峰值内存.

### 修复

- `scripts/preprocess_multipleview_from_videos.py` 增加 COLMAP 调参开关,并把默认值调整为更省内存:
  - `--colmap-num-threads`(默认 4)
  - `--colmap-sift-max-image-size`(默认 3200)
  - `--colmap-sift-max-num-features`(默认 8192)
  - `--colmap-sift-affine` / `--colmap-sift-dsp`(默认关闭)
- `_run_cmd()` 在遇到负 returncode(信号终止)时输出更明确的提示;遇到 SIGKILL 会提示 OOM 并给出降参建议.
- `--keep-colmap-tmp` 行为增强: 即使中途失败也会尽量保留 `_colmap_tmp`,便于排查.

### 验证

1. 帮助信息包含新增参数:
   - `python3 scripts/preprocess_multipleview_from_videos.py --help`
2. 若仍遇到 `returncode=-9`,优先尝试更保守参数:
   - `--colmap-num-threads 1 --colmap-sift-max-image-size 1600 --colmap-sift-max-num-features 4096`
