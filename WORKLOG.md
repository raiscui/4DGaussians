# WORKLOG

> 只追加,不在中间插入.

## 2026-02-21T06:26:20+00:00

- 新任务开始: 将 notebook 依赖安装迁移到 Pixi.

## 2026-02-21T06:26:20+00:00 完成情况

- 新增 `pixi.toml`,把 notebook/脚本运行所需依赖收敛到 Pixi manifest,并把两个 CUDA 子模块作为 editable PyPI 依赖纳入安装流程.
- notebook 改造:
  - `4DGaussians.ipynb`: 用 `pixi install` 替代 pip 安装,并把所有 `!python ...` 统一改为 `!pixi run python ...`.
  - `4DGaussians_rais.ipynb`: 同上,另外把 submodule 重装步骤改为 `!pixi run python -m pip install -e ...` 以便在补丁后重编译扩展.
- 移除 `mmcv` 重依赖:
  - `utils/params_utils.py` 增加 `load_config_file()` 支持 `_base_` 继承与 dict 合并.
  - `train.py`,`render.py`,`export_perframe_3DGS.py`,`merge_many_4dgs.py` 改用轻量配置加载器.
  - `requirements.txt` 移除 `mmcv` 与 `argparse`.
- 文档与忽略规则:
  - `README.md` 增加 Pixi 安装与运行示例.
  - `.gitignore` 增加忽略 `.pixi/`.

- 追加修复: Pixi DNS 解析失败导致无法安装
  - 新增 `pixi.mirrors.toml`,提供 conda channel 镜像重定向.
  - 两个 notebook 的安装命令增加回退: `pixi install` 失败则自动改用 `pixi install --config pixi.mirrors.toml`.
  - `README.md` 增加 DNS 报错时的替代命令说明.

## 2026-02-21T08:13:26+00:00 追加修复: 版本组合与 Pixi 求解稳定性

- 切换 PyTorch 版本组合:
  - conda `pytorch` channel 无 `pytorch 2.6`/`pytorch-cuda 12.6`,改为 PyPI cu126 wheel(`torch~=2.6.0`,`torchvision~=0.21.0`).
  - `pixi.toml` 默认 Python 升级到 `3.12.*`.
- 修复错误用法:
  - `pixi install` 不支持 `--config`,镜像配置改为写入 `.pixi/config.toml`(notebook 已更新回退逻辑).
- 解决 GitHub DNS 依赖:
  - 新增 `conda_pypi_map.json`,并在 `pixi.toml` 里用 `conda-pypi-map` 指向本地映射,避免访问 `raw.githubusercontent.com`.
- 调整 CUDA 扩展安装方式:
  - 不再把两个子模块作为 pypi path 依赖参与 lock(避免 metadata 阶段因 `import torch` 失败).
  - 新增 Pixi task: `pixi run install-ext`,并在 notebook 安装步骤中自动执行.

## 2026-02-21T08:31:00+00:00 追加修复: Pixi editable 扩展安装与 `simple_knn` 导入

- 修复 `pixi run install-ext` 报 `ModuleNotFoundError: torch`:
  - `pixi.toml` 的 `install-ext` task 增加 `--no-build-isolation`,让 pip 构建过程直接复用 Pixi 环境里的 torch.
- 修复 `simple_knn` 安装成功但无法导入:
  - 新增 `submodules/simple-knn/simple_knn/__init__.py`,确保 `from simple_knn._C import distCUDA2` 在运行期可用.
- 自检通过:
  - `pixi run python -c "import torch; print(torch.__version__, torch.version.cuda)"`
  - `pixi run install-ext`
  - `pixi run python -c "import diff_gaussian_rasterization; from simple_knn._C import distCUDA2; print('ok')"`

## 2026-02-21T09:40:00+00:00 追加: 从多机位视频生成 MultipleView 数据集

- 新增全流程脚本: `scripts/preprocess_multipleview_from_videos.py`
  - 输入: 多路视频目录 + dataset name
  - 输出: `data/multipleview/<dataset>/` 下的 `camXX/frame_*.jpg`, `sparse_`, `points3D_multipleview.ply`, `poses_bounds_multipleview.npy`
- Pixi task 增加入口: `pixi run prep-multipleview ...`(`pixi.toml:61`)
- 兼容 headless COLMAP:
  - 自动设置 `QT_QPA_PLATFORM=offscreen`
  - 显式禁用 SIFT GPU: `--SiftExtraction.use_gpu 0`, `--SiftMatching.use_gpu 0`
- 小样本验证通过:
  - 使用 `/cloud/cloud-s3fs/SelfCap/bar-release/videos` 前 2 路视频生成: `data/multipleview/bar-release_mv_test/`
  - `readMultipleViewinfos("data/multipleview/bar-release_mv_test")` 可正常读取(2 cams * 20 frames => train 40, test 6)

## 2026-02-21T10:00:00+00:00 追加: README 手册化与对标 FreeTimeGsVanilla 的体量建议

- README 增加 MultipleView(mp4) 一键生成说明与参数手册:
  - `README.md` 新增 "Generate MultipleView data from multi-camera videos (mp4)" 小节
  - 包含 quick sanity run / full run / flags 解释 / verify 命令
- 给出对标 FreeTimeGsVanilla 默认 demo 体量([0,61) 连续帧)的参数建议:
  - 帧数对齐: `--fps 60 --max-frames 61`
  - 分辨率对齐: bar-release(最长边约 3760)上,
    - `DATA_FACTOR=4` 约等价 `--max-size 960`
    - `DATA_FACTOR=8` 约等价 `--max-size 480`

## 2026-02-21T10:10:00+00:00 更正: bar-release 的 `--max-size` 示例参数

- FreeTimeGsVanilla 的 `cfg.yml` 显示 `data_factor: 4`,而 bar-release 原视频最长边约 3760,
  所以 1/4 下采样更接近 `3760/4=940`,不是 960(960 更像是 3840/4 的标准 4K 情况).
- 已更新 `README.md` 的 bar-release 示例命令,将 `--max-size` 从 960 更正为 940.
