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

## 2026-02-21T12:35:44+00:00 追加: 为公平评估对比,对齐 FreeTimeGsVanilla 的 data_factor 语义

- MultipleView 训练侧下采样(等价 FreeTime `data_factor`):
  - `scene/multipleview_dataset.py` 增加 `downsample_factor`,并用 floor 下采样图片,同时把 focal 按相同 factor 缩放,保证 FOV 语义不变.
  - `scene/dataset_readers.py` 的 `readMultipleViewinfos(..., resolution=...)` 支持把 `resolution` 当作 downsample factor.
  - `scene/__init__.py` 在 MultipleView 分支把 `args.resolution` 传入,因此 `train.py --resolution 4/8` 对 MultipleView 生效.
- MultipleView 数据生成按帧索引对齐:
  - `scripts/preprocess_multipleview_from_videos.py` 增加 `--frame-start/--frame-end/--frame-step`,
    使用 ffmpeg `select` 精确实现 `[start_frame,end_frame)` 语义,并在该模式下加 `-vsync 0` 避免补帧/重复.
- 文档同步:
  - `README.md` 增加 "Fair comparison with FreeTimeGsVanilla" 小节,明确推荐:
    - 生成阶段 `--max-size 0` 保留原始帧质量.
    - 训练阶段用 `--resolution 4/8` 做等价 data_factor 的下采样.

- 自检(本地):
  - `pixi run python -m py_compile scene/multipleview_dataset.py scene/dataset_readers.py scene/__init__.py scripts/preprocess_multipleview_from_videos.py`
  - `pixi run python -c "from scene.dataset_readers import readMultipleViewinfos; s=readMultipleViewinfos('data/multipleview/bar-release_mv_test'); print('ok', len(s.train_cameras), len(s.test_cameras))"` => `ok 40 6`
  - `pixi run python -c "from scene.dataset_readers import readMultipleViewinfos; s=readMultipleViewinfos('data/multipleview/bar-release_mv_test', resolution=4); img,_,_=s.train_cameras[0]; print(img.shape)"` => `torch.Size([3, 240, 134])`

## 2026-02-21T13:09:14+00:00 追加: README 记录对比命令(生成 + 训练)

- `README.md` 的 "Fair comparison with FreeTimeGsVanilla" 小节补充了训练命令示例:
  - `pixi run prep-multipleview ... --dataset-name bar-release_fullres_0_61 ... --max-size 0`
  - `pixi run train -s data/multipleview/bar-release_fullres_0_61 --configs arguments/multipleview/xxx.py --resolution 4`

## 2026-02-21T13:43:41+00:00 追加: COLMAP feature_extractor 被 SIGKILL(-9) 的稳态修复

- 问题现象: 运行 `scripts/preprocess_multipleview_from_videos.py` 时,`colmap feature_extractor` 返回 `-9`(SIGKILL),脚本报 `RuntimeError: returncode=-9`.
- 修复与改良:
  - `scripts/preprocess_multipleview_from_videos.py` 暴露 COLMAP 关键 SIFT 参数与线程数为 CLI 参数:
    - `--colmap-num-threads`
    - `--colmap-sift-max-image-size`
    - `--colmap-sift-max-num-features`
    - `--colmap-sift-affine` / `--colmap-sift-dsp`
  - 默认值回归到更接近 COLMAP 默认(更省内存),并默认关闭 affine/dsp.
  - `_run_cmd()` 在检测到信号终止(负 returncode,尤其 SIGKILL)时,给出更明确的 OOM 调参提示.
  - `--keep-colmap-tmp` 行为增强: 即使 COLMAP 中途失败也会尽量把临时目录拷到 `data/multipleview/<dataset>/_colmap_tmp/`,方便排查.
- 文档同步:
  - `README.md` 的 MultipleView 生成参数说明补充了上述 COLMAP 调参开关与 OOM 建议.

## 2026-02-21T17:04:40+00:00 追加: 准备提交并推送到 `raiscui/4DGaussians`

- 推送目标: `https://github.com/raiscui/4DGaussians.git`
- 卫生检查:
  - `data.zip` 为本地大文件,将加入 `.gitignore`,避免误提交.
  - `.envrc.private` 已在 `.gitignore` 中忽略,不会被提交.
  - `.envrc` 仅做环境变量引用与开发辅助配置,不包含 token 实值,计划纳入提交.

## 2026-02-21T17:52:38+00:00 完成: 已提交并推送到 `raiscui/4DGaussians`

- 已将远端 `origin` 设置为 `https://github.com/raiscui/4DGaussians.git`,并 push `master`.
- 主提交为 `a889dec`(Improve COLMAP preprocessing robustness).
- `.envrc` 做了安全性收尾:
  - 明确 `.envrc` 为可提交的公开配置,敏感信息只允许放在 `.envrc.private`(gitignore).
  - 在检测到 `GITHUB_TOKEN` 时,自动生成 `.direnv/git-askpass.sh` 用于 https 非交互 push(目录已 gitignore).

## 2026-02-21T18:07:39+00:00 追加: 修复 MultipleView 训练示例的 `--configs` 占位符

- 背景: README 示例里写了 `--configs arguments/multipleview/xxx.py`,但仓库内并不存在该文件,容易误导使用者.
- 修复:
  - `README.md` 把占位符改为实际存在的 `arguments/multipleview/default.py`.
  - `README.md` 的 Training 小节补充说明: 如需按数据集调参,复制/继承 `default.py` 生成 `arguments/multipleview/<dataset>.py`,再把 `--configs` 指向它.
- 备注: `train.py` 会通过 `utils/params_utils.load_config_file()` 加载该文件,支持 `_base_` 继承与 dict 深度合并.

## 2026-02-22T07:47:00+00:00 追加: 修复 MultipleView 渲染 test mp4 "视角乱跳"

- 根因定位:
  - 旧实现的 MultipleView test split 是"每个相机只抽 3 帧",并把多相机视角帧直接串成一条 `output/test/.../video_rgb.mp4`.
  - 对多相机数据来说,这会造成肉眼观感的"视角乱跳"(每 3 帧切一次机位).
- 改良: MultipleView train/test 按相机切分(camera hold-out)
  - `scene/dataset_readers.py` 的 `readMultipleViewinfos()` 现在会按 `llffhold` 在相机维度做划分:
    - 解析 COLMAP 的 `imageN.jpg` => cam_id=N
    - cam_id 排序后,idx % llffhold == 0 的相机进入 test,其余进入 train
  - `scene/multipleview_dataset.py` 支持 `cam_ids` 过滤,并为 train/test 都保留完整时间序列(不再对 test 只取 3 帧).
- 改良: MultipleView 的 test mp4 按相机输出,默认 video_rgb.mp4 指向 cam01
  - `render.py` 在 `cam_type=="MultipleView" && name=="test"` 时:
    - `video_rgb.mp4`: 输出 primary 相机(最小 cam_id,通常是 cam01)的完整时间序列,便于直接观看.
    - `video_rgb_camXX.mp4`: 每路 hold-out 相机各输出一条 mp4.
    - `video_rgb_allcams.mp4`: 保留一个把所有 test 帧按 dataset 顺序拼接的对照版本(可能会跳视角).
- 自检:
  - `pixi run python -c "from scene.dataset_readers import readMultipleViewinfos; s=readMultipleViewinfos('data/multipleview/bar-release_fullres_0_61', llffhold=8, resolution=4); print(len(s.train_cameras), len(s.test_cameras), len(s.video_cameras))"` => `915 183 300`
  - `pixi run python render.py --model_path output --iteration 30000 --skip_train --skip_video --configs arguments/multipleview/default.py`
    - `output/test/ours_30000/renders` 变为 183 张(3 cams * 61 frames)
    - `output/test/ours_30000/video_rgb_cam01.mp4`,`video_rgb_cam09.mp4`,`video_rgb_cam17.mp4` 均生成

## 2026-02-22T07:59:50+00:00 追加: 消除 imageio 写 mp4 的 macro_block_size resize warning

- 背景:
  - MultipleView 在 `--resolution 4` 下的渲染尺寸是 `527x940`(floor 下采样).
  - imageio/ffmpeg writer 会因为尺寸不能被 16 整除而自动 resize 到 `528x944`,并打印 warning.
- 修复:
  - `render.py` 在写 mp4 前对每帧做 edge padding,补齐到 16 的倍数.
  - 这样不会发生缩放插值,也不会再刷 warning.
- 自检:
  - 渲染 test:
    - `pixi run python render.py --model_path output --iteration 30000 --skip_train --skip_video --configs arguments/multipleview/default.py`
    - 预期: 不再出现 `macro_block_size=16` warning.
  - 渲染 spiral video:
    - `pixi run python render.py --model_path output --iteration 30000 --skip_train --skip_test --configs arguments/multipleview/default.py`
    - 预期: 同上.

## 2026-02-22T08:44:10+00:00 追加: 对比 4DGaussians vs FreeTimeGsVanilla 产物体积差异

- 解释用户疑问: 本项目 `output/point_cloud/iteration_30000/` 体积小,主要因为它保存的是"用于渲染的高斯点云(Ply) + deformation 网络权重",默认不保存 optimizer state.
- 量化本项目产物:
  - `point_cloud.ply` 头部显示 `element vertex 119497`,property 数 62,因此数据区大小约 `119497*62*4=28.26MB`,与实际文件大小一致.
  - `deformation.pth` 约 9.5MB,`deformation_accum.pth` 约 1.37MB,`deformation_table.pth` 约 0.11MB.
- 量化 FreeTime checkpoint:
  - `ckpt_29999.pt` 为 978MB,顶层含 `splats`(模型参数)与 `optimizers`(Adam 动量).
  - `splats` 总张量约 326MB,其中 `shN` 占 229MB.
  - `optimizers` 总张量约 652MB,其中 `optimizers.shN` 约 458MB(=2*shN,对应 `exp_avg/exp_avg_sq`).
  - FreeTime 的高斯数量约 133 万,也显著大于本项目示例的约 12 万.
- 对齐建议:
  - 比"推理模型体积"时: FreeTime 只看 `splats`,本项目看 `point_cloud.ply` + `deformation.pth`.
  - 比"可继续训练的 checkpoint"时: 本项目需启用 `--checkpoint_iterations` 保存 `chkpnt_*.pth`(含 optimizer state).

## 2026-02-22T08:48:56+00:00 追加: MultipleView video 镜头在中心视角停留更久

- 需求: 用户观看 `output/video/ours_30000/video_rgb.mp4` 时希望镜头更稳,在中心视角附近停留更久.
- 定位: MultipleView 的 video(spiral) 轨迹由 `scene/multipleview_dataset.py:get_video_cam_infos()` 读取 `poses_bounds_multipleview.npy` 并调用 `get_spiral()` 生成.
- 实现: 增加可配置的 spiral 轨迹参数,并在起始 pose 处增加停留帧:
  - `scene/neural_3D_dataset_NDC.py:get_spiral()` 新增 `N_rots`/`zrate` 参数.
  - `scene/multipleview_dataset.py` 支持:
    - `video_spiral_hold_start`: 起始视角重复 N 帧(相机不动,但 time 按总帧数推进).
    - `video_spiral_n_rots`/`video_spiral_rads_scale`/`video_n_views`: 控制绕圈次数/半径/采样帧数.
  - `scene/__init__.py` 把上述参数从 args 透传到 `readMultipleViewinfos()`,并用 `getattr` 兼容旧 `output/cfg_args`.
  - `utils/params_utils.py:merge_hparams()` 允许 merge 配置时补齐旧 cfg_args 缺失字段.
- 默认配置:
  - `arguments/multipleview/default.py` 增加 `ModelParams`:
    - `video_spiral_hold_start=60`
    - `video_spiral_n_rots=1`
- 复现/验证方式:
  - 直接重跑用户命令即可:
    - `pixi run python render.py --model_path \"output\" --skip_train --configs arguments/multipleview/default.py`
  - 若想更久停留/更慢绕圈,改 `arguments/multipleview/default.py` 里的 `ModelParams` 对应值.

## 2026-02-22T09:43:07+00:00 追加: MultipleView video 时间维度改为 loop(动作持续运动)

- 背景: 用户指出 “有效帧 x < video 帧数 N 时应该 loop,而不是用 hold 把相机固定住”.
- 根因: 若 time 用 `i/N` 均匀推进,动作会被拉慢约 `N/x` 倍,导致主观上“该动的没动”.
- 修复: 增加 `ModelParams.video_time_mode`:
  - `linear`(旧): `time=i/N`
  - `loop`(新): `time=(i % x)/x`,其中 x 通过统计 `data/.../cam01` 的 `frame_*.jpg` 得到.
- 默认策略调整:
  - `arguments/multipleview/default.py` 默认 `video_time_mode=\"loop\"`
  - `video_spiral_hold_start` 默认回退为 0
  - `video_spiral_n_rots=1` 保持更慢更稳
- 快速验证(本地):
  - `readMultipleViewinfos(..., video_time_mode='loop')` 下,`time[61]==0.0` 表明按 61 帧循环.

## 2026-02-22T09:17:41+00:00 追加: README 补充"点数增长机制"与"checkpoint 体积差异"说明

- 更新 `README.md` 在 MultipleView 的 "Fair comparison with FreeTimeGsVanilla" 段落后追加 FAQ:
  - 解释"为什么训练出来的高斯点数只有 12 万左右":
    - 初始点云来自 COLMAP sparse,常见较小.
    - `densify_until_iter` 决定 densify/prune 的时间窗口.
    - `train.py` 里存在 `gaussians.get_xyz.shape[0] < 360000` 的 densify 硬上限.
    - 点数较大时 prune 会抑制增长.
  - 解释"为什么本项目 iteration_x 快照比 FreeTime 的 ckpt_*.pt 小":
    - `output/point_cloud/iteration_x/` 是渲染/评估快照,默认不含 optimizer state.
    - `chkpnt_*_x.pth` 是可继续训练的完整 checkpoint,需用 `--checkpoint_iterations` 显式开启保存.
    - FreeTime checkpoint 通常同时包含模型参数与 Adam optimizer state,因此体积远大.
  - 补充 PyTorch >= 2.6 的 `torch.load(weights_only=...)` 兼容提醒与安全注意事项.
