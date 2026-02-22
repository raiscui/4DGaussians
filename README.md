# 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

## CVPR 2024

### [Project Page](https://guanjunwu.github.io/4dgs/index.html)| [arXiv Paper](https://arxiv.org/abs/2310.08528)

[Guanjun Wu](https://guanjunwu.github.io/) <sup>1*</sup>, [Taoran Yi](https://github.com/taoranyi) <sup>2*</sup>,
[Jiemin Fang](https://jaminfong.cn/) <sup>3‡</sup>, [Lingxi Xie](http://lingxixie.com/) <sup>3 </sup>, </br>[Xiaopeng Zhang](https://scholar.google.com/citations?user=Ud6aBAcAAAAJ&hl=zh-CN) <sup>3 </sup>, [Wei Wei](https://www.eric-weiwei.com/) <sup>1 </sup>,[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/) <sup>2 </sup>, [Qi Tian](https://www.qitian1987.com/) <sup>3 </sup> , [Xinggang Wang](https://xwcv.github.io) <sup>2‡✉</sup>

<sup>1 </sup>School of CS, HUST &emsp; <sup>2 </sup>School of EIC, HUST &emsp; <sup>3 </sup>Huawei Inc. &emsp;

<sup>\*</sup> Equal Contributions. <sup>$\ddagger$</sup> Project Lead. <sup>✉</sup> Corresponding Author.



![block](assets/teaserfig.jpg)
Our method converges very quickly and achieves real-time rendering speed.

New Colab demo:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wz0D5Y9egAlcxXy8YO9UmpQ9oH51R7OW?usp=sharing) (Thanks [Tasmay-Tibrewal
](https://github.com/Tasmay-Tibrewal))

Old Colab demo:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hustvl/4DGaussians/blob/master/4DGaussians.ipynb) (Thanks [camenduru](https://github.com/camenduru/4DGaussians-colab).)

Light Gaussian implementation: [This link](https://github.com/pablodawson/4DGaussians) (Thanks [pablodawson](https://github.com/pablodawson))


## News

2024.6.25: we clean the code and add an explanation of the parameters.

2024.3.25: Update guidance for hypernerf and dynerf dataset.

2024.03.04: We change the hyperparameters of the Neu3D dataset, corresponding to our paper.

2024.02.28: Update SIBR viewer guidance.

2024.02.27: Accepted by CVPR 2024. We delete some logging settings for debugging, the corrected training time is only **8 mins** (20 mins before) in D-NeRF datasets and **30 mins** (1 hour before) in HyperNeRF datasets. The rendering quality is not affected.

## Environmental Setups

Please follow the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.

```bash
git clone https://github.com/hustvl/4DGaussians
cd 4DGaussians
git submodule update --init --recursive
conda create -n Gaussians4D python=3.7 
conda activate Gaussians4D

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

In our environment, we use pytorch=1.13.1+cu116.

### Pixi (Alternative)

If you prefer a more reproducible, one-command environment setup (especially for notebooks),
you can use Pixi:

```bash
# Install pixi (Linux/macOS)
curl -fsSL https://pixi.sh/install.sh | sh

cd 4DGaussians
git submodule update --init --recursive

# Install all dependencies defined in pixi.toml
pixi install

# Build & install the two CUDA extensions used by this repo
pixi run install-ext

# If you hit DNS errors for conda.anaconda.org (Name does not resolve),
# you can enable the provided mirror config by writing it to Pixi's local config:
# mkdir -p .pixi
# cp -f pixi.mirrors.toml .pixi/config.toml
# pixi install

# Example run
pixi run python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py
```

## Data Preparation

**For synthetic scenes:**
The dataset provided in [D-NeRF](https://github.com/albertpumarola/D-NeRF) is used. You can download the dataset from [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0).

**For real dynamic scenes:**
The dataset provided in [HyperNeRF](https://github.com/google/hypernerf) is used. You can download scenes from [Hypernerf Dataset](https://github.com/google/hypernerf/releases/tag/v0.1) and organize them as [Nerfies](https://github.com/google/nerfies#datasets). 

Meanwhile, [Plenoptic Dataset](https://github.com/facebookresearch/Neural_3D_Video) could be downloaded from their official websites. To save the memory, you should extract the frames of each video and then organize your dataset as follows.

```
├── data
│   | dnerf 
│     ├── mutant
│     ├── standup 
│     ├── ...
│   | hypernerf
│     ├── interp
│     ├── misc
│     ├── virg
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│       ├── cam01
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```

**For multipleviews scenes:**
If you want to train your own dataset of multipleviews scenes, you can orginize your dataset as follows:

```
├── data
|   | multipleview
│     | (your dataset name) 
│   	  | cam01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | cam02
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
```
After that, you can use the  `multipleviewprogress.sh` we provided to generate related data of poses and pointcloud.You can use it as follows:
```bash
bash multipleviewprogress.sh (youe dataset name)
```
You need to ensure that the data folder is organized as follows after running multipleviewprogress.sh:
```
├── data
|   | multipleview
│     | (your dataset name) 
│   	  | cam01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | cam02
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
│   	  | sparse_
│     		  ├── cameras.bin
│     		  ├── images.bin
│     		  ├── ...
│   	  | points3D_multipleview.ply
│   	  | poses_bounds_multipleview.npy
```

### (Recommended) Generate MultipleView data from multi-camera videos (mp4)

如果你的原始输入是"每路相机一个 mp4"(例如一个目录下有 `02.mp4`, `03.mp4`, ...),
推荐使用我们提供的全流程脚本,一键完成:

- 抽帧到 `camXX/frame_*.jpg`
- 用每路相机的首帧跑一次 COLMAP,得到静态多相机内外参(`sparse_`)
- 生成 `poses_bounds_multipleview.npy`(LLFF 兼容格式)
- 生成 `points3D_multipleview.ply`(并下采样到 <= 40000 点)

该脚本已经针对 headless 服务器环境做过兼容:
- 自动设置 `QT_QPA_PLATFORM=offscreen`
- 强制禁用 SIFT GPU,避免无 OpenGL context 时崩溃

#### Prerequisites

- `pixi install` 并完成 `pixi run install-ext`
- 系统安装 `ffmpeg` 与 `colmap`

#### Quick sanity run (2 cameras, 20 frames each)

```bash
pixi run prep-multipleview \
  --videos-dir /cloud/cloud-s3fs/SelfCap/bar-release/videos \
  --dataset-name bar-release_mv_test \
  --limit-cams 2 \
  --fps 2 \
  --max-size 940 \
  --max-frames 20 \
  --overwrite \
  --pointcloud sparse \
  --keep-colmap-tmp
```

#### Full run (all cameras)

```bash
pixi run prep-multipleview \
  --videos-dir /cloud/cloud-s3fs/SelfCap/bar-release/videos \
  --dataset-name bar-release \
  --fps 2 \
  --max-size 940
```

#### Fair comparison with FreeTimeGsVanilla (recommended for metrics)

FreeTimeGsVanilla 的 `data_factor` 是"训练侧下采样",并不会把落盘帧文件缩小.
为了保证输入数据质量对等,建议:

1. 生成阶段保留原始分辨率: `--max-size 0`
2. 用 `--frame-start/--frame-end` 精确对齐帧段语义(例如 `[0,61)`).
3. 训练阶段再用 `train.py --resolution 4/8` 做等价 data_factor 下采样.

```bash
pixi run prep-multipleview \
  --videos-dir /cloud/cloud-s3fs/SelfCap/bar-release/videos \
  --dataset-name bar-release_fullres_0_61 \
  --frame-start 0 \
  --frame-end 61 \
  --frame-step 1 \
  --max-size 0
```

然后用 `--resolution 4` 启动训练(等价 `data_factor=4`):

```bash
pixi run train \
  -s data/multipleview/bar-release_fullres_0_61 \
  --configs arguments/multipleview/default.py \
  --resolution 4
```

说明: 仓库内提供的 MultipleView 基线配置是 `arguments/multipleview/default.py`.
如果你需要按数据集微调参数,可以复制/继承这个文件生成自己的 `<dataset>.py`,再把 `--configs` 指向它.

#### FAQ: 为什么训练出来的高斯点数只有 12 万左右? 如何增大点数?

很多人会用 `output/point_cloud/iteration_x/point_cloud.ply` 的 vertex 数量来理解"模型里有多少个高斯".
例如你可能会看到类似 `element vertex 119497`(约 12 万).

这通常不是数据集"限制死"的,而是由训练过程中的 densification/pruning 共同决定:

1. 初始点云通常很小:
   - MultipleView 默认使用 COLMAP sparse 的 `points3D_multipleview.ply` 作为初始化点云.
   - 这份点云常见只有几千到几万点,后续主要靠 densification 扩增.
2. densification 有"时间窗口":
   - 训练循环里只有在 `iteration < densify_until_iter` 时才会 densify/prune(见 `train.py`).
   - MultipleView 基线配置 `arguments/multipleview/default.py` 默认 `densify_until_iter=10_000`,
     因此点数往往在前 10k iter 左右就基本停止增长.
3. densification 还有"硬上限":
   - 训练循环里写死了一个上限: 只有当 `gaussians.get_xyz.shape[0] < 360000` 才会 densify(见 `train.py`).
   - 因此在不改代码的前提下,点数通常不会超过 36 万.
4. prune 会在点数较大时抑制增长:
   - 例如当点数 > 20 万时,训练循环会进入 prune 分支(见 `train.py`),
     一边 densify,一边按 opacity/屏幕大小等条件裁掉不重要的点.

如果你希望点数变大,推荐按目标分两档调整:

- 目标: 30 万以内(不改代码,改配置即可).
  - 把 `arguments/multipleview/default.py` 里的 `densify_until_iter` 提高到 20000 或 30000.
  - 预期结果: 点数增长更久,但显存与训练耗时也会增加.
- 目标: 接近 FreeTimeGsVanilla 那种 100 万级(需要改代码).
  - 需要同时:
    - 提高 `densify_until_iter`(配置).
    - 把 `train.py` 里 densify 的 `360000` 硬上限改大.
  - 注意: 点数越大,保存出来的 `point_cloud.ply`/checkpoint 也会按比例变大.

一个快速检查点数的方式(读取 PLY 头部):

```bash
python3 - <<'PY'
from pathlib import Path
path = Path("output/point_cloud/iteration_30000/point_cloud.ply")
with path.open("rb") as f:
    for _ in range(200):
        line = f.readline().decode("ascii", errors="ignore").strip()
        if line.startswith("element vertex "):
            print(line)
            break
PY
```

#### FAQ: 为什么本项目的 `output/point_cloud/iteration_x/` 比 FreeTimeGsVanilla 的 `ckpt_*.pt` 小很多?

这两者本质上不是同一种东西,对比时需要先对齐"保存内容":

- 本项目的 `output/point_cloud/iteration_x/` 更像是"渲染/评估用快照":
  - 主要包含 `point_cloud.ply`(高斯参数) + `deformation.pth`(deformation 网络权重)等.
  - 默认不保存 optimizer state,因此体积相对小.
- 本项目的"可继续训练的 checkpoint"是 `chkpnt_*_x.pth`:
  - 只有当你在训练时传入 `--checkpoint_iterations ...` 才会生成(见 `train.py` 的保存逻辑).
  - 该 checkpoint 会包含 optimizer state(用于 resume),体积会明显增大.
- FreeTimeGsVanilla 的 `ckpt_*.pt` 通常会同时保存:
  - 模型参数(例如 splats/SH 等).
  - optimizer state(例如 Adam 的 `exp_avg`/`exp_avg_sq`),因此体积非常大(常见接近"参数本体的 3 倍").

建议的公平对比方式:

- 比"推理/渲染模型大小": 只对比模型参数本体(不要把 optimizer state 算进去).
- 比"可继续训练的完整 checkpoint": 两边都保存 optimizer state,再比体积与恢复训练行为.

补充: 在 PyTorch >= 2.6 中,`torch.load()` 的 `weights_only` 默认值发生过变化.
如果你要加载含有 python 对象的旧 checkpoint,可能需要显式 `weights_only=False`.
请只对你信任来源的 checkpoint 这么做(因为 pickle 反序列化存在代码执行风险).

#### Important flags

- `--videos-dir`: 输入视频目录.脚本会按文件名排序,并映射为 `cam01`, `cam02`, ...
- `--dataset-name`: 输出到 `data/multipleview/<dataset-name>/`.
- `--fps`: 抽帧帧率.默认 `10`.
  - `--fps 60` + `--max-frames 61` 等价于取前 61 帧(约 1 秒@60fps),体量接近很多 demo pipeline.
  - `--fps 1` 更适合长视频做"低频采样"(例如 1fps 覆盖 60 秒).
- `--frame-start/--frame-end/--frame-step`: 按"帧索引"抽取,更贴近 FreeTimeGsVanilla 的语义.
  - 帧索引是 0-based.
  - 当你设置了这些参数之一时,脚本会启用 frame range 模式,并忽略 `--fps` 的重采样语义.
  - 例: `--frame-start 0 --frame-end 61 --frame-step 1` 等价于取 `[0,61)` 连续 61 帧.
- `--max-size`: 抽帧时对图片做缩放,控制最长边(保持比例).默认 `1920`.
  - 重要: 这是"生成阶段"的不可逆缩放,会损失输入质量.
  - 若你关心最终模型评估对比的公平性,建议 `--max-size 0` 保留原始帧,
    并在训练时用 `train.py --resolution 4/8` 做等价 data_factor 的"训练侧下采样".
  - 若只是想快速预览/试跑,再考虑用 `--max-size 940`(bar-release 约 1/4)之类的值降低开销.
- `--max-frames`: 每路相机最多抽多少帧(0 表示不限制).
- `--pointcloud`: 点云来源:
  - `sparse`(默认): 直接用 COLMAP sparse 的 points3D,速度快.
  - `dense`: 走 COLMAP MVS 生成 fused.ply,再下采样,更慢但更密.
- `--overwrite`: 允许覆盖已存在的 `camXX/frame_*.jpg`.
- `--keep-colmap-tmp`: 保留中间 COLMAP 工作目录到 `data/multipleview/<dataset>/_colmap_tmp/` 便于排查(即使中途失败也会尽量保留).
- `--colmap-num-threads`: COLMAP 线程数(同时用于特征提取与匹配).小内存机器建议设小一些,例如 `--colmap-num-threads 1`.
- `--colmap-sift-max-image-size`: SIFT 特征提取的最大边长(默认 3200).如果你的视频接近 4K 且遇到 OOM,可以先试 `--colmap-sift-max-image-size 2000` 或 `1600`.
- `--colmap-sift-max-num-features`: 每张图最多特征点数(默认 8192).遇到 OOM 可先试 `--colmap-sift-max-num-features 4096`.
- `--colmap-sift-affine` / `--colmap-sift-dsp`: 更鲁棒但更慢更吃内存,默认关闭.若你的多机位视角差异很大且匹配困难,再考虑打开.

#### Verify output

```bash
pixi run python -c "from scene.dataset_readers import readMultipleViewinfos; s=readMultipleViewinfos('data/multipleview/bar-release'); print('ok', len(s.train_cameras), len(s.test_cameras), len(s.video_cameras))"
```

说明:
- MultipleView 的 train/test 现在采用"按相机切分"(camera hold-out),而不是"每个相机抽几帧".
  - `len(train_cameras) = 训练相机数 * 每路相机帧数`
  - `len(test_cameras) = hold-out 相机数 * 每路相机帧数`
- `llffhold` 控制 hold-out 的相机选择(按 cam_id 排序后,idx % llffhold == 0 的相机进入 test).
  - 如果你只想 hold-out 一路相机用于渲染/观察,可以把 `--llffhold` 设得很大(例如 `9999`),这样通常只会把 `cam01` 分到 test.


## Training

For training synthetic scenes such as `bouncingballs`, run

```
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
```

For training dynerf scenes such as `cut_roasted_beef`, run
```python
# First, extract the frames of each video.
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
# Second, generate point clouds from input data.
bash colmap.sh data/dynerf/cut_roasted_beef llff
# Third, downsample the point clouds generated in the second step.
python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
# Finally, train.
python train.py -s data/dynerf/cut_roasted_beef --port 6017 --expname "dynerf/cut_roasted_beef" --configs arguments/dynerf/cut_roasted_beef.py 
```
For training hypernerf scenes such as `virg/broom`: Pregenerated point clouds by COLMAP are provided [here](https://drive.google.com/file/d/1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5/view). Just download them and put them in to correspond folder, and you can skip the former two steps. Also, you can run the commands directly.

```python
# First, computing dense point clouds by COLMAP
bash colmap.sh data/hypernerf/virg/broom2 hypernerf
# Second, downsample the point clouds generated in the first step. 
python scripts/downsample_point.py data/hypernerf/virg/broom2/colmap/dense/workspace/fused.ply data/hypernerf/virg/broom2/points3D_downsample2.ply
# Finally, train.
python train.py -s  data/hypernerf/virg/broom2/ --port 6017 --expname "hypernerf/broom2" --configs arguments/hypernerf/broom2.py 
```

For training multipleviews scenes, you can start with the baseline config `arguments/multipleview/default.py`.
If you want a per-dataset config, copy it to `arguments/multipleview/(your dataset name).py` (or inherit via `_base_`) and use that file path in `--configs`.
```python
# Quick start (recommended): use the baseline config shipped with this repo.
python train.py -s  data/multipleview/(your dataset name) --port 6017 --expname "multipleview/(your dataset name)" --configs arguments/multipleview/default.py

# If you created `arguments/multipleview/(your dataset name).py`, point `--configs` to it instead.
python train.py -s  data/multipleview/(your dataset name) --port 6017 --expname "multipleview/(your dataset name)" --configs arguments/multipleview/(your dataset name).py
```

MultipleView 的训练侧下采样(用于对齐 FreeTimeGsVanilla 的 `data_factor`)可以直接用 `--resolution`:

```python
# 等价 data_factor=4: 训练侧把图像与 focal 都按 4x 下采样(落盘原图不变)
python train.py -s  data/multipleview/(your dataset name) --port 6017 --expname "multipleview/(your dataset name)" --configs arguments/multipleview/(your dataset name).py --resolution 4
```


For your custom datasets, install nerfstudio and follow their [COLMAP](https://colmap.github.io/) pipeline. You should install COLMAP at first, then:

```python
pip install nerfstudio
# computing camera poses by colmap pipeline
ns-process-data images --data data/your-data --output-dir data/your-ns-data
cp -r data/your-ns-data/images data/your-ns-data/colmap/images
python train.py -s data/your-ns-data/colmap --port 6017 --expname "custom" --configs arguments/hypernerf/default.py 
```
You can customize your training config through the config files.

## Checkpoint

Also, you can train your model with checkpoint.

```python
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py --checkpoint_iterations 200 # change it.
```

Then load checkpoint with:

```python
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py --start_checkpoint "output/dnerf/bouncingballs/chkpnt_coarse_200.pth"
# finestage: --start_checkpoint "output/dnerf/bouncingballs/chkpnt_fine_200.pth"
```

## Rendering

Run the following script to render the images.

```
python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py 
```

MultipleView 补充说明:
- MultipleView 的 `test` 集合可能包含多路 hold-out 相机.
- 为了避免把不同视角硬拼成一条 mp4 导致"视角乱跳",render 会在 `output/test/ours_*/` 下额外输出:
  - `video_rgb.mp4`: 默认指向 `cam01` 的完整时间序列(便于直接观看).
  - `video_rgb_camXX.mp4`: 每路相机各一条 mp4.
  - `video_rgb_allcams.mp4`: 按 dataset 顺序把所有 test 帧拼成一条(仅用于对照,可能会跳视角).
- 如果你看到过 imageio 的 `macro_block_size=16` resize warning:
  - 这是因为 (H,W) 不是 16 的倍数时,ffmpeg writer 会为了兼容性自动 resize.
  - 本仓库现在会在写 mp4 前对帧做 edge padding 到 16 倍数,避免隐式 resize 与 warning.

## Evaluation

You can just run the following script to evaluate the model.

```
python metrics.py --model_path "output/dnerf/bouncingballs/" 
```


## Viewer
[Watch me](./docs/viewer_usage.md)
## Scripts

There are some helpful scripts, please feel free to use them.

`export_perframe_3DGS.py`:
get all 3D Gaussians point clouds at each timestamps.

usage:

```python
python export_perframe_3DGS.py --iteration 14000 --configs arguments/dnerf/lego.py --model_path output/dnerf/lego 
```

You will a set of 3D Gaussians are saved in `output/dnerf/lego/gaussian_pertimestamp`.

`weight_visualization.ipynb`:

visualize the weight of Multi-resolution HexPlane module.

`merge_many_4dgs.py`:
merge your trained 4dgs.
usage:

```python
export exp_name="dynerf"
python merge_many_4dgs.py --model_path output/$exp_name/sear_steak
```

`colmap.sh`:
generate point clouds from input data

```bash
bash colmap.sh data/hypernerf/virg/vrig-chicken hypernerf 
bash colmap.sh data/dynerf/sear_steak llff
```

**Blender** format seems doesn't work. Welcome to raise a pull request to fix it.

`downsample_point.py` :downsample generated point clouds by sfm.

```python
python scripts/downsample_point.py data/dynerf/sear_steak/colmap/dense/workspace/fused.ply data/dynerf/sear_steak/points3D_downsample2.ply
```

In my paper, I always use `colmap.sh` to generate dense point clouds and downsample it to less than 40000 points.

Here are some codes maybe useful but never adopted in my paper, you can also try it.

## Awesome Concurrent/Related Works

Welcome to also check out these awesome concurrent/related works, including but not limited to

[Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction](https://ingra14m.github.io/Deformable-Gaussians/)

[SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes](https://yihua7.github.io/SC-GS-web/)

[MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly Deformable Scenes](https://md-splatting.github.io/)

[4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency](https://vita-group.github.io/4DGen/)

[Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models](https://github.com/VITA-Group/Diffusion4D)

[DreamGaussian4D: Generative 4D Gaussian Splatting](https://github.com/jiawei-ren/dreamgaussian4d)

[EndoGaussian: Real-time Gaussian Splatting for Dynamic Endoscopic Scene Reconstruction](https://github.com/yifliu3/EndoGaussian)

[EndoGS: Deformable Endoscopic Tissues Reconstruction with Gaussian Splatting](https://github.com/HKU-MedAI/EndoGS)

[Endo-4DGS: Endoscopic Monocular Scene Reconstruction with 4D Gaussian Splatting](https://arxiv.org/abs/2401.16416)



## Contributions

**This project is still under development. Please feel free to raise issues or submit pull requests to contribute to our codebase.**


Some source code of ours is borrowed from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [K-planes](https://github.com/Giodiro/kplanes_nerfstudio), [HexPlane](https://github.com/Caoang327/HexPlane), [TiNeuVox](https://github.com/hustvl/TiNeuVox), [Depth-Rasterization](https://github.com/ingra14m/depth-diff-gaussian-rasterization). We sincerely appreciate the excellent works of these authors.

## Acknowledgement

We would like to express our sincere gratitude to [@zhouzhenghong-gt](https://github.com/zhouzhenghong-gt/) for his revisions to our code and discussions on the content of our paper.

## Citation

Some insights about neural voxel grids and dynamic scenes reconstruction originate from [TiNeuVox](https://github.com/hustvl/TiNeuVox). If you find this repository/work helpful in your research, welcome to cite these papers and give a ⭐.

```
@InProceedings{Wu_2024_CVPR,
    author    = {Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
    title     = {4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {20310-20320}
}

@inproceedings{TiNeuVox,
  author = {Fang, Jiemin and Yi, Taoran and Wang, Xinggang and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Nie\ss{}ner, Matthias and Tian, Qi},
  title = {Fast Dynamic Radiance Fields with Time-Aware Neural Voxels},
  year = {2022},
  booktitle = {SIGGRAPH Asia 2022 Conference Papers}
}
```
