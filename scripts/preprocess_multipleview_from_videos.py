import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

"""
从多机位视频生成本仓库 MultipleView 数据集.

核心目标是把一个视频目录(多路 mp4/mov 等)转换成:

data/multipleview/<dataset>/
  cam01/frame_00001.jpg ...
  cam02/frame_00001.jpg ...
  sparse_/cameras.bin images.bin points3D.bin ...
  points3D_multipleview.ply
  poses_bounds_multipleview.npy

说明:
- 本仓库的 MultipleView 训练假设每个相机(camXX)帧数一致.
- COLMAP 只用每路视频的第一帧来估计多相机的静态相机参数.
"""


# -----------------------------------------------------------------------------
# 通用工具
# -----------------------------------------------------------------------------


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    """运行外部命令,失败则抛异常,并保留足够的信息用于定位问题."""
    cmd_str = " ".join(cmd)
    print(f"[cmd] {cmd_str}")
    # 说明:
    # - 某些环境下的 colmap CLI 仍然会初始化 Qt(即使不打开 GUI),
    #   若没有可用 display 会直接 SIGABRT.
    # - 设置 QT_QPA_PLATFORM=offscreen 可强制走无窗口后端,避免该类崩溃.
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        # 负 returncode 表示被信号终止(例如 -9 => SIGKILL).
        # 这类问题经常是 OOM killer 或外部强杀导致,需要把信息提示清楚.
        extra_hint = ""
        if proc.returncode < 0:
            sig_num = -int(proc.returncode)
            try:
                sig_name = signal.Signals(sig_num).name
            except ValueError:
                sig_name = f"SIG{sig_num}"

            extra_hint = f"\n[hint] 进程被信号终止: {sig_name}({sig_num}).\n"
            if sig_num == int(signal.SIGKILL) and len(cmd) >= 2 and cmd[0] == "colmap":
                # SIGKILL 基本无法由进程自行捕获,所以更可能是系统层面的强杀.
                # 最常见的是内存不够触发 OOM killer.
                extra_hint += (
                    "[hint] SIGKILL 常见原因是 OOM(内存不足)导致系统直接杀死进程.\n"
                    "[hint] 若你在跑 `colmap feature_extractor`,可以尝试降低 SIFT 开销:\n"
                    "       - 下调 `--colmap-sift-max-image-size`(例如 2000/1600)\n"
                    "       - 下调 `--colmap-sift-max-num-features`(例如 4096)\n"
                    "       - 下调 `--colmap-num-threads`(例如 1 或 2)\n"
                    "       - 关闭 affine/dsp(本脚本默认已关闭)\n"
                )

        raise RuntimeError(f"命令失败(returncode={proc.returncode}): {cmd_str}{extra_hint}\n{proc.stdout}")


def _list_videos(videos_dir: Path) -> list[Path]:
    """列出视频文件并排序,用于稳定的 cam01/cam02 映射."""
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    videos = [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(videos, key=lambda p: p.name)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _count_frames(cam_dir: Path) -> int:
    return len(list(cam_dir.glob("frame_*.jpg")))


def _truncate_frames(cam_dir: Path, keep: int) -> None:
    """把 camXX 目录里的帧截断到 keep,确保多路相机长度一致."""
    frames = sorted(cam_dir.glob("frame_*.jpg"))
    for p in frames[keep:]:
        p.unlink()


def _ffmpeg_scale_filter(max_size: int) -> str:
    """
    生成 ffmpeg scale 表达式,保证最长边 <= max_size,并保持比例.
    - 用 -2 让另一边自动取偶数,避免某些编码器/下游工具对奇数尺寸不友好.
    """
    # 如果 iw > ih,限制宽度; 否则限制高度.
    return (
        "scale="
        f"'if(gt(iw,ih),min(iw\\,{max_size}),-2)'"
        f":'if(gt(iw,ih),-2,min(ih\\,{max_size}))'"
    )


def _ffmpeg_select_filter(frame_start: int, frame_end: int | None, frame_step: int) -> str:
    """
    生成 ffmpeg select 表达式,按"帧索引"抽取,更贴近 FreeTime 的 [start_frame,end_frame) 语义.

    说明:
    - n 是解码后的帧序号(从 0 开始).
    - 这里用 gte/lt/mod 组合,并对表达式里的逗号做 `\\,` 转义,避免被 filtergraph 解析成分隔符.
    """
    if frame_step <= 0:
        raise ValueError(f"frame_step 必须 >= 1,当前为: {frame_step}")

    expr = f"gte(n\\,{frame_start})"
    if frame_end is not None:
        expr += f"*lt(n\\,{frame_end})"
    if frame_step != 1:
        expr += f"*not(mod(n-{frame_start}\\,{frame_step}))"

    return f"select='{expr}'"


def _extract_frames_one_video(
    video_path: Path,
    cam_dir: Path,
    fps: int | None,
    max_size: int | None,
    max_frames: int | None,
    frame_start: int | None,
    frame_end: int | None,
    frame_step: int,
    jpg_quality: int,
    overwrite: bool,
) -> None:
    """
    使用 ffmpeg 抽帧到 cam_dir/frame_%05d.jpg.

    参数选择逻辑:
    - 若提供 frame_start/frame_end/frame_step(任意一个),则按帧索引抽取(更贴近 FreeTime).
    - 否则按 fps 重采样抽帧.
    - max_size: 可选缩放,用于控制落盘分辨率(注意会不可逆降质).
    - max_frames: 可选上限,用于快速试跑或强制对齐(限制输出帧数).
    """
    _ensure_dir(cam_dir)

    if overwrite:
        for p in cam_dir.glob("frame_*.jpg"):
            p.unlink()

    out_pattern = str(cam_dir / "frame_%05d.jpg")

    vf_parts: list[str] = []
    use_frame_range = frame_start is not None or frame_end is not None or frame_step != 1
    if use_frame_range:
        start = 0 if frame_start is None else int(frame_start)
        end = None if frame_end is None else int(frame_end)
        vf_parts.append(_ffmpeg_select_filter(start, end, int(frame_step)))
    else:
        if fps is not None:
            vf_parts.append(f"fps={fps}")
    if max_size is not None:
        vf_parts.append(_ffmpeg_scale_filter(max_size))
    vf = ",".join(vf_parts) if vf_parts else None

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path)]
    if vf is not None:
        cmd += ["-vf", vf]
    if use_frame_range:
        # 避免基于时间戳的补帧/重复,确保输出序列严格是 select 选出的帧.
        cmd += ["-vsync", "0"]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-q:v", str(jpg_quality), "-start_number", "1", out_pattern]
    _run_cmd(cmd)


def _copy_sparse_model(src_sparse0: Path, dst_sparse_: Path) -> None:
    """把 COLMAP sparse/0 拷贝到数据集 sparse_ 目录."""
    if dst_sparse_.exists():
        shutil.rmtree(dst_sparse_)
    _ensure_dir(dst_sparse_)
    for p in src_sparse0.iterdir():
        if p.is_file():
            shutil.copy2(p, dst_sparse_ / p.name)


def _keep_colmap_tmp_dir(colmap_dir: Path, dataset_dir: Path) -> None:
    """把 COLMAP 临时目录复制到数据集目录下,用于排查失败原因."""
    keep_dir = dataset_dir / "_colmap_tmp"
    if keep_dir.exists():
        shutil.rmtree(keep_dir)
    shutil.copytree(colmap_dir, keep_dir)
    print(f"[info] 已保留 COLMAP 临时目录到: {keep_dir}")


# -----------------------------------------------------------------------------
# poses_bounds_multipleview.npy 生成(按 LLFF 约定)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _ColmapImage:
    image_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray


def _compute_poses_bounds_from_colmap(colmap_dir: Path, out_path: Path) -> None:
    """
    从 COLMAP sparse/0 模型计算 LLFF 兼容的 poses_bounds.npy(Nx17).

    这里复刻 LLFF 的核心逻辑:
    - pose: c2w(3x4) + hwf(3x1) => 3x5
    - 轴变换: [-u, r, -t]
    - bounds: 对每个相机,统计可见点的深度分位数(0.1%, 99.9%)
    """
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # 使用仓库自带的 COLMAP loader(读相机/图像),并复用 scripts/colmap_converter 的 points3D 解析(含 track).
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

    # scripts 目录是 namespace package,可直接 import.
    from scripts.colmap_converter import read_points3D_binary

    sparse0 = colmap_dir / "sparse" / "0"
    cameras = read_intrinsics_binary(str(sparse0 / "cameras.bin"))
    images = read_extrinsics_binary(str(sparse0 / "images.bin"))
    points3d = read_points3D_binary(str(sparse0 / "points3D.bin"))

    if not cameras:
        raise RuntimeError("COLMAP cameras.bin 为空,无法生成 poses_bounds.")

    # MultipleView 代码侧默认只使用 camera_id=1 的内参.
    # 为了与现有实现兼容,这里也取第一个相机作为全局 intrinsics.
    first_cam = cameras[sorted(cameras.keys())[0]]
    h = int(first_cam.height)
    w = int(first_cam.width)
    focal = float(first_cam.params[0])
    hwf = np.array([h, w, focal], dtype=np.float32).reshape(3, 1)

    # 按 image name 排序,保证输出顺序稳定.
    image_list: list[_ColmapImage] = []
    for image_id, im in images.items():
        image_list.append(_ColmapImage(image_id=image_id, name=im.name, qvec=im.qvec, tvec=im.tvec))
    image_list.sort(key=lambda x: x.name)

    image_id_to_index = {im.image_id: idx for idx, im in enumerate(image_list)}

    # 组装 c2w pose.
    bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 4)
    w2c_mats: list[np.ndarray] = []
    for im in image_list:
        r = qvec2rotmat(im.qvec).astype(np.float32)
        t = im.tvec.astype(np.float32).reshape(3, 1)
        m = np.concatenate([np.concatenate([r, t], axis=1), bottom], axis=0)  # 4x4
        w2c_mats.append(m)
    w2c = np.stack(w2c_mats, axis=0)  # (N,4,4)
    c2w = np.linalg.inv(w2c)  # (N,4,4)

    poses = c2w[:, :3, :4].transpose(1, 2, 0)  # (3,4,N)
    poses = np.concatenate([poses, np.tile(hwf[..., None], (1, 1, poses.shape[-1]))], axis=1)  # (3,5,N)

    # LLFF 轴变换: [-u, r, -t]
    poses = np.concatenate(
        [
            poses[:, 1:2, :],
            poses[:, 0:1, :],
            -poses[:, 2:3, :],
            poses[:, 3:4, :],
            poses[:, 4:5, :],
        ],
        axis=1,
    )

    # 计算每个相机的深度 bounds.
    n_images = poses.shape[-1]
    cam_pos = poses[:3, 3, :].T  # (N,3)
    cam_z = poses[:3, 2, :].T  # (N,3)

    depth_lists: list[list[float]] = [[] for _ in range(n_images)]
    for pt in points3d.values():
        p = np.asarray(pt.xyz, dtype=np.float32)
        for image_id in pt.image_ids:
            idx = image_id_to_index.get(int(image_id))
            if idx is None:
                continue
            depth = -float(np.dot(p - cam_pos[idx], cam_z[idx]))
            depth_lists[idx].append(depth)

    bounds = np.zeros((n_images, 2), dtype=np.float32)
    for i in range(n_images):
        zs = np.asarray(depth_lists[i], dtype=np.float32)
        if zs.size == 0:
            raise RuntimeError(f"相机 index={i} 没有任何可见点,无法估计 near/far.")
        close_depth = float(np.percentile(zs, 0.1))
        inf_depth = float(np.percentile(zs, 99.9))
        bounds[i, 0] = close_depth
        bounds[i, 1] = inf_depth

    save_arr = []
    for i in range(n_images):
        save_arr.append(np.concatenate([poses[..., i].ravel(), bounds[i]], axis=0))
    save_arr = np.asarray(save_arr, dtype=np.float32)

    _ensure_dir(out_path.parent)
    np.save(str(out_path), save_arr)


# -----------------------------------------------------------------------------
# 点云生成
# -----------------------------------------------------------------------------


def _make_pointcloud_from_sparse(colmap_dir: Path, out_ply: Path) -> None:
    """把 COLMAP sparse/0/points3D.bin 转成 ply,用于 MultipleView 初始化."""
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from scene.dataset_readers import storePly
    from scripts.colmap_converter import read_points3D_binary

    sparse0 = colmap_dir / "sparse" / "0"
    points3d = read_points3D_binary(str(sparse0 / "points3D.bin"))
    if not points3d:
        raise RuntimeError("COLMAP points3D.bin 为空,无法生成点云.")

    xyz = np.stack([np.asarray(p.xyz, dtype=np.float32) for p in points3d.values()], axis=0)
    rgb = np.stack([np.asarray(p.rgb, dtype=np.float32) for p in points3d.values()], axis=0)

    _ensure_dir(out_ply.parent)
    storePly(str(out_ply), xyz, rgb)

    # 与仓库其他数据处理保持一致: 把点数下采样到 <= 40000.
    downsample_script = repo_root / "scripts" / "downsample_point.py"
    tmp_out = out_ply.with_suffix(".tmp.ply")
    _run_cmd([sys.executable, str(downsample_script), str(out_ply), str(tmp_out)])
    tmp_out.replace(out_ply)


def _make_pointcloud_from_dense(colmap_dir: Path, out_ply: Path, keep_intermediate: bool) -> None:
    """
    运行 COLMAP dense 重建并输出 fused.ply,再用 downsample_point.py 生成 points3D_multipleview.ply.
    """
    dense_dir = colmap_dir / "dense"
    _ensure_dir(dense_dir)

    sparse0 = colmap_dir / "sparse" / "0"
    images_dir = colmap_dir / "images"
    fused_ply = dense_dir / "fused.ply"

    _run_cmd(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(images_dir),
            "--input_path",
            str(sparse0),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
        ]
    )
    _run_cmd(
        [
            "colmap",
            "patch_match_stereo",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
        ]
    )
    _run_cmd(
        [
            "colmap",
            "stereo_fusion",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(fused_ply),
        ]
    )

    # 下采样到 <= 40000 点.
    repo_root = Path(__file__).resolve().parents[1]
    downsample_script = repo_root / "scripts" / "downsample_point.py"
    _run_cmd([sys.executable, str(downsample_script), str(fused_ply), str(out_ply)])

    if not keep_intermediate:
        # dense/workspace 很大,默认清掉.
        # 注意: 我们的最终输出 out_ply 写在数据集目录下,不在 dense_dir 内,因此可直接删除.
        shutil.rmtree(dense_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="从多机位视频生成 MultipleView 数据集")
    parser.add_argument("--videos-dir", type=str, required=True, help="多机位视频目录,例如 /cloud/.../videos")
    parser.add_argument("--dataset-name", type=str, required=True, help="输出数据集名称,会写到 data/multipleview/<name>")
    parser.add_argument("--data-root", type=str, default="data/multipleview", help="输出根目录(默认 data/multipleview)")

    parser.add_argument("--fps", type=int, default=10, help="抽帧帧率(默认 10).设为 0 表示导出全部帧(通常非常大)")
    parser.add_argument("--max-size", type=int, default=1920, help="抽帧时最长边缩放上限(默认 1920).设为 0 表示不缩放")
    parser.add_argument("--max-frames", type=int, default=0, help="每路视频最多抽取多少帧(0 表示不限制)")
    parser.add_argument("--frame-start", type=int, default=0, help="按帧索引抽取: 起始帧(包含,0-based).用于对齐 FreeTime 的 start_frame")
    parser.add_argument("--frame-end", type=int, default=0, help="按帧索引抽取: 结束帧(不包含,0-based).0 表示不限制.用于对齐 FreeTime 的 end_frame")
    parser.add_argument("--frame-step", type=int, default=1, help="按帧索引抽取: 步长(默认 1).用于对齐 FreeTime 的 frame_step")
    parser.add_argument("--jpg-quality", type=int, default=2, help="ffmpeg JPG 输出质量(1-31,越小越好,默认 2)")
    parser.add_argument("--overwrite", action="store_true", help="若目标 camXX 已存在帧,允许覆盖重抽")
    parser.add_argument("--limit-cams", type=int, default=0, help="只处理前 N 路视频(用于快速验证,0 表示全处理)")

    parser.add_argument(
        "--pointcloud",
        type=str,
        default="sparse",
        choices=["sparse", "dense"],
        help="点云来源: sparse(快) 或 dense(慢但更密).默认 sparse",
    )
    parser.add_argument("--keep-colmap-tmp", action="store_true", help="保留 COLMAP 临时目录,便于排查")

    # COLMAP 参数:
    # - 脚本默认只用每路相机的首帧做静态标定,通常不需要把 SIFT 参数拉得很激进.
    # - 在高分辨率(接近 4K) + 小内存的环境里,过高的 max_image_size/max_num_features + 多线程
    #   很容易触发 OOM killer,表现为 returncode=-9(SIGKILL).
    default_colmap_threads = min(4, os.cpu_count() or 4)
    parser.add_argument(
        "--colmap-num-threads",
        type=int,
        default=default_colmap_threads,
        help=(
            "COLMAP 线程数(同时用于 SiftExtraction.num_threads 与 SiftMatching.num_threads). "
            f"默认 {default_colmap_threads}. 设为 -1 表示让 COLMAP 自动(可能更快但更吃内存)"
        ),
    )
    parser.add_argument(
        "--colmap-sift-max-image-size",
        type=int,
        default=3200,
        help="COLMAP SIFT 特征提取最大边长(默认 3200,更省内存).例如 2000/1600 会更稳",
    )
    parser.add_argument(
        "--colmap-sift-max-num-features",
        type=int,
        default=8192,
        help="COLMAP SIFT 每张图最多特征点数(默认 8192).例如 4096 会更稳",
    )
    parser.add_argument("--colmap-sift-affine", action="store_true", help="启用 affine shape estimation(更慢更吃内存)")
    parser.add_argument("--colmap-sift-dsp", action="store_true", help="启用 domain size pooling(DSP)(更慢更吃内存)")

    args = parser.parse_args()

    # 参数校验: 尽早失败,避免把无效参数带进 colmap 里产生更难懂的错误.
    if int(args.colmap_num_threads) == 0 or int(args.colmap_num_threads) < -1:
        raise ValueError("--colmap-num-threads 只能为 -1(自动) 或 >= 1(显式线程数)")
    if int(args.colmap_sift_max_image_size) <= 0:
        raise ValueError("--colmap-sift-max-image-size 必须 >= 1")
    if int(args.colmap_sift_max_num_features) <= 0:
        raise ValueError("--colmap-sift-max-num-features 必须 >= 1")

    videos_dir = Path(args.videos_dir).resolve()
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos_dir 不存在: {videos_dir}")

    videos = _list_videos(videos_dir)
    if not videos:
        raise RuntimeError(f"目录里没找到视频文件: {videos_dir}")

    if args.limit_cams and args.limit_cams > 0:
        videos = videos[: args.limit_cams]

    # fps/max_size/max_frames 0 表示禁用.
    fps = None if args.fps == 0 else int(args.fps)
    max_size = None if args.max_size == 0 else int(args.max_size)
    max_frames = None if args.max_frames == 0 else int(args.max_frames)

    # frame range 模式:
    # - frame_end=0 表示不限制(通常不建议,可能会导出非常多帧).
    # - 只要 frame_start/frame_end/frame_step 任意一个不是默认值,就启用该模式,并忽略 fps 的语义.
    use_frame_range = (int(args.frame_start) != 0) or (int(args.frame_end) != 0) or (int(args.frame_step) != 1)
    frame_start = None
    frame_end = None
    frame_step = 1
    if use_frame_range:
        frame_start = max(0, int(args.frame_start))
        frame_end = None if int(args.frame_end) == 0 else int(args.frame_end)
        frame_step = int(args.frame_step)
        if frame_step <= 0:
            raise ValueError(f"--frame-step 必须 >= 1,当前为: {frame_step}")
        if frame_end is not None and frame_end <= frame_start:
            raise ValueError(f"--frame-end 必须 > --frame-start. 当前为: start={frame_start}, end={frame_end}")

    dataset_dir = Path(args.data_root).resolve() / args.dataset_name
    print(f"[info] 输出数据集目录: {dataset_dir}")
    _ensure_dir(dataset_dir)

    # 1) 抽帧: videos -> camXX/frame_*.jpg
    cam_dirs: list[Path] = []
    for idx, video_path in enumerate(videos, start=1):
        cam_dir = dataset_dir / f"cam{idx:02d}"
        cam_dirs.append(cam_dir)
        print(f"[info] 抽帧: {video_path.name} -> {cam_dir.name}")
        _extract_frames_one_video(
            video_path=video_path,
            cam_dir=cam_dir,
            fps=fps,
            max_size=max_size,
            max_frames=max_frames,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_step=frame_step,
            jpg_quality=int(args.jpg_quality),
            overwrite=bool(args.overwrite),
        )

    # 2) 对齐帧数: 多路相机取最小长度并截断.
    frame_counts = [_count_frames(d) for d in cam_dirs]
    if any(c == 0 for c in frame_counts):
        raise RuntimeError(f"至少有一路相机抽帧为 0,请检查视频是否损坏. counts={frame_counts}")
    min_len = min(frame_counts)
    max_len = max(frame_counts)
    if min_len != max_len:
        print(f"[warn] 多路相机帧数不一致,将截断到最短长度: min={min_len}, max={max_len}")
        for d in cam_dirs:
            _truncate_frames(d, keep=min_len)

    # 3) COLMAP: 只用每路相机第一帧估计内外参.
    with tempfile.TemporaryDirectory(prefix="colmap_multipleview_") as tmp:
        colmap_dir = Path(tmp)
        try:
            images_dir = colmap_dir / "images"
            _ensure_dir(images_dir)

            # 把每路相机的第一帧复制为 image1.jpg/image2.jpg...
            for idx, cam_dir in enumerate(sorted(cam_dirs), start=1):
                src = cam_dir / "frame_00001.jpg"
                if not src.exists():
                    raise RuntimeError(f"缺少首帧: {src}")
                dst = images_dir / f"image{idx}.jpg"
                shutil.copy2(src, dst)

            db_path = colmap_dir / "database.db"
            sparse_dir = colmap_dir / "sparse"
            _ensure_dir(sparse_dir)

            # feature_extractor: 强制 single_camera,与 multipleview_dataset 的假设一致.
            _run_cmd(
                [
                    "colmap",
                    "feature_extractor",
                    "--database_path",
                    str(db_path),
                    "--image_path",
                    str(images_dir),
                    "--ImageReader.single_camera",
                    "1",
                    "--ImageReader.camera_model",
                    "SIMPLE_PINHOLE",
                    # 在无 GPU/无 OpenGL 的 headless 环境里,强制使用 CPU SIFT.
                    "--SiftExtraction.use_gpu",
                    "0",
                    "--SiftExtraction.num_threads",
                    str(int(args.colmap_num_threads)),
                    "--SiftExtraction.max_image_size",
                    str(int(args.colmap_sift_max_image_size)),
                    "--SiftExtraction.max_num_features",
                    str(int(args.colmap_sift_max_num_features)),
                    "--SiftExtraction.estimate_affine_shape",
                    "1" if bool(args.colmap_sift_affine) else "0",
                    "--SiftExtraction.domain_size_pooling",
                    "1" if bool(args.colmap_sift_dsp) else "0",
                ]
            )
            _run_cmd(
                [
                    "colmap",
                    "exhaustive_matcher",
                    "--database_path",
                    str(db_path),
                    # 同上,禁用 GPU 匹配.
                    "--SiftMatching.use_gpu",
                    "0",
                    "--SiftMatching.num_threads",
                    str(int(args.colmap_num_threads)),
                ]
            )
            _run_cmd(
                [
                    "colmap",
                    "mapper",
                    "--database_path",
                    str(db_path),
                    "--image_path",
                    str(images_dir),
                    "--output_path",
                    str(sparse_dir),
                ]
            )

            sparse0 = sparse_dir / "0"
            if not sparse0.exists():
                raise RuntimeError("COLMAP mapper 没有生成 sparse/0,请检查匹配是否成功.")

            # 4) 拷贝 sparse_ 到数据集.
            dst_sparse_ = dataset_dir / "sparse_"
            _copy_sparse_model(sparse0, dst_sparse_)

            # 5) 生成 poses_bounds_multipleview.npy
            poses_out = dataset_dir / "poses_bounds_multipleview.npy"
            _compute_poses_bounds_from_colmap(colmap_dir, poses_out)

            # 6) 生成点云
            ply_out = dataset_dir / "points3D_multipleview.ply"
            if args.pointcloud == "dense":
                try:
                    _make_pointcloud_from_dense(colmap_dir, ply_out, keep_intermediate=bool(args.keep_colmap_tmp))
                except Exception as e:
                    print(f"[warn] dense 点云生成失败,将回退到 sparse 点云. error={e}")
                    _make_pointcloud_from_sparse(colmap_dir, ply_out)
            else:
                _make_pointcloud_from_sparse(colmap_dir, ply_out)
        except Exception:
            # 即使失败也尽量保留现场,否则 TemporaryDirectory 会在异常回溯时自动清理,很难排障.
            if args.keep_colmap_tmp:
                try:
                    _keep_colmap_tmp_dir(colmap_dir, dataset_dir)
                except Exception as e:
                    print(f"[warn] 保留 COLMAP 临时目录失败: {e}")
            raise
        else:
            if args.keep_colmap_tmp:
                _keep_colmap_tmp_dir(colmap_dir, dataset_dir)

    print("[done] MultipleView 数据集生成完成.")
    print(f"[done] dataset_dir: {dataset_dir}")


if __name__ == "__main__":
    main()
