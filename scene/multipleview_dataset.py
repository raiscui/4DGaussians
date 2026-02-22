import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split,
        downsample_factor: int = 1,
        cam_ids: list[int] | None = None,
        video_n_views: int = 300,
        video_spiral_n_rots: float = 2,
        video_spiral_rads_scale: float = 1.0,
        video_spiral_hold_start: int = 0,
        video_time_mode: str = "linear",
        video_time_loop_period: int = 0,
    ):
        # MultipleView 的公平对比诉求:
        # - 数据生成阶段尽量保留原始帧(不做不可逆缩放).
        # - 训练/加载阶段再做下采样,语义上等价 FreeTimeGsVanilla 的 data_factor.
        self.downsample_factor = max(1, int(downsample_factor))
        self.split = str(split)

        # video(spiral) 渲染轨迹参数.
        # 说明:
        # - 这些参数只影响 split=="test" 时构造的 `video_cam_infos`.
        # - 训练/测试集的相机划分(cam_ids)与这里无关.
        self.video_n_views = max(1, int(video_n_views))
        self.video_spiral_n_rots = float(video_spiral_n_rots)
        self.video_spiral_rads_scale = float(video_spiral_rads_scale)
        self.video_spiral_hold_start = max(0, int(video_spiral_hold_start))
        # video 时间维度控制参数.
        # - time_mode: "linear" 或 "loop".
        # - loop_period: loop 周期(帧数). <=0 表示自动使用真实序列长度 x.
        self.video_time_mode = str(video_time_mode or "linear")
        self.video_time_loop_period = int(video_time_loop_period)

        # 说明:
        # - MultipleView 的 train/test 更合理的切分方式是"按相机划分"(camera hold-out),
        #   而不是"每个相机抽几帧".
        # - 这里通过 cam_ids 控制当前 dataset 实例包含哪些相机(camXX).
        # - 若 cam_ids=None,表示包含全部相机.
        self.cam_ids = sorted({int(x) for x in cam_ids}) if cam_ids is not None else None

        # MultipleView 默认 single_camera,但为了更健壮这里取第一个内参作为全局 intrinsics.
        first_cam_id = sorted(cam_intrinsics.keys())[0]
        base_cam = cam_intrinsics[first_cam_id]

        base_focal = float(base_cam.params[0])
        base_height = int(base_cam.height)
        base_width = int(base_cam.width)

        # 与 FreeTime 一致: floor 下采样,并同步缩放 focal(像素单位),保证 FOV 不变.
        height = max(1, base_height // self.downsample_factor)
        width = max(1, base_width // self.downsample_factor)
        focal = base_focal / float(self.downsample_factor)

        self.focal = [focal, focal]
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)

        if self.downsample_factor > 1:
            self.transform = T.Compose(
                [
                    T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR),
                    T.ToTensor(),
                ]
            )
        else:
            self.transform = T.ToTensor()
        (
            self.image_paths,
            self.image_poses,
            self.image_times,
            self.image_cam_ids,
        ) = self.load_images_path(cam_folder, cam_extrinsics, split)

        # 约定:
        # - video_cam_infos 用于渲染 spiral camera path,与 train/test 的 hold-out 相机选择无关.
        # - 当前代码在 readMultipleViewinfos 中会从 test dataset 上读取 video_cam_infos,
        #   因此这里保持 split=="test" 时构建该字段.
        if self.split == "test":
            self.video_cam_infos = self.get_video_cam_infos(cam_folder)
        
    
    def _parse_cam_id_from_colmap_name(self, name: str) -> int | None:
        """
        从 COLMAP 的 image name 里解析出 cam id.

        约定来源:
        - `scripts/preprocess_multipleview_from_videos.py` 会把每路相机首帧复制为:
          - image1.jpg, image2.jpg, ...
        - 训练数据目录则是:
          - cam01/frame_00001.jpg, cam02/frame_00001.jpg, ...
        """
        base = os.path.basename(str(name))
        stem, _ = os.path.splitext(base)
        if not stem.startswith("image"):
            return None
        suffix = stem[len("image") :]
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _iter_selected_extrinsics(self, cam_extrinsics):
        """把 cam_extrinsics 过滤并按 cam_id 排序,保证输出顺序稳定."""
        selected: list[tuple[int, object]] = []
        for key in cam_extrinsics:
            extr = cam_extrinsics[key]
            cam_id = self._parse_cam_id_from_colmap_name(getattr(extr, "name", ""))
            if cam_id is None:
                continue
            if self.cam_ids is not None and cam_id not in self.cam_ids:
                continue
            selected.append((cam_id, extr))
        selected.sort(key=lambda x: x[0])
        return selected

    def load_images_path(self, cam_folder, cam_extrinsics, split):
        cam_entries = self._iter_selected_extrinsics(cam_extrinsics)
        if not cam_entries:
            raise ValueError(
                "MultipleView 数据集没有找到任何可用相机. "
                "请检查 sparse_/images.bin 的 image name 是否为 imageN.jpg,以及 cam_ids 过滤条件是否正确."
            )

        # MultipleView 假设每个相机的帧数一致,这里用第一个相机目录确定长度.
        first_cam_id = int(cam_entries[0][0])
        first_cam_dir = os.path.join(cam_folder, f"cam{first_cam_id:02d}")
        frame_files = [p for p in os.listdir(first_cam_dir) if p.startswith("frame_") and p.lower().endswith(".jpg")]
        image_length = len(frame_files)
        if image_length <= 0:
            raise ValueError(f"MultipleView 相机目录没有找到任何帧: {first_cam_dir}")

        image_paths = []
        image_poses = []
        image_times = []
        image_cam_ids = []

        # 说明:
        # - train/test 的差异由 readMultipleViewinfos 决定(按相机切分).
        # - 单个 dataset 内部,我们保持"同一相机按时间递增"的顺序,更便于渲染成连续视频.
        for cam_id, extr in cam_entries:
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            images_folder = os.path.join(cam_folder, f"cam{int(cam_id):02d}")

            # 注意:
            # - 这里不再对 test 只抽 3 帧,而是保留完整时间序列.
            # - 如果需要小数据快速试跑,建议在数据生成阶段用 `--max-frames` 或 frame range 控制体量.
            for i in range(image_length):
                num = i + 1
                image_path = os.path.join(images_folder, "frame_" + str(num).zfill(5) + ".jpg")
                image_paths.append(image_path)
                image_poses.append((R,T))
                image_times.append(float(i / image_length))
                image_cam_ids.append(int(cam_id))

        return image_paths, image_poses, image_times, image_cam_ids
    
    def get_video_cam_infos(self,datadir):
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # MultipleView 的 video 输出使用 NeRF/LLFF 风格 spiral 相机轨迹.
        # 为了满足“镜头在中心视角停留更久”的需求,这里支持:
        # - 通过 `video_spiral_n_rots` 减少绕圈次数(更慢更稳).
        # - 通过 `video_spiral_hold_start` 在起始 pose 处重复若干帧(相机不动,但 time 仍推进).
        val_poses = get_spiral(
            poses,
            near_fars,
            rads_scale=self.video_spiral_rads_scale,
            N_views=self.video_n_views,
            N_rots=self.video_spiral_n_rots,
        )
        if self.video_spiral_hold_start > 0 and len(val_poses) > 0:
            hold = np.repeat(val_poses[:1], self.video_spiral_hold_start, axis=0)
            val_poses = np.concatenate([hold, val_poses], axis=0)

        cameras = []
        len_poses = len(val_poses)
        # ---------------------------------------------------------------------
        # video 的 time 序列生成策略(关键):
        #
        # - linear(旧行为): time 按 [0,1) 均匀推进,即 i/len_poses.
        #   当真实序列长度 x 远小于 len_poses(N)时,动作会被拉慢约 N/x 倍,看起来“几乎不动”.
        #
        # - loop(推荐): time 按真实序列长度 x 循环播放,即 (i % x)/x.
        #   这样相机轨迹仍然是 N 帧的平滑 spiral,但内容会按 x 帧持续运动.
        # ---------------------------------------------------------------------
        time_mode = self.video_time_mode.lower().strip()
        if time_mode == "loop":
            # x: 真实序列长度(每路相机帧数). MultipleView 默认各相机帧数一致.
            x = 0
            try:
                # self.image_paths 是按 cam_id 分组拼出来的,每个相机的帧数相同,
                # 这里用 cam01 的目录重新数一遍,避免依赖 self.image_paths 的布局细节.
                first_cam_dir = os.path.join(datadir, "cam01")
                frame_files = [
                    p
                    for p in os.listdir(first_cam_dir)
                    if p.startswith("frame_") and p.lower().endswith(".jpg")
                ]
                x = len(frame_files)
            except Exception:
                x = 0

            if x <= 0:
                # 兜底: 退回 linear,避免除零.
                times = [i / len_poses for i in range(len_poses)]
            else:
                period = int(self.video_time_loop_period) if int(self.video_time_loop_period) > 0 else int(x)
                period = max(1, min(int(period), int(x)))
                times = [(i % period) / float(period) for i in range(len_poses)]
        else:
            times = [i / len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]
