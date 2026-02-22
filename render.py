#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def _pad_frame_to_multiple(frame_u8: np.ndarray, multiple: int = 16) -> np.ndarray:
    """
    为 mp4 编码做兼容性 padding,避免 imageio/ffmpeg writer 触发隐式 resize.

    背景:
    - imageio 的 ffmpeg writer 默认要求帧尺寸能被 macro_block_size(常见为 16)整除.
    - 若不满足,它会自动 resize(插值),并打印 warning:
      "input image is not divisible by macro_block_size=16, resizing ..."
    - 对渲染结果来说,我们更希望"不缩放内容",因此选择在写视频前做轻量 padding.

    约定:
    - 只在右侧与下侧补边,并用 edge 模式复制边界像素,避免出现黑边.
    """
    if multiple <= 1:
        return frame_u8
    if frame_u8.ndim < 2:
        return frame_u8

    height = int(frame_u8.shape[0])
    width = int(frame_u8.shape[1])
    pad_h = (-height) % int(multiple)
    pad_w = (-width) % int(multiple)
    if pad_h == 0 and pad_w == 0:
        return frame_u8

    if frame_u8.ndim == 2:
        pad_spec = ((0, pad_h), (0, pad_w))
    else:
        pad_spec = ((0, pad_h), (0, pad_w), (0, 0))

    return np.pad(frame_u8, pad_spec, mode="edge")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    # MultipleView 的 test 集合往往包含多个相机视角.
    # 如果把不同相机的帧直接拼成一条 mp4,观看上会表现为"视角乱跳".
    # 这里按 cam_id 分组,额外输出 per-camera 的 mp4,并把默认 video_rgb.mp4 指向第一路相机,便于直观看时序.
    multipleview_cam_ids = None
    multipleview_render_images_by_cam = None
    if cam_type == "MultipleView" and name == "test":
        try:
            dataset = getattr(views, "dataset", None)
            cam_ids = getattr(dataset, "image_cam_ids", None)
            if cam_ids is not None and len(cam_ids) == len(views):
                multipleview_cam_ids = cam_ids
                multipleview_render_images_by_cam = {}
        except Exception:
            multipleview_cam_ids = None
            multipleview_render_images_by_cam = None

    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        
        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type)["render"]
        rendering_u8 = to8b(rendering).transpose(1,2,0)
        rendering_u8 = _pad_frame_to_multiple(rendering_u8, multiple=16)
        render_images.append(rendering_u8)
        if multipleview_cam_ids is not None and multipleview_render_images_by_cam is not None:
            cam_id = int(multipleview_cam_ids[idx])
            multipleview_render_images_by_cam.setdefault(cam_id, []).append(rendering_u8)
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    
    out_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
    if (
        cam_type == "MultipleView"
        and name == "test"
        and multipleview_render_images_by_cam is not None
        and len(multipleview_render_images_by_cam) > 0
    ):
        # 说明:
        # - 默认 video_rgb.mp4 输出为"第一路相机"的完整时间序列,避免多相机混合造成观感乱跳.
        # - 同时保留一个 allcams 版本,并按 camXX 分别输出,方便逐个相机检查.
        cam_ids_sorted = sorted(multipleview_render_images_by_cam.keys())
        primary_cam_id = int(cam_ids_sorted[0])

        imageio.mimwrite(os.path.join(out_dir, "video_rgb.mp4"), multipleview_render_images_by_cam[primary_cam_id], fps=30)
        imageio.mimwrite(os.path.join(out_dir, "video_rgb_allcams.mp4"), render_images, fps=30)

        for cam_id in cam_ids_sorted:
            imageio.mimwrite(
                os.path.join(out_dir, f"video_rgb_cam{int(cam_id):02d}.mp4"),
                multipleview_render_images_by_cam[int(cam_id)],
                fps=30,
            )

        print(
            "[MultipleView] test 视频已按相机分组输出: "
            f"primary=cam{primary_cam_id:02d}, allcams=video_rgb_allcams.mp4"
        )
    else:
        imageio.mimwrite(os.path.join(out_dir, "video_rgb.mp4"), render_images, fps=30)
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        from utils.params_utils import load_config_file, merge_hparams
        config = load_config_file(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
