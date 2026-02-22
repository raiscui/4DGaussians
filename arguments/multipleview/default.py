ModelHiddenParams = dict(
    kplanes_config={"grid_dimensions": 2, "input_coordinate_dim": 4, "output_coordinate_dim": 16, "resolution": [64, 64, 64, 150]},
    multires=[1, 2],
    defor_depth=0,
    net_width=128,
    plane_tv_weight=0.0002,
    time_smoothness_weight=0.001,
    l1_time_planes=0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False,
)
OptimizationParams = dict(
    dataloader=True,
    iterations=30000,
    batch_size=1,
    coarse_iterations=3000,
    densify_until_iter=10_000,
    # opacity_reset_interval = 60000,
    opacity_threshold_coarse=0.005,
    opacity_threshold_fine_init=0.005,
    opacity_threshold_fine_after=0.005,
    # pruning_interval = 2000
)

ModelParams = dict(
    # MultipleView 的 video(spiral) 渲染轨迹调参:
    # - video_time_mode=\"loop\": 当真实序列长度 x(例如 61) < video 轨迹帧数 N(例如 300)时,
    #   让时间维度按 x 循环播放,避免动作被拉慢到“几乎不动”.
    # - video_spiral_n_rots: 减少绕圈次数,整体运动更慢更稳.
    #
    # 说明:
    # - 之前的 `video_spiral_hold_start` 只能让相机不动,并不能解决“动作循环播放”的诉求,
    #   因此这里默认回退为 0(不额外停留).
    video_time_mode="loop",
    video_spiral_hold_start=0,
    video_spiral_n_rots=1,
    # 关键: 缩小晃动范围(建议先从 0.6 开始试)
    video_spiral_rads_scale=0.6,
)
