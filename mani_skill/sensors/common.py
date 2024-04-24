from typing import Dict

from mani_skill.sensors.camera import CameraConfig


def update_camera_cfgs_from_dict(
    camera_cfgs: Dict[str, CameraConfig], cfg_dict: Dict[str, dict]
):
    # Update CameraConfig to StereoDepthCameraConfig
    if cfg_dict.pop("use_stereo_depth", False):
        from .stereodepth import StereoDepthCameraConfig  # fmt: skip
        for name, cfg in camera_cfgs.items():
            camera_cfgs[name] = StereoDepthCameraConfig.fromCameraConfig(cfg)

    # First, apply global configuration
    for k, v in cfg_dict.items():
        if k in camera_cfgs:
            continue
        for cfg in camera_cfgs.values():
            if k == "add_segmentation":
                # TODO (stao): doesn't work this way anymore
                cfg.texture_names += ("Segmentation",)
            elif not hasattr(cfg, k):
                raise AttributeError(f"{k} is not a valid attribute of CameraConfig")
            else:
                setattr(cfg, k, v)
    # Then, apply camera-specific configuration
    for name, v in cfg_dict.items():
        if name not in camera_cfgs:
            continue

        # Update CameraConfig to StereoDepthCameraConfig
        if v.pop("use_stereo_depth", False):
            from .stereodepth import StereoDepthCameraConfig  # fmt: skip
            cfg = camera_cfgs[name]
            camera_cfgs[name] = StereoDepthCameraConfig.fromCameraConfig(cfg)

        cfg = camera_cfgs[name]
        for kk in v:
            assert hasattr(cfg, kk), f"{kk} is not a valid attribute of CameraConfig"
        cfg.__dict__.update(v)
