from mmdet.datasets.builder import PIPELINES
import mmcv
import numpy as np
import cv2 as cv

@PIPELINES.register_module(force=True)
class LoadMultiViewImagesFromFiles:
    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results["img_filenames"]
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [i.astype(np.float32) for i in img]
        results["img"] = img
        results["img_shape"] = [i.shape for i in img]
        results["ori_shape"] = [i.shape for i in img]
        results["pad_shape"] = [i.shape for i in img]
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        results["img_fields"] = ["img"]
        return results
