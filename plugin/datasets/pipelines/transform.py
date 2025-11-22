from mmdet.datasets.builder import PIPELINES
import mmcv
import numpy as np


@PIPELINES.register_module(force=True)
class ResizeMultiViewImages:
    def __init__(self, size, change_intrinsics=False):
        self.size = size
        self.change_intrinsics = change_intrinsics

    def __call__(self, results):
        new_imgs, post_intrinsics, post_ego2imgs = [], [], []

        for img, cam_intrinsic, ego2img in zip(
            results["img"], results["cam_intrinsics"], results["ego2img"]
        ):
            tmp, scaleW, scaleH = mmcv.imresize(
                img, (self.size[1], self.size[0]), return_scale=True
            )
            rot_resize_matrix = np.array(
                [[scaleW, 0, 0, 0], [0, scaleH, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
            new_imgs.append(tmp)
            post_intrinsic = rot_resize_matrix[:3, :3] @ cam_intrinsic
            post_ego2img = rot_resize_matrix @ ego2img
            post_intrinsics.append(post_intrinsic)
            post_ego2imgs.append(post_ego2img)
        
        results['img'] = new_imgs
        results['img_shape'] = [img.shape for img in new_imgs]
        if self.change_intrinsics:
            results.update({
                'cam_intrinsics': post_intrinsics,
                'ego2img': post_ego2imgs
            })
        return results