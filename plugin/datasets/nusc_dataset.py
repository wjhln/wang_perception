from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
import numpy as np
import time
import mmcv
from mmdet3d.datasets.pipelines import Compose


@DATASETS.register_module()
class NuscDataset(Dataset):
    def __init__(
        self,
        ann_file,
        modality=dict(
            use_camera=True,
            use_lidar=False,
            use_radar=False,
            use_map=True,
            use_exernal=False,
        ),
        pipeline=None,
        interval=1,
    ):
        super().__init__()
        self.ann_file = ann_file
        self.modality = modality
        self.interval = interval

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        self.load_annotations(self.ann_file)
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self, ann_file):
        print("collecting samples...")
        start_time = time.time()
        samples = mmcv.load(ann_file)[:: self.interval]
        print(f"collected {len(samples)} samples in {(time.time() - start_time):.2f}s")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def get_sample(self, index):
        sample = self.samples[index]
        location = sample['location']

        ego2img_rts = []
        for cam in sample['cams'].values():
            extrinsic, intrinsic = (
                np.array(cam['extrinsics']),
                np.array(cam['intrinsics']),
            )
            ego2img_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            ego2img_rt = viewpad @ ego2img_rt
            ego2img_rts.append(ego2img_rt)

        input_dict = {
            'sample_idx': sample['token'],
            'location': location,
            'img_filenames': [cam['img_fpath'] for cam in sample['cams'].values()],
            'cam_intrinsics': [cam['intrinsics'] for cam in sample['cams'].values()],
            'cam_extrinsics': [cam['extrinsics'] for cam in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'ego2global_translation': sample['e2g_translation'],
            'ego2global_rotation': sample['e2g_rotation'],
        }

        if self.modality['use_lidar']:
            input_dict.update(dict(pts_filename=sample['lidar_path']))

        return input_dict

    def prepare_data(self, index):
        input_dict = self.get_sample(index)
        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, index):
        data = self.prepare_data(index)
        return data
