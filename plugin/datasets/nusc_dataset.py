from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
import numpy as np
import time
import mmcv


@DATASETS.register_module()
class NuscDataset(Dataset):
    def __init__(self, ann_file, interval=1):
        super().__init__()
        self.ann_file = ann_file
        self.interval = interval


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
    
    def __getitem__(self, index):
        data = []
        return data
