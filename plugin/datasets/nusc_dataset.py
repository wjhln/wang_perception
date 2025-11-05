from mmdet.datasets import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module()
class NuscDataset(Dataset):
    def __init__(self, ann_file):
        super().__init__()
        self.ann_file = ann_file