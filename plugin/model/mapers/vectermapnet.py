from mmdet3d.models.builder import DETECTORS as MAPPERS
import torch.nn as nn
import torch


@MAPPERS.register_module()
class VectorMapNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dummy_layer = nn.Linear(10, 10)
    
    def forward(self):
        print('forward')

    def train_step(self, data_dict, optimizer):
        loss = torch.tensor(0.0, requires_grad=True)
        log_vars, num_samples = {}, 0
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=num_samples
        )
        return outputs