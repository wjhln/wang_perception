from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module(force=True)
class VectorizeLocalMap():
    def __init__(self):
        pass