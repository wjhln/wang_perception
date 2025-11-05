import argparse
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import train_model

def parse_args():
    parese = argparse.ArgumentParser(description='Train a detector')
    parese.add_argument('config')

    args = parese.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    datasets = build_dataset(cfg.data.train)
    train_model()

    # train_model(

    # )

if __name__ == '__main__':
    main()