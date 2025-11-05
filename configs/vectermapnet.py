

custom_imports = dict(
    imports=['plugin'],
    allow_failed_imports=False
)

data = dict(
    train = dict(
        type = 'NuscDataset',
        ann_file = './datasets/wang_vectermapnet_infos_train.pkl',
    )
)

model = dict(
    type='VecterMapNet',
)