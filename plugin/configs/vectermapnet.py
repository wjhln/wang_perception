dist_params = dict(backend="nccl")


log_level = "INFO"

custom_imports = dict(imports=["plugin"], allow_failed_imports=False)


train_pipelines = [
    dict(type="LoadMultiViewImagesFromFiles"),
    dict(
        type="ResizeMultiViewImages",
        size=(int(128 * 2), int((16 / 9 * 128) * 2)),
        change_intrinsics=True,
    ),
    dict(type="Collect3D", keys=["img"], meta_keys=["img_shape"]),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type="NuscDataset",
        ann_file="./datasets/wang_vectermapnet_infos_train.pkl",
        pipeline=train_pipelines,
    ),
)

model = dict(
    type="VectorMapNet",
)

optimizer = dict(
    type="AdamW",
    lr=1e-3,
)
optimizer_config = dict(grad_clip=dict(max_norm=0.5, norm_type=2))

lr_config = dict(
    policy="step", warmup="linear", warmup_iters=400, warmup_ratio=0.1, step=[100, 120]
)

checkpoint_config = dict(interval=100)
load_from = None
resume_from = None

log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])


total_epochs = 130
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
workflow = [("train", 1)]
