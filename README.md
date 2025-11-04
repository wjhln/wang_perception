# 自动驾驶感知各sota方案整理练习
开始系统深入学习感知，萌新一个，欢迎交流与指教

TODO：
- [ ] VectorMapNet

## 环境安装 
```shell
# 克隆源码mmdet3d v1.0.0rc6版本
git clone --branch=v1.0.0rc6 --single-branch https://github.com/open-mmlab/mmdetection3d.git
# 创建conda环境
conda create -n mmdet3d python=3.8
conda activate mmdet3d

# 安装torch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# 安装mmcv3d
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0/index.html

pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e . -v

```
