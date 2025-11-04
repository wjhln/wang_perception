import argparse
import mmcv
import numpy as np
from pyquaternion import Quaternion
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=str, help="specify the root path of dataset"
    )
    parser.add_argument(
        "-v",
        "--version",
        choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
        default="v1.0-trainval",
    )
    parser.add_argument(
        '--dest_path',
        type=str,
        help="specify the destination path"
    )
    args = parser.parse_args()
    print(args)
    return args

def create_nuscenes_infos_map(root_path, dest_path=None, version='v1.0-trainval'):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.train
        val_scenes = []
    else:
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    is_test = 'test' in version
    if is_test:
        print('test scense: {}'.format(len(train_scenes))) 
    else:
        print('train scense: {}, val scense: {}'.format(len(train_scenes), len(val_scenes)))

    train_samples, val_samples, test_samples = [], [], []
    
    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sample_data = nusc.get('sample_data', lidar_token)
        calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])         
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        lidar_path = nusc.get_sample_data_path(lidar_token)

        mmcv.check_file_exist(lidar_path)
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        location = log['location']
        scene_name = scene['name']

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'cams': {},
            'lidar2ego_translation': calibrated_sensor['translation'],
            'lidar2ego_rotation': calibrated_sensor['rotation'],
            'e2g_traslation': ego_pose['translation'],
            'e2g_rotation': ego_pose['rotation'],
            'timestamp': sample['timestamp'],
            'location': location,
            'scene_name': scene_name
        }

        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            sample_data = nusc.get('sample_data', cam_token)
            calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

            cam2ego_translation = np.array(calibrated_sensor['translation'])
            cam2ego_rotation = Quaternion(calibrated_sensor['rotation']).rotation_matrix

            ego2cam_rotation = cam2ego_rotation.T
            ego2cam_translation = ego2cam_rotation.dot(-cam2ego_translation)

            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = ego2cam_rotation
            transform_matrix[:3, 3] = ego2cam_translation

            cam_path = nusc.get_sample_data_path(cam_token)
            cam_info = dict(
                extrinsics = transform_matrix, # ego2cam
                intrinsics = calibrated_sensor['camera_intrinsic'],
                img_fpath = str(cam_path)
            )
            info['cams'][cam] = cam_info

        if scene_name in train_scenes:
            train_samples.append(info)
        elif scene_name in val_samples:
            val_samples.append(info)
        else:
            test_samples.append(info)

    if dest_path is None:
        dest_path = root_path

    if is_test:
        info_path = os.path.join(dest_path, 'nuscenes_map_infos_test.pkl')
        print(f'saving test set to {info_path}')
        mmcv.dump(test_samples, info_path)
    else:
        info_path = os.path.join(dest_path, 'nuscenes_map_infos_train.pkl')
        print(f'saving training set to {info_path}')
        mmcv.dump(train_samples, info_path)

        info_path = os.path.join(dest_path, 'nusenes_map_infos_val.pkl')
        print(f'saving validation set to {info_path}')
        mmcv.dump(val_samples, info_path)


if __name__ == "__main__":
    args = parse_args()
    create_nuscenes_infos_map(root_path=args.data_root, dest_path=args.dest_path, version=args.version)