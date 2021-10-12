import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

def eliminate_duplicates(pose):
    '''
        pose: n x 3
    '''
    clean_pose = []
    for i in range(pose.shape[0]):
        if i == 0:
            curr = pose[i]
            continue
        if np.abs((curr - pose[i])).sum() >  0.0:
            clean_pose.append(pose[i])
            curr = pose[i]

    clean_pose = np.array(clean_pose) 
        
    return clean_pose


def get_poses(pose_path, calib_file_path='', frame_no=0, proj=True, dim=3):

    f = open(pose_path, 'r')
    lines = f.readlines()
    poses = []
    mdd_type = ''
    for line in lines:
        if line.startswith('Skeletool'):
            mdd_type = 'captury'
            continue
        if line.startswith('VNECT'):
            mdd_type = 'vnect'
            continue
        row = line.strip('\n').split(',')
        if mdd_type == 'captury':
            row = list(map(float, row[2:])) # Cuz captury mddd files have an extra ,,
        elif mdd_type == 'vnect':
            row = list(map(float, row[1:]))

        pose = np.array(row).reshape(-1, dim)
        if mdd_type == 'captury':
            pose = eliminate_duplicates(pose)

        # import pdb; pdb.set_trace()
        # pose = np.unique(pose, axis=0)
        poses.append(pose.reshape(1, -1, dim))

    # import pdb; pdb.set_trace()
    # poses = np.array(poses)
    poses = np.concatenate(poses)

    if proj and dim == 3:
        with open(calib_file_path, 'rb') as f:
            data = pickle.load(f)

        proj_mat = data['projection'][:, :3, :]
        extrinsics = data['extrinsics']
        pose_cam, kp_cam = get_cam_poses(poses, proj_mat, extrinsics, frame_no)
        return poses, pose_cam[..., :3], kp_cam[..., :2]
    else:
        return poses

def get_cam_poses(poses, proj_mat, extrinsics, frame_no=0):
    kp_cam = proj_mat @ np.vstack((poses[frame_no].transpose(), np.ones((1, poses.shape[1]))))
    kp_cam = kp_cam.transpose(0, 2, 1)
    kp_cam = kp_cam / kp_cam[..., 2:3]
    pose_cam = extrinsics @  np.vstack((poses[frame_no].transpose(), np.ones((1, poses.shape[1]))))
    pose_cam = pose_cam.transpose(0, 2, 1)

    return pose_cam[..., :3], kp_cam[..., :2]


if __name__ == "__main__":
    path = sys.argv[1]
    calib_file_path = sys.argv[2]
    with open(calib_file_path, 'rb') as f:
        data = pickle.load(f)

    proj_mat = data['projection'][:, :3, :]
    extrinsics = data['extrinsics']

    f = open(path, 'r')
    lines = f.readlines()
    poses = []
    for line in lines:
        if line.startswith('Skeletool'):
            continue
        row = line.strip('\n').split(',')
        row = list(map(float, row[2:]))
        pose = np.array(row).reshape(-1, 3)
        import pdb; pdb.set_trace()
        pose = np.unique(pose, axis=0)
        poses.append(pose)

    import pdb; pdb.set_trace()
    poses = np.array(poses)
    kp_cam = proj_mat @ np.vstack((poses[79].transpose(), np.ones((1, poses.shape[1]))))
    kp_cam = kp_cam.transpose(0, 2, 1)
    kp_cam = kp_cam / kp_cam[..., 2:3]
    pose_cam = extrinsics @  np.vstack((poses[79].transpose(), np.ones((1, poses.shape[1]))))
    pose_cam = pose_cam.transpose(0, 2, 1)

