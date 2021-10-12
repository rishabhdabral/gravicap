import numpy as np
import os
import sys
import pickle


kinematic_chain = [[2,6], [1,2], [0,1], [3,6], [4,3], [5,4],
                [7,6], [8,7], [9,8], [12,8], [11,12], [10,11],
                [13,8], [14,13], [15,14]]

def construct_pose(bl, p3d, tensor='torch'):
    if tensor == 'torch':
        out = torch.zeros(p3d.shape).cuda()
    else:
        out = np.zeros((p3d.shape))
    for i, bone in enumerate(kinematic_chain):
        child = bone[0]
        parent = bone[1]
        vecnorm = p3d[:, child]
        out[:, child] = out[:, parent] + bl[i] * vecnorm
    
    return out

def initialize(poses_3d):
    '''
        Converts a set of 3D poses into appropriate initializations
        Arguments:
            poses_3d: K x J x 3
        Returns:
            init_len: a list of length J
    '''
    K = poses_3d.shape[0]
    len_3d = np.zeros((poses_3d.shape[1]-1))
    # dzs = np.zeros((K, J))
    for j, bone in enumerate(kinematic_chain):
        b1 = bone[0] # child
        b2 = bone[1] # parent
        len_3d[j] = (((poses_3d[:, b1] - poses_3d[:, b2] )**2).sum(-1)**0.5).mean()

    return len_3d

def vnect_to_h36m(pose):
    '''
        convert vnect joints to h36m lik format
    '''
    vnect_to_h36m_map = [10, 9, 8, 11, 12, 13, 14, 14, 1, 0, 4, 3, 2, 5, 6, 7]
    out = pose[:, vnect_to_h36m_map]
    out[:, 7] = (out[:, 6] + out[:, 8]) / 2
    return out

def recover_outputs(outputs, episodes, n_poses):
    '''
        parse the outputs of the optimizer
    '''
    cam_params = outputs[:4]
    obj_params = outputs[4:4+len(episodes)*6]

    mid_pt = obj_params.shape[0] // 2
    # initial velocity 
    u = obj_params[:mid_pt].reshape(-1, 3)
    # initial position
    b0 = obj_params[mid_pt:].reshape(-1, 3) 

    root_params = outputs[4+len(episodes)*6: 4+len(episodes)*6+3*n_poses] 
    s = cam_params[0]
    root = root_params.reshape(-1, 3)
    a_pred = cam_params[1:4]
    return s, u, root, a_pred, b0


def prepare_3d_pose(p3d, bl_on=True, tensor='torch'):
    ''' Preprocesses the 3D poses
        Arguments:
            pose: n x n_joints x 3 
        Returns:
            p3d: n x n_joints x 3
    '''
    p3d = p3d / 1000
    p3d = p3d - p3d[:, 6:7]
    out = p3d.copy()
    for bone in kinematic_chain:
        parent = bone[1]
        child = bone[0]
        direction_vec = p3d[:, child] - p3d[:, parent]
        norm_vec = direction_vec / np.linalg.norm(direction_vec, 2, 1).reshape(-1, 1)
        out[:, child] = norm_vec

    if tensor == 'numpy':
        if bl_on is True:
            return out
        else:
            return p3d

    if bl_on is True:
        p3d = torch.from_numpy(out).float().cuda()
    else:
        p3d = torch.from_numpy(p3d).float().cuda() 
    
    return p3d

def prepare_2d_pose(p2d, c=[600, 438], tensor='torch'):
    ''' pose: n x n_joints x 2 '''
    p2d = p2d - c
    p2d = p2d / 1000
    if tensor == 'numpy':
        return p2d
    p2d = torch.from_numpy(p2d).float().cuda() 
    return p2d

def prepare_2d_trajectory(b2d, c=[600, 438], tensor='torch'):
    ''' b2d: n x 2 '''
    b2d = b2d - c
    # b2d[:, 0] = (b2d[:, 0] - 603) 
    # b2d[:, 1] = (b2d[:, 1] - 440) 
    b2d = b2d / 1000
    if tensor == 'numpy':
        return b2d
    b2d = torch.from_numpy(b2d).float().cuda()
    return b2d
