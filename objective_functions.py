import pickle
import sys
import os
import numpy as np
import os

from utils import initialize, construct_pose

lc = 0.1
lg = 1 
lb = 1
lp = 1.0
lre = 1.0 
lm = 0.5
lcont = 0.1
ls = 0.1
lsym = 1.0
indices = [27, 26, 25, 20, 21, 22, 6, 4, 7, 8, 18, 17, 16, 11, 12, 13]
kinematic_chain = [[2,6], [1,2], [0,1], [3,6], [4,3], [5,4],
                [7,6], [8,7], [9,8], [12,8], [11,12], [10,11],
                [13,8], [14,13], [15,14]]
sym_pairs = [[0,3], [1,4], [2,5], [9,12], [10,13], [11,14]]


def objective(x, opts, a, f, p2d, p3d, b2d, T, n_poses, episode_annots, contacts, bl_on=True):
    """
        x: the param vector
        p3d: a numpy array of shape TxJx3
        p2d: a numpy array of shape TxJx2
        b2d: a numpy array of shape Tx2
        T:   a numpy array of shape T
        returns: loss
    """
    residual = []
    episodes = episode_annots[0]['episodes']
    start_frame = episode_annots[0]['traj_start']
    # focal length
    if opts.dof == 10:
        f = np.abs(x[0])
    a = x[1:4]

    obj_params = x[4:4+len(episodes)*6]
    # initial velocity 
    mid_pt = obj_params.shape[0] // 2
    u = obj_params[:mid_pt].reshape(-1, 3)
    # initial position
    b0 = obj_params[mid_pt:].reshape(-1, 3)

    # root of the person
    root_params = x[4+len(episodes)*6: 4+len(episodes)*6+3*n_poses]
    proot = root_params.reshape(-1, 1, 3)

    # absolute pose
    p_abs = p3d + proot # + p_deltas

    # 3D pose projection constraint
    E_p = ((f * p_abs[:, :, :2] / p_abs[:, :, 2:3] - p2d)).reshape(-1)


    E_cont, E_b, E_c = [], 0, 0
    b3d = np.zeros((p2d.shape[0], 3))
    for i in range(len(episode_annots)):
        u_ = u[i]
        b0_ = b0[i]
        if i > 0: 
            E_cont.append((b3d_[-1] - b0[i]).reshape(-1))

        T = episode_annots[i]['T'].reshape(-1, 1)
        init_frame = episodes[i][0] - episodes[0][0]
        end_frame = episodes[i][1] - episodes[0][0]

        # Trajectory of the ball
        b3d_ = b0_ + u_*T + 0.5*a*T**2
        b3d[init_frame: end_frame+1] += b3d_

        # The value of the last frame/first frame must be averaged across
        # trajectories
        if i > 0:
            b3d[init_frame] /= 2.0

    if len(E_cont) > 0:
        E_cont = np.concatenate(E_cont).squeeze()
    else:
        E_cont = np.array(E_cont)

    # 3D Ball projection constraint
    E_b = (f * b3d[:, :2] / b3d[:, 2:3] - b2d).reshape(-1)

    # TT racket extended arm
    E_c = []
    for contact in contacts:
        f_no = contact[0] - episodes[0][0]
        if f_no > p_abs.shape[0]:
            continue

        E_c.append((p_abs[f_no, contact[1]] - b3d[f_no]).reshape(-1))
 
    if len(E_c) > 0:
        E_c = np.concatenate(E_c).squeeze()
    else:
        E_c = np.array(E_c)
    
    # import pdb; pdb.set_trace()
    E_m = 0
    E_m = mutual_distance_loss(f, 
                               p_abs,
                               p2d,
                               b3d, b2d, [2,3,6,7,8])


    E_r = [np.linalg.norm(a,2) - 9.81]
    # E_r = [0]

    E_r = np.array(E_r)

    E_sym = []
    if bl_on is True:
        for j, pair in enumerate(sym_pairs):
            l_bone = pair[0]
            r_bone = pair[1]
            E_sym.append(bl[l_bone] - bl[r_bone])

    E_sym = np.array(E_sym)

    T = T[1:]
    E_sm = smoothness_constraint(proot, 5)

    res = np.concatenate((opts.lc*E_c, opts.lp*E_p, opts.lb*E_b, opts.lm*E_m, opts.lg*E_r, opts.ls*E_sm, opts.lcont*E_cont))
    # print(f, a[1], a[2])
    # print((E_c**2).mean(),( E_p**2).mean(), (E_b**2).mean(), (E_m**2).mean())
    return res

def objective_pose(y, opts, x, a, f, p2d, p3d, b2d, rr_p3d, T, n_poses, episode_annots, contacts, bl_on=True):
    """
        x: the param vector
        p3d: a numpy array of shape TxJx3
        p2d: a numpy array of shape TxJx2
        b2d: a numpy array of shape Tx2
        T:   a numpy array of shape T
        returns: loss
    """
    residual = []
    episodes = episode_annots[0]['episodes']
    start_frame = episode_annots[0]['traj_start']
    # focal length
    if opts.dof == 10:
        f = np.abs(x[0])
    a = x[1:4]

    obj_params = x[4:4+len(episodes)*6]
    # initial velocity 
    mid_pt = obj_params.shape[0] // 2
    u = obj_params[:mid_pt].reshape(-1, 3)
    # initial position
    b0 = obj_params[mid_pt:].reshape(-1, 3)

    root_params = x[4+len(episodes)*6: 4+len(episodes)*6+3*n_poses]
    # root of the person
    proot = root_params.reshape(-1, 1, 3)

    # bone lengths and the corresponding pose
    bl = np.abs(y[:])
    p3d = construct_pose(bl, p3d, tensor='numpy')

    # absolute pose
    p_abs = p3d + proot

    # 3D pose projection constraint
    E_p = ((f * p_abs[:, :, :2] / p_abs[:, :, 2:3] - p2d)).reshape(-1)

    # E_d = deltas.reshape(-1)

    E_d = ((p3d - rr_p3d)).reshape(-1)

    E_cont, E_b, E_c = [], 0, 0
    b3d = np.zeros((p2d.shape[0], 3))

    for i in range(len(episode_annots)):
        u_ = u[i]
        b0_ = b0[i]
        if i > 0: 
            E_cont.append((b3d_[-1] - b0[i]).reshape(-1))

        T = episode_annots[i]['T'].reshape(-1, 1)
        init_frame = episodes[i][0] - episodes[0][0]
        end_frame = episodes[i][1] - episodes[0][0]
        # Trajectory of the ball
        b3d_ = b0_ + u_*T + 0.5*a*T**2
        b3d[init_frame: end_frame+1] += b3d_

        # The value of the last frame/first frame must be averaged across
        # trajectories
        if i > 0:
            b3d[init_frame] /= 2.0

    # E_cont = np.concatenate(E_cont)
    # 3D Ball projection constraint
    # E_b = (f * b3d[:, :2] / b3d[:, 2:3] - b2d).reshape(-1)

    # TT racket extended arm
    E_c = []
    for contact in contacts:
        f_no = contact[0] - episodes[0][0]
        if f_no > p_abs.shape[0]:
            continue

        E_c.append((p_abs[f_no, contact[1]] - b3d[f_no]).reshape(-1))
 
    if len(E_c) > 0:
        E_c = np.concatenate(E_c).squeeze()
    else:
        E_c = np.array(E_c)
    
    # import pdb; pdb.set_trace()
    E_m = 0
    E_m = mutual_distance_loss(f, 
                               p_abs,
                               p2d,
                               b3d, b2d, [2,3,6,7,8])


    E_r = [np.linalg.norm(a,2) - 9.81]
    E_r = np.array(E_r)

    E_sym = []
    if bl_on is True:
        for j, pair in enumerate(sym_pairs):
            l_bone = pair[0]
            r_bone = pair[1]
            E_sym.append(bl[l_bone] - bl[r_bone])

    E_sym = np.array(E_sym)

    T = T[1:]
    E_sm = smoothness_constraint(proot, 5)

    res = np.concatenate((opts.lc*E_c, opts.lp*E_p, opts.lm*E_m, opts.ls*E_sm, 0.1*E_d, opts.lsym*E_sym))
    # print(f, a[1], a[2], bl[0], bl[1])
    # print((E_c**2).mean(),( E_p**2).mean(), (E_m**2).mean(),(E_d**1).mean())
    return res


def smoothness_constraint(roots, window=5):
    '''
        roots: n x 3
    '''
    roots = roots.squeeze()
    n_frames = roots.shape[0]
    E_sm = []
    for i in range(roots.shape[0]-1):
       E_sm.append((roots[i] - roots[i+1]).reshape(-1))
    E_sm = np.concatenate(E_sm)
    return E_sm
    '''
    # Pad roots
    pad_front = torch.zeros(window // 2, roots.shape[1]).cuda() + roots[0]
    pad_end = torch.zeros(window // 2, roots.shape[1]).cuda() + roots[-1]
    
    roots_padded = torch.cat((pad_front, roots, pad_end), 0)
    E_sm = 0
    
    for i in range(window//2, window//2 + roots.shape[0]):
        start = i - window//2
        end = i + window//2 + 1
        E_sm += ((roots_padded[i] - roots_padded[start:end].mean(0))**2).sum() / roots.shape[0]

    '''

    return E_sm
    

def mutual_distance_loss(f, p_abs, p2d, b3d, b2d, joints, n_samples=20):
    '''
        Implements the mutual distance loss. The vector between the
        object's position and the set of joints in the absolute human
        pose should project to the corresponding vector in the 2D image.
        Parameters:
            f: a scalar for the focal length
            p_abs: a tensor of shape n_frames x n_joints x 3
            p2d: a tensor of shape n_frames x n_joints x 2
            b3d:  a tensor of shape n_frames x 3
            b2d:  a tensor of shape n_frames x 2
            joints: a list of joints to be considered for the loss
            n_samples: number of points to be sampled between the object
                       and the joint position
        returns:
            loss: the mutual distance loss
    '''
    b3d = b3d.reshape(-1, 1, 3)
    b2d = b2d.reshape(-1, 1, 2)

    n_frames = p_abs.shape[0] 
    assert p_abs.shape[0] == p2d.shape[0] and b3d.shape[0] == b2d.shape[0]
    assert p_abs.shape[0] == b3d.shape[0]

    vec_3d = b3d - p_abs[:, joints] 
    vec_2d = b2d - p2d[:, joints] 

    '''
    pts_3d = vec_3d.unsqueeze(2).expand(
                                    n_frames, len(joints), n_samples, 3)
    pts_2d = vec_2d.unsqueeze(2).expand(
                                    n_frames, len(joints), n_samples, 2)
    '''

    E_m = []
    for i in range(1, n_samples):
        pt_3d = p_abs[:, joints] + vec_3d * i / n_samples
        pt_2d =  p2d[:, joints] + vec_2d * i / n_samples
        E_m.append((f * pt_3d[..., :2] / pt_3d[..., 2:3] - pt_2d).reshape(-1))
    E_m = np.concatenate(E_m)

    return E_m



