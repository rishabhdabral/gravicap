import argparse
import torch
import pickle
import cv2
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import os
from parse_3d import get_poses, get_cam_poses
from scipy.optimize import least_squares
# from parse_alphapose_results import parse_alphapose
from utils import initialize, construct_pose
from utils import prepare_2d_pose, prepare_3d_pose, prepare_2d_trajectory
from utils import vnect_to_h36m
from utils import recover_outputs

from objective_functions import objective, objective_pose

lc = 0.1
lg = 1 
lb = 1
lp = 1.0
lr = 1.0 
lm = 0.5
lcont = 0.1
ls = 0.1
lsym = 1.0
indices = [27, 26, 25, 20, 21, 22, 6, 4, 7, 8, 18, 17, 16, 11, 12, 13]
kinematic_chain = [[2,6], [1,2], [0,1], [3,6], [4,3], [5,4],
                [7,6], [8,7], [9,8], [12,8], [11,12], [10,11],
                [13,8], [14,13], [15,14]]
sym_pairs = [[0,3], [1,4], [2,5], [9,12], [10,13], [11,14]]



def get_gt_poses(opts, pose_path, calib_path, proj_mat, extrinsics, episode):
    indices = [27, 26, 25, 20, 21, 22, 6, 4, 7, 8, 18, 17, 16,
                        11, 12, 13]
    poses, p3d, p2d = get_poses(pose_path, calib_path, 98)
    poses_cam_3d = []
    poses_cam_2d = []
    for i in range(episode[0], episode[1]+1):
        p3d, p2d = get_cam_poses(poses, proj_mat, extrinsics, i)
        poses_cam_3d.append(p3d.reshape(-1, 1, 30, 3))
        poses_cam_2d.append(p2d.reshape(-1, 1, 30, 2))

    p3d = np.concatenate(poses_cam_3d, 1)
    p2d = np.concatenate(poses_cam_2d, 1)

    p3d = p3d[opts.cam_no]
    p2d = p2d[opts.cam_no]

    p3d = p3d[:, indices]
    p2d = p2d[:, indices]


    return p2d, p3d

def generate_trajectory(u, b0, a, episode_annots):
    episodes = episode_annots[0]['episodes'] 
    n_frames = episodes[-1][1] - episodes[0][0] + 1
    b3d = np.zeros((n_frames, 3))
    for i in range(len(episode_annots)):
        u_ = u[i]
        b0_ = b0[i]

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
    return b3d


    
def get_pred_poses(opts, gt_p2d, gt_p3d, start, end):
    indices = [27, 26, 25, 20, 21, 22, 6, 4, 7, 8, 18, 17, 16,
                        11, 12, 13]
    if opts.mode == 'gt':
        p2d = gt_p2d
        p3d = gt_p3d
    if opts.mode == 'vnect_alphapose':
        p3d = get_poses(os.path.join(opts.pose_path, 'ddd.mddd'), proj=False, dim=3)
        p3d = vnect_to_h36m(p3d)
        p3d = p3d[start: end]
        p2d, _ = parse_alphapose(f'{annot_dir}/alphapose/{cam_no}/alphapose-results.json')
        p2d = p2d[start: end]
    if opts.mode == 'vnect':
        p3d = get_poses(os.path.join(opts.pose_path, 'ddd.mddd'), proj=False, dim=3)
        p2d = get_poses(os.path.join(opts.pose_path, 'ddd.mdd'), proj=False, dim=2)
        p3d = vnect_to_h36m(p3d)
        p2d = vnect_to_h36m(p2d)
        p3d = p3d[start: end]
        p2d = p2d[start: end]
    if opts.mode == 'vnect_gt':
        p3d = get_poses(os.path.join(opts.pose_path, 'ddd.mddd'), proj=False, dim=3)
        p3d = vnect_to_h36m(p3d)
        p3d = p3d[start:end]
        p2d = gt_p2d

    return p2d, p3d


class Processor():
    def __init__(self, opts):
        self.opts = opts
        self.n_joints = 16

        self.get_annots()
        self.get_calib()
        self.get_poses(opts)
        
    def get_annots(self):
        with open(os.path.join(self.opts.annot_dir, 
                                f'episode{self.opts.eps}',
                                'gt_trajectory.pkl'), 'rb') as f:
           self.annot_dict = pickle.load(f)
        self.episodes = self.annot_dict['episodes'][:]


    def get_calib(self):
        with open(self.opts.calib_path, 'rb') as f:
            cameras = pickle.load(f)
            self.extrinsics = cameras['extrinsics']
            self.intrinsics = cameras['intrinsics']
            self.proj_mat = cameras['projection']
        
        cam_no = self.opts.cam_no
        self.c = [self.intrinsics[cam_no, 0, 2], self.intrinsics[cam_no, 1, 2]]
        self.f = self.intrinsics[cam_no, 0, 0] / 1000
        # Computing the gravity vector, assuming the x-z plane of WCS is
        # placed horizontally.
        origin = [0, 0, 0, 1]
        g_norm = [0, -1, 0, 1]
        g_vec = self.extrinsics @ g_norm - self.extrinsics @ origin
        g_vec = g_vec[:, :3]
        norms = np.linalg.norm(g_vec, 2, 1)
        self.g_vec = g_vec / norms.reshape(-1, 1)

    def get_poses(self, opts):
        self.gt_p2d, self.gt_p3d = get_gt_poses(opts,
                                        opts.gt_pose_path, opts.calib_path, 
                                        self.proj_mat, self.extrinsics, 
                                        [self.episodes[0][0], self.episodes[-1][1]])
        self.p2d, self.p3d = get_pred_poses(
                                        opts, self.gt_p2d, self.gt_p3d,
                                        self.episodes[0][0], self.episodes[-1][1]+1)

    def process(self): 
        contacts = self.annot_dict['contacts']
        traj_start = self.annot_dict['start']

        bl_on = True

        # import pdb; pdb.set_trace()
        rr_p3d = prepare_3d_pose(self.p3d.reshape(-1, self.n_joints,3), bl_on=False, tensor='numpy')
        p3d = prepare_3d_pose(self.p3d.reshape(-1, self.n_joints,3), bl_on, tensor='numpy')
        p2d = prepare_2d_pose(self.p2d, c=self.c, tensor='numpy') 

        # gt_3d = gt_3d[:, indices] / 1000
        gt_3d = self.gt_p3d - self.gt_p3d[:, 6:7]

        #####################################
        # Initialize the parameters
        #####################################

        n_poses = self.episodes[-1][1] - self.episodes[0][0] + 1
        # root_params = np.ones((n_poses*3)) * 3
        root_params = self.p3d[:, 6].reshape(-1) / 1000
        # Cam params = focal length and gravity direction
        cam_params = np.ones((4))
        cam_params[0] = self.f
        cam_params[2] = 9.0
        # pose_paramse: bone-lengths
        # pose_params = np.ones((16*n_poses*3))*0.1
        # pose_params = np.ones((15))*0.1
        pose_params = initialize(rr_p3d)

        # obj_params: object parameters (init velocity and init position)
        init_v = np.ones((3*len(self.episodes)))*0.1
        # init_b = np.ones((3*len(episodes)))*4
        init_b = self.p3d[:len(self.episodes), 6].reshape(-1)/1000
        obj_params = np.concatenate((init_v, init_b))
        # obj_params = np.ones((6*len(episodes)))*0.5

        # Initialized parameters
        init_vec = np.concatenate((cam_params, obj_params, root_params),0)

        # Package the episodes related annotations in a dictionary list
        episode_annots = [] 
        for e in range(len(self.episodes)):
            dataset = np.genfromtxt(os.path.join(opts.annot_dir, f'episode{opts.eps}',
                             f'traj_{opts.cam_no}.txt'), delimiter=',')

            start = self.episodes[e][0] - traj_start
            end = self.episodes[e][1] - traj_start

            b2d_full = prepare_2d_trajectory(dataset[self.episodes[0][0]-traj_start:self.episodes[-1][1]+1-traj_start, 2:4], 
                                                c=self.c, tensor='numpy')

            # Only for edgar pingpong
            # b2d_full = prepare_2d_trajectory(dataset[4:episodes[-1][1]+1-traj_start, 2:4], 
                                                           #  c=c, tensor='numpy')
            dataset = dataset[start:end+1, 2:]
            T = (dataset[:, -1] - dataset[0, -1]) / 25 
            a = 9.81 * self.g_vec[opts.cam_no] 

            annot = {'b2d': b2d_full, 'T':T, 'traj_start': traj_start,
                     'start': start, 'end': end, 'episodes': self.episodes}
            episode_annots.append(annot)


        # import pdb; pdb.set_trace()
        # print(f'starting exp for cam_no {cam_no}, annot: {annot_dir} {eps}')
        res = least_squares(objective, init_vec, method='lm', args=(opts, a, self.f, p2d, rr_p3d, b2d_full, 
                                                T, n_poses, episode_annots, 
                                                contacts, False,))
        # init_vec = np.concatenate((cam_params, obj_params, root_params),0)
        res1 = least_squares(objective_pose, pose_params, method='lm', args=(opts, res.x, a, self.f, p2d, p3d, b2d_full, 
                                                rr_p3d, T, n_poses, episode_annots, 
                                                contacts, bl_on,))
        outputs = res.x
        s, u, root, a_pred, b0 = recover_outputs(outputs, self.episodes, n_poses)
        b3d = generate_trajectory(u, b0, a_pred, episode_annots)
        
        pose_params = res1.x[-15:]
        bl = np.abs(pose_params[:])
        j3d = construct_pose(res1.x, p3d, tensor='numpy')
        '''
        print('scale is ', s, self.f)
        print(a_pred)
        print(u)

        print(root[0])
        print(b0)
        '''

        root_diff = (((self.gt_p3d[:, 6] - root*1000)**2).sum(-1)**0.5).mean() 
        print('root_diff: ', root_diff)
        if opts.mode == 'vnect' or opts.mode == 'vnect_gt' or opts.mode == 'vnect_alphapose':
            gt_diff = (((self.gt_p3d[:, 6] - self.p3d[:, 6])**2).sum(-1)**0.5).mean() 
            print('gt_diff: ', gt_diff)
            mpjpe_vnect = (((gt_3d - rr_p3d*1000)**2).sum(-1)**0.5)
            mpjpe_lm = (((gt_3d - j3d*1000)**2).sum(-1)**0.5)

        object_diff = (((self.annot_dict['traj_cam'][opts.cam_no, :b3d.shape[0], :3] - b3d[:]*1000)**2).sum(-1)**0.5).mean()
        print('object_diff ', object_diff)
        cosine_similarity = np.dot(a, a_pred) / (np.linalg.norm(a) * np.linalg.norm(a_pred))
        vnect_bl = initialize(rr_p3d)
        gt_bl = initialize(gt_3d)
        bl_error = np.abs(gt_bl - bl*1000)
        save_dict = {'cam': cam_params,
                     'pose': pose_params,
                     'root': root_params,
                     'obj': obj_params,
                     'gt_p3d': self.gt_p3d,
                     'traj_cam': self.annot_dict['traj_cam'][opts.cam_no],
                     'episodes': self.episodes,
                     'start': start, 'bl_on': bl_on, 'mode': opts.mode,
                     'cam_no': opts.cam_no, 'lm': opts.lm, 'lc': opts.lc, 'lp': opts.lp, 'lg': opts.lg, 
                     'ls': opts.ls, 'lcont': opts.lcont, 'b3d': b3d,
                     'root_diff': root_diff, 'object_diff': object_diff,
                     'cosine_sim': cosine_similarity, 
                     'bl_error': bl_error, 
                     }

        if opts.mode != 'gt':
            vnect_bl_error = np.abs(vnect_bl*1000 - gt_bl)
            save_dict['mpjpe_lm'] = mpjpe_lm
            save_dict['mpjpe_vnect'] = mpjpe_vnect
            save_dict['vnect_bl_error'] = vnect_bl_error
            save_dict['vnect_bl'] = vnect_bl

            vals = [['eps', 'obj_diff', 'root_diff', 'vnect_diff', 'mpjpe', 'vmjpe',  'cosine', 'bl_error', 'vnect_bl_error']]
            vals.append([opts.eps, opts.cam_no, int(object_diff), int(root_diff), int(gt_diff), 
                int(mpjpe_lm.mean()), int(mpjpe_vnect.mean()), cosine_similarity, int(bl_error.mean()), int(vnect_bl_error.mean())])
            print(vals[0])
            print(vals[1])

        if opts.save_name:
            name = opts.save_name
        else:
            name = f'abl_lm_bl_True_{opts.mode}_cam_{opts.cam_no}_gt_init_known_g_f.pkl'

        with open(os.path.join(opts.annot_dir, f'episode{opts.eps}', name), 'wb') as f:
            pickle.dump(save_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Optimization weights
    parser.add_argument('--lc', default=0.1, 
                           help='Weight for the contact term')
    parser.add_argument('--lb', default=1.0, 
                           help='Weight for the object projection term')
    parser.add_argument('--lp', default=1.0, 
                           help='Weight for the pose projection term')
    parser.add_argument('--lm', default=0.5, 
                           help='Weight for the mutual localization term')
    parser.add_argument('--lg', default=1.0, 
                           help='Weight for the gravity term')
    parser.add_argument('--lcont', default=0.1, 
                           help='Weight for the continuity term')
    parser.add_argument('--ls', default=0.1, 
                           help='Weight for the smoothness term')
    parser.add_argument('--lsym', default=0.1, 
                           help='Weight for the symmetry term')
    # Paths
    parser.add_argument('--calib_path', type=str, default='../calibration_hps.pkl', 
                           help='Path of calibration file')
    parser.add_argument('--pose_path', type=str, 
                           help='Path of predicted pose directory')
    parser.add_argument('--gt_pose_path', type=str, 
                           help='Path of captury pose directory')
    parser.add_argument('--annot_dir', type=str, 
                           help='Path of annotation directory')

    # Inference info
    parser.add_argument('--dof', type=int, default=9,
                           help='The DoF of the optimzation.')
    parser.add_argument('--mode', type=str, default='vnect', 
                           help='Choose between vnect, vnect_alphapose, gt')
    parser.add_argument('--bl_on', action='store_true', 
                    help='Whether bonelength/root-relative pose needs to be optimized')
    parser.add_argument('--eps', type=int, default=0,
                           help='Episode number')
    parser.add_argument('--cam_no', type=int, default=0,
                            help='Camera number')
    parser.add_argument('--save_name', type=str, default='results.pkl', 
                            help='Name of the saved file')

    opts = parser.parse_args()
    engine = Processor(opts)
    engine.process()

