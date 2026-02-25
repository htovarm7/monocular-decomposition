import os
import sys
import numpy as np
import torch
import cv2
import open3d as o3d

sys.path.append("./src/core_multilayers")
from raft_mvs_multilayers import RAFTMVS_2Layer
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
from D415_camera import CameraMgr


def load_image(imfile):
    img = cv2.imread(imfile, 1)
    img = img[:, :, ::-1].copy()
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()


def load_projmatrix_render_d415():
    cameraMgr = CameraMgr()
    calib_rgb = cameraMgr.sim_d415_rgb()
    calib_ir = cameraMgr.sim_d415_ir()
    pose_rgb = cameraMgr.sim_d415_rgb_pose()
    pose_ir1 = cameraMgr.sim_d415_ir1_pose()
    pose_ir2 = cameraMgr.sim_d415_ir2_pose()

    depth_min = torch.tensor(0.2)
    depth_max = torch.tensor(1.50)

    poses = np.stack([pose_rgb, pose_ir1, pose_ir2], 0).astype(np.float32)
    intrinsics = np.stack([calib_rgb, calib_ir, calib_ir], 0).astype(np.float32)

    poses = torch.from_numpy(poses)
    intrinsics = torch.from_numpy(intrinsics)

    proj = poses.clone()
    proj[:, :3, :4] = torch.matmul(intrinsics, poses[:, :3, :4])

    return proj[None].cuda(), depth_min[None].cuda(), depth_max[None].cuda()


class TransparentObjectDetector:
    """Detects transparent objects using MVS 2-layer depth estimation."""

    def __init__(self, args):
        self.args = args

        # Load MVS model
        mvs_net = torch.nn.DataParallel(RAFTMVS_2Layer(args)).cuda()
        mvs_net.load_state_dict(torch.load(args.restore_ckpt))
        mvs_net = mvs_net.module
        mvs_net.eval()
        self.mvs_net = mvs_net

        # Camera setup
        intrinsic = CameraMgr().sim_d415_rgb()
        self.camera = CameraInfo(640.0, 360.0, intrinsic[0][0], intrinsic[1][1],
                                 intrinsic[0][2], intrinsic[1][2], 1.0)
        self.proj_matrices, self.dmin, self.dmax = load_projmatrix_render_d415()

    def infer(self, rgb_path, ir1_path, ir2_path, depth_diff_threshold=0.01):
        """
        Detect transparent objects from RGB + stereo IR images.

        Args:
            rgb_path: path to RGB image
            ir1_path: path to left IR image
            ir2_path: path to right IR image
            depth_diff_threshold: minimum depth difference (m) between layers
                                  to consider a region as transparent

        Returns:
            dict with keys:
                - 'foreground_cloud': point cloud of layer 0 (all objects)
                - 'background_cloud': point cloud of layer 1 (behind transparent)
                - 'transparent_cloud': points identified as transparent objects
                - 'transparent_mask': 2D binary mask of transparent regions
                - 'depth_layer0': foreground depth map
                - 'depth_layer1': background completion depth map
        """
        with torch.no_grad():
            color = load_image(rgb_path)
            ir1 = load_image(ir1_path)
            ir2 = load_image(ir2_path)

            depth_up = self.mvs_net(color, ir1, ir2,
                                    self.proj_matrices.clone(),
                                    self.dmin, self.dmax,
                                    iters=self.args.valid_iters,
                                    test_mode=True)
            depth_up = depth_up.squeeze()
            depth_2layer = depth_up.detach().cpu().numpy().squeeze()

        # Layer 0: foreground (transparent surfaces included)
        depth0 = depth_2layer[0]
        # Layer 1: background completion (what's behind transparent objects)
        depth1 = depth_2layer[1]

        # Valid depth range
        valid_mask0 = (depth0 > 0.25) & (depth0 < 1.0)
        valid_mask1 = (depth1 > 0.25) & (depth1 < 1.0)

        # Transparent object mask: regions where background is significantly
        # deeper than foreground (light passes through the object)
        transparent_mask = (valid_mask0 & valid_mask1 &
                           (depth1 - depth0 > depth_diff_threshold))

        # Generate point clouds
        cloud0 = create_point_cloud_from_depth_image(depth0, self.camera, organized=True)
        cloud1 = create_point_cloud_from_depth_image(depth1, self.camera, organized=True)

        # Get color for visualization
        color_np = color.squeeze().detach().cpu().numpy()
        color_np = np.transpose(color_np, [1, 2, 0])

        # Foreground cloud (all valid points in layer 0)
        fg_points = cloud0[valid_mask0]
        fg_colors = color_np[valid_mask0]

        # Background cloud (valid layer 1 points)
        bg_points = cloud1[valid_mask1]
        bg_colors = color_np[valid_mask1]

        # Transparent object cloud (foreground points in transparent regions)
        trans_points = cloud0[transparent_mask]
        trans_colors = color_np[transparent_mask]

        print(f"Foreground points: {len(fg_points)}")
        print(f"Background points: {len(bg_points)}")
        print(f"Transparent object points: {len(trans_points)}")
        print(f"Transparent pixel ratio: {transparent_mask.sum() / valid_mask0.sum():.2%}")

        return {
            'foreground_cloud': fg_points,
            'foreground_colors': fg_colors,
            'background_cloud': bg_points,
            'background_colors': bg_colors,
            'transparent_cloud': trans_points,
            'transparent_colors': trans_colors,
            'transparent_mask': transparent_mask,
            'depth_layer0': depth0,
            'depth_layer1': depth1,
        }

    def visualize(self, result, mode='transparent'):
        """
        Visualize detection results with Open3D.

        Args:
            result: dict from infer()
            mode: 'transparent' - only transparent objects (highlighted)
                  'all'         - full scene with transparent objects colored red
                  'layers'      - both depth layers side by side
        """
        geometries = []

        if mode == 'transparent':
            # Show only transparent object points in red
            if len(result['transparent_cloud']) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(result['transparent_cloud'])
                pcd.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(pcd)
                print("Showing transparent object points (red)")
            else:
                print("No transparent objects detected.")
                return

        elif mode == 'all':
            # Full foreground cloud with transparent regions highlighted in red
            pcd_fg = o3d.geometry.PointCloud()
            pcd_fg.points = o3d.utility.Vector3dVector(result['foreground_cloud'])
            colors = result['foreground_colors'].astype(np.float64) / 255.0
            pcd_fg.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(pcd_fg)

            # Overlay transparent points in red
            if len(result['transparent_cloud']) > 0:
                pcd_trans = o3d.geometry.PointCloud()
                pcd_trans.points = o3d.utility.Vector3dVector(result['transparent_cloud'])
                pcd_trans.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(pcd_trans)

        elif mode == 'layers':
            # Layer 0 (foreground) in original colors
            pcd0 = o3d.geometry.PointCloud()
            pcd0.points = o3d.utility.Vector3dVector(result['foreground_cloud'])
            colors0 = result['foreground_colors'].astype(np.float64) / 255.0
            pcd0.colors = o3d.utility.Vector3dVector(colors0)
            geometries.append(pcd0)

            # Layer 1 (background) in blue
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(result['background_cloud'])
            pcd1.paint_uniform_color([0.0, 0.4, 1.0])
            geometries.append(pcd1)
            print("Foreground: original colors | Background: blue")

        o3d.visualization.draw_geometries(geometries,
                                          window_name=f"Transparent Object Detection ({mode})")

    def save_depth_visualizations(self, result, output_dir='./output'):
        """Save depth maps and transparent mask as images."""
        os.makedirs(output_dir, exist_ok=True)

        depth0 = result['depth_layer0']
        depth1 = result['depth_layer1']
        trans_mask = result['transparent_mask']

        # Normalize depths for visualization
        valid = depth0 > 0
        if valid.any():
            dmin, dmax = depth0[valid].min(), depth0[valid].max()
            depth0_vis = np.zeros_like(depth0)
            depth0_vis[valid] = (depth0[valid] - dmin) / (dmax - dmin + 1e-6)
            depth0_vis = (depth0_vis * 255).astype(np.uint8)
            depth0_color = cv2.applyColorMap(depth0_vis, cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(output_dir, 'depth_layer0.png'), depth0_color)

        valid1 = depth1 > 0
        if valid1.any():
            dmin1, dmax1 = depth1[valid1].min(), depth1[valid1].max()
            depth1_vis = np.zeros_like(depth1)
            depth1_vis[valid1] = (depth1[valid1] - dmin1) / (dmax1 - dmin1 + 1e-6)
            depth1_vis = (depth1_vis * 255).astype(np.uint8)
            depth1_color = cv2.applyColorMap(depth1_vis, cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(output_dir, 'depth_layer1.png'), depth1_color)

        # Transparent mask
        mask_vis = (trans_mask.astype(np.uint8)) * 255
        cv2.imwrite(os.path.join(output_dir, 'transparent_mask.png'), mask_vis)

        print(f"Saved visualizations to {output_dir}/")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Transparent Object Detection via MVS 2-Layer Depth')

    # MVS architecture args
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--valid_iters', type=int, default=16)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument('--shared_backbone', action='store_true')
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--context_norm', type=str, default="batch",
                        choices=['group', 'batch', 'instance', 'none'])
    parser.add_argument('--slow_fast_gru', action='store_true')
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--num_sample', type=int, default=96)
    parser.add_argument('--depth_min', type=float, default=0.2)
    parser.add_argument('--depth_max', type=float, default=1.5)
    parser.add_argument('--train_2layer', default=True)
    parser.add_argument('--restore_ckpt', default='./checkpoints/raftmvs_2layer.pth')

    # Detection args
    parser.add_argument('--depth_diff_threshold', type=float, default=0.01,
                        help='Min depth difference (m) between layers to detect transparency')
    parser.add_argument('--vis_mode', type=str, default='all',
                        choices=['transparent', 'all', 'layers'],
                        help='Visualization mode')
    parser.add_argument('--save_images', action='store_true',
                        help='Save depth maps and mask to ./output/')

    # Input paths
    parser.add_argument('--rgb', default='./test_data/00100_0000_color.png')
    parser.add_argument('--ir1', default='./test_data/00100_0000_ir_l.png')
    parser.add_argument('--ir2', default='./test_data/00100_0000_ir_r.png')

    args = parser.parse_args()

    detector = TransparentObjectDetector(args)
    result = detector.infer(args.rgb, args.ir1, args.ir2,
                           depth_diff_threshold=args.depth_diff_threshold)

    if args.save_images:
        detector.save_depth_visualizations(result)

    detector.visualize(result, mode=args.vis_mode)
