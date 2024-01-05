import mujoco_py as mj
from PIL import Image
import dm_env
import numpy as np
import open3d as o3d
import math
from datetime import datetime
import os


class MultiCamPCD():
    def __init__(self, xml_path, cameras, width=200, height=200, depth=True):
        self.xml_path = xml_path
        self.cameras = cameras
        self.width = width
        self.height = height
        self.depth = depth


    def render_rgb_depth(self, cam_name): # returns [rgb_array, scaled depth_array]
        self.model = mj.load_model_from_path(self.xml_path)
        self.sim = mj.MjSim(self.model)

        a = self.sim.render(self.width, self.height, camera_name=cam_name, depth=True)
        rgb_array = a[0]
        depth_array = a[1]

        # print(depth_array)
        # print(depth_array.shape)

        '''
        Commented lines below show and save RGB image
        '''
        # rgb_img = Image.fromarray(rgb_array)
        # rgb_img.save("rgb_depth_images/rgb_img.png")
        # rgb_img.show()

        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent

        # near = 3 # Remove this line for accurate Meters representation. It is just here so I can visualize the images better

        # print(extent)
        # print(near)
        # print(far)

        depth_array = (np.vectorize(self.depthimg2Meters)(near, far, depth_array))

        # print(depth_array)
        # print(depth_array.dtype)
        # print(depth_array[0].dtype)
        # print((depth_array[0][0]))
        # print(depth_array.shape)

        '''
        Commented lines below show and save depth image
        '''

        # depth_img = Image.fromarray(depth_array)
        # depth_img = depth_img.convert('RGB')
        # depth_img.save("rgb_depth_images/depth_img.png")
        # depth_img.show()

        return [rgb_array, depth_array]

    def depthimg2Meters(self, near, far, depth):
        image = near / (1 - depth * (1 - near / far))
        return image
    
    def make_RGBD(self, rgb, depth):
        rgb_layer = o3d.geometry.Image(rgb)
        depth_layer = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_layer, depth_layer)
        return rgbd
    
    def single_pcd(self, rgbd_img, cam_name):
        fovy = self.sim.model.cam_fovy[self.model.camera_name2id(cam_name)]
        # print(fovy) # check camera fov
        f = 0.5 * self.height / math.tan(fovy * math.pi / 360)
        pinhole_cam = o3d.camera.PinholeCameraIntrinsic(
            width = self.width,
            height = self.height,
            intrinsic_matrix = np.array(((f, 0, self.width / 2), (0, f, self.height / 2), (0, 0, 1)))
            )
        # print(pinhole_cam.intrinsic_matrix) # Check intrinsic matrix
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, pinhole_cam)
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return pcd
    
    def multicam_point_cloud(self):
        time = str(datetime.today())
        save_path = f"rgb_depth_images/point_clouds/{time[:10]}"
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        if os.path.isdir(f"{save_path}/{time[11:19]}") == False:
            os.mkdir(f"{save_path}/{time[11:19]}")
        for cam in self.cameras:
            print(f"Generating point cloud for camera: {cam}")
            rgbd = self.render_rgb_depth(cam)
            rgbd_image = self.make_RGBD(rgbd[0], rgbd[1])
            pcd = self.single_pcd(rgbd_image, cam)
            o3d.io.write_point_cloud(f"{save_path}/{time[11:19]}/{cam}.ply", pcd)
            
        return len(self.cameras)



