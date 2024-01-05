import mujoco_py as mj
from PIL import Image
import dm_env
import numpy as np
import open3d as o3d
import math

class MultiCamPCD():
    def __init__(self, cameras, width=200, height=200, depth=True):
        self.cameras = cameras
        self.width = width
        self.height = height
        self.depth = depth


    def render_rgb_depth(self, xml_path): # returns [rgb_array, scaled depth_array]
        self.model = mj.load_model_from_path(xml_path)
        self.sim = mj.MjSim(self.model)

        a = self.sim.render(self.width, self.height, camera_name=self.cameras, depth=True)
        rgb_array = a[0]
        depth_array = a[1]

        print(depth_array)
        print(depth_array.shape)

        rgb_img = Image.fromarray(rgb_array)
        rgb_img.save("rgb_depth_images/rgb_img.png")
        rgb_img.show()

        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent

        # near = 3 # Remove this line for accurate Meters representation. It is just here so I can visualize the images better

        print(extent)
        print(near)
        print(far)

        depth_array = (np.vectorize(self.depthimg2Meters)(near, far, depth_array))

        print(depth_array)
        print(depth_array.dtype)
        print(depth_array[0].dtype)
        print((depth_array[0][0]))
        print(depth_array.shape)

        depth_img = Image.fromarray(depth_array)
        depth_img = depth_img.convert('RGB')
        depth_img.save("rgb_depth_images/depth_img.png")
        depth_img.show()

        return [rgb_array, depth_array]

    def depthimg2Meters(self, near, far, depth):
        image = near / (1 - depth * (1 - near / far))
        return image
    
    def make_RGBD(self, rgb, depth):
        rgb_layer = o3d.geometry.Image(rgb)
        depth_layer = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_layer, depth_layer)
        return rgbd
    
    def make_point_cloud(self, rgbd):
        fovy = self.sim.model.cam_fovy[self.model.camera_name2id(self.cameras)]
        # print(fovy) # check camera fov
        f = 0.5 * self.height / math.tan(fovy * math.pi / 360)
        pinhole_cam = o3d.camera.PinholeCameraIntrinsic(
            width = self.width,
            height = self.height,
            intrinsic_matrix = np.array(((f, 0, self.width / 2), (0, f, self.height / 2), (0, 0, 1)))
            )
        print(pinhole_cam.intrinsic_matrix) # Check intrinsic matrix
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_cam)
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd_actual = o3d.io.write_point_cloud("rgb_depth_images/pointcloud2.ply", pcd)
        return pcd
    
    def multicam_point_cloud(self):
        for cam in cameras:
            num = 0

