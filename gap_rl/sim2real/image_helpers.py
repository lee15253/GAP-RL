from typing import Tuple, Optional

import numpy as np
import time
import cv2
import pyrealsense2 as rs
from matplotlib import pyplot as plt

WIDTH, HEIGHT = 640, 360


class RealsenseCamera:
    def __init__(self, camera_sn, hw=None, rs_ratio=1.0):
        if hw is None:
            hw = [360, 640]
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(camera_sn)
        config.enable_stream(rs.stream.depth, hw[1], hw[0], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, hw[1], hw[0], rs.format.rgb8, 30)
        # config.enable_stream(rs.stream.infrared, 1, hw[1], hw[0], rs.format.y8, 30)
        # config.enable_stream(rs.stream.infrared, 2, hw[1], hw[0], rs.format.y8, 30)

        self.rs_ratio = rs_ratio
        self.rs_size = (int(hw[1] * rs_ratio), int(hw[0] * rs_ratio))
        # set align mode
        align_to = rs.stream.color  # depth align to color
        # align_to = rs.stream.depth  # color align to depth
        self.align = rs.align(align_to)

        self.cfg = self.pipeline.start(config)
        time.sleep(2)  # wait for pipeline fully start
        print('Realsense camera ready!')

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.cfg.get_device().first_depth_sensor()
        # print(depth_sensor.get_supported_options())
        # depth_sensor.set_option(rs.option.auto_exposure_priority, True)
        depth_sensor.set_option(rs.option.exposure, 9000.000)  # 1732.000, 7320.000, 12000
        print("depth sensor exposure: ", depth_sensor.get_option(rs.option.exposure))

        color_sensor = self.cfg.get_device().first_color_sensor()
        # print(color_sensor.get_supported_options())
        # color_sensor.set_option(rs.option.auto_exposure_priority, True)
        color_sensor.set_option(rs.option.exposure, 477.000)  # 200 # 170
        print("color sensor exposure: ", color_sensor.get_option(rs.option.exposure))

        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)

        ###################################################################################
        # realsense-viewer, postprocess order：
        # decimation_filter --> HDR Merge --> threshold_filter --> Depth to Disparity --> spatial_filter
        # --> temporal_filter --> Disparity to Depth
        g_rs_downsample_filter = rs.decimation_filter(magnitude=1)  # 2 ** 1, downsampling rate
        g_rs_thres_filter = rs.threshold_filter(min_dist=0.1, max_dist=1.0)
        g_rs_depth2disparity_trans = rs.disparity_transform(True)
        g_rs_spatical_filter = rs.spatial_filter(
            magnitude=2, smooth_alpha=0.5, smooth_delta=20, hole_fill=0,
        )
        g_rs_templ_filter = rs.temporal_filter(
            smooth_alpha=0.1, smooth_delta=40.0, persistence_control=3
        )
        g_rs_disparity2depth_trans = rs.disparity_transform(False)
        self.g_rs_depth_postprocess_list = [
            g_rs_downsample_filter,
            g_rs_thres_filter,
            g_rs_depth2disparity_trans,
            g_rs_spatical_filter,
            g_rs_templ_filter,
            g_rs_disparity2depth_trans,
        ]
        ###################################################################################

        self.colorizer = rs.colorizer()

    def __del__(self):
        self.pipeline.stop()

    def get_image(self, color_depth=True):
        try:
            while True:
                # get frame
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)  # align image
                # convert rgb to np.array
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    continue

                # filter depth
                depth_frame_filter = depth_frame
                for trans in self.g_rs_depth_postprocess_list:
                    depth_frame_filter = trans.process(depth_frame_filter)
                depth_frame = depth_frame_filter

                image_rgb = np.asanyarray(color_frame.get_data())
                image_depth = np.asanyarray(depth_frame.get_data())
                # colorized_depth = np.asanyarray(
                #     self.colorizer.colorize(depth_frame).get_data()
                # )
                rs_rgb = cv2.resize(
                    image_rgb, self.rs_size, interpolation=cv2.INTER_NEAREST
                )
                rs_depth = cv2.resize(
                    image_depth, self.rs_size, interpolation=cv2.INTER_NEAREST
                )
                if color_depth:
                    rs_depth = colorize(rs_depth, clipping_range=(None, 5000), colormap=cv2.COLORMAP_JET)  # cv2.COLORMAP_HSV
                return rs_rgb / 255.0, rs_depth
        finally:
            pass
            # self.pipeline.stop()

    def get_pointcloud(self):
        try:
            while True:
                # get frame
                frames = self.pipeline.wait_for_frames()
                # frames = self.pipeline.poll_for_frames()
                aligned_frames = self.align.process(frames)  # align image
                # convert rgb to np.array
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    continue

                # filter depth
                depth_frame_filter = depth_frame
                for trans in self.g_rs_depth_postprocess_list:
                    depth_frame_filter = trans.process(depth_frame_filter)
                depth_frame = depth_frame_filter

                # cal pointcloud
                pc = rs.pointcloud()
                pc.map_to(color_frame)
                cloud = pc.calculate(depth_frame)
                cloud = cloud.get_vertices(dims=2)
                cloud = np.array(cloud)
                # reshape
                # shape = np.shape(color_frame.get_data())
                # cloud = cloud.reshape(shape)
                return cloud
        finally:
            pass
            # self.pipeline.stop()

    def get_image_pointcloud(self, color_depth=True):
        try:
            while True:
                # get frame
                frames = self.pipeline.wait_for_frames()
                # frames = self.pipeline.poll_for_frames()
                aligned_frames = self.align.process(frames)  # align image
                # convert rgb to np.array
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    continue

                # infrared_frame = frames.first(rs.stream.infrared)
                # IR_image = np.asanyarray(infrared_frame.get_data())

                # filter depth
                depth_frame_filter = depth_frame
                for trans in self.g_rs_depth_postprocess_list:
                    depth_frame_filter = trans.process(depth_frame_filter)
                depth_frame = depth_frame_filter

                # get rbg, depth
                image_rgb = np.asanyarray(color_frame.get_data())
                image_depth = np.asanyarray(depth_frame.get_data())
                # colorized_depth = np.asanyarray(
                #     self.colorizer.colorize(depth_frame).get_data()
                # )
                rs_rgb = cv2.resize(
                    image_rgb, self.rs_size, interpolation=cv2.INTER_NEAREST
                )
                rs_depth = cv2.resize(
                    image_depth, self.rs_size, interpolation=cv2.INTER_NEAREST
                )
                if color_depth:
                    rs_depth = colorize(rs_depth, clipping_range=(None, 5000), colormap=cv2.COLORMAP_JET)  # cv2.COLORMAP_HSV

                # cal pointcloud
                pc = rs.pointcloud()
                pc.map_to(color_frame)
                cloud = pc.calculate(depth_frame)
                cloud = cloud.get_vertices(dims=2)
                cloud = np.array(cloud)
                # reshape
                # shape = np.shape(color_frame.get_data())
                # cloud = cloud.reshape(shape)
                return cloud, rs_rgb / 255.0, rs_depth
        finally:
            pass
            # self.pipeline.stop()

    def get_intrinsics_matrix(self):
        # get frame
        profile = self.pipeline.get_active_profile().get_streams()[1]
        profile = profile.as_video_stream_profile()
        print(profile)
        intrinsics = profile.get_intrinsics()
        print(intrinsics)
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy
        # intrinsics_matrix = np.array(
        #     [[fx, 0, cx],
        #      [0, fy, cy],
        #      [0, 0, 1]]
        # )
        rs_intrinsics_matrix = np.array(
            [[fx * self.rs_ratio, 0, cx * self.rs_ratio],
             [0, fy * self.rs_ratio, cy * self.rs_ratio],
             [0, 0, 1]]
        )
        print(rs_intrinsics_matrix)
        return rs_intrinsics_matrix


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


if __name__ == '__main__':
    camera = RealsenseCamera(camera_sn='138422075756', rs_ratio=0.5)
    start_time = time.time()
    for i in range(1):
        image_rgb, image_depth = camera.get_image()
        image_rgb = (image_rgb * 255.0).astype(np.uint8)
        image_depth = (image_depth * 255.0).astype(np.uint8)
        # plt.imshow(image_rgb)
        # plt.show()
        # plt.imshow(image_depth)
        # plt.show()
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite('rgb.png', image_rgb)
        cv2.imwrite('depth.png', image_depth)
        # pointcloud = camera.get_pointcloud()
    print(time.time() - start_time)
    print(camera.get_intrinsics_matrix())
