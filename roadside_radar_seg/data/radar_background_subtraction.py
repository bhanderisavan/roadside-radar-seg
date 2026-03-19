from roadside_radar_seg.configs import configurable
import numpy as np


class RadarBGSubtractor:

    @configurable
    def __init__(
        self,
        *,
        maximum_valid_range: float,
        range_cell_size: float,
        maximum_valid_azimuth_angle: float,
        azimuth_cell_size: float,
        maximum_valid_elevation_angle: float,
        elevation_cell_size: float,
    ) -> None:

        self.maximum_valid_range = maximum_valid_range
        self.range_cell_size = range_cell_size
        self.maximum_valid_azimuth_angle = maximum_valid_azimuth_angle
        self.azimuth_cell_size = azimuth_cell_size
        self.maximum_valid_elevation_angle = maximum_valid_elevation_angle
        self.elevation_cell_size = elevation_cell_size

    @classmethod
    def from_config(cls, cfg):
        return {
            "maximum_valid_range": cfg.BGSUB.MAXIMUM_VALID_RANGE,
            "range_cell_size": cfg.BGSUB.RANGE_CELL_SIZE,
            "maximum_valid_azimuth_angle": cfg.BGSUB.MAXIMUM_VALID_AZIMUTH_ANGLE,
            "azimuth_cell_size": cfg.BGSUB.AZIMUTH_CELL_SIZE,
            "maximum_valid_elevation_angle": cfg.BGSUB.MAXIMUM_VALID_ELEVATION_ANGLE,
            "elevation_cell_size": cfg.BGSUB.ELEVATION_CELL_SIZE,
        }

    def perform_bg_sub(self, bg_grid: np.ndarray, radar_static_cloud: np.recarray):
        # static cloud for performing bg sub. (apply v thresh before passing the cloud.)

        cloud = radar_static_cloud.copy()

        excess_cloud = cloud[cloud["range"] > self.maximum_valid_range]
        pcd_cloud = cloud[cloud["range"] <= self.maximum_valid_range]

        # convert azimuth_angles and elevation angles from rad to degree
        pcd_cloud["azimuth_angle"] = np.rad2deg(pcd_cloud["azimuth_angle"])
        pcd_cloud["elevation_angle"] = np.rad2deg(pcd_cloud["elevation_angle"])

        range_grid_index = np.asarray(
            pcd_cloud["range"] / self.range_cell_size,
            dtype="int",
        )
        # print(range_grid_index)

        # shift the azimuth range from [-60°, +60°] to [0, 120°]
        azimuth_grid_index = np.asarray(
            (pcd_cloud["azimuth_angle"] + (self.maximum_valid_azimuth_angle / 2))
            / self.azimuth_cell_size,
            dtype="int",
        )

        # shift the elevation from [-25°, +25°] to [0, 50°]
        elevation_grid_index = np.asarray(
            (pcd_cloud["elevation_angle"] + (self.maximum_valid_elevation_angle / 2))
            / self.elevation_cell_size,
            dtype="int",
        )

        t = bg_grid[elevation_grid_index, range_grid_index, azimuth_grid_index]

        background_index = np.where(t == 1)
        foreground_index = np.where(t == 0)

        radar_points_foreground = pcd_cloud[foreground_index[0]]
        radar_points_background = pcd_cloud[background_index[0]]
        radar_points_background = np.append(radar_points_background, excess_cloud)

        return radar_points_foreground, radar_points_background


def build_radar_bg_subtractor(cfg):
    return RadarBGSubtractor(cfg)
