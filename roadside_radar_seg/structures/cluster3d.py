import numpy as np
from collections import namedtuple
from typing import List, Union
import ast
from roadside_radar_seg.structures import ObjectCategory, TimeStamp
from numbers import Real

ClusterCentroidTuple = namedtuple("Centroid", ["x", "y", "z"])


class RadarCluster3d:
    def __init__(
        self,
        radar_points_list: Union[List[List], np.recarray],
        centroid: ClusterCentroidTuple = ClusterCentroidTuple(0.0, 0.0, 0.0),
        velocity: float = 0.0,
        category: ObjectCategory = ObjectCategory.UNKNOWN,
        category_confidence: float = 0.0,
    ) -> None:

        if not isinstance(radar_points_list, (List, np.recarray)):
            raise ValueError(
                f"Expected list for radar_points_list or np.recarray, got {type(radar_points_list)}"
            )

        if isinstance(radar_points_list, np.recarray):
            self.radar_points = radar_points_list
        else:
            self._fields = "['index', 'range', 'azimuth_angle', 'elevation_angle', 'range_rate', 'rcs', 'x', 'y', 'z']"
            self._dtypes = "['uint16', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32']"

            np_dtype = np.dtype(
                {
                    "names": ast.literal_eval(self._fields),
                    "formats": ast.literal_eval(self._dtypes),
                }
            )

            # convert radar points list into recarray
            points_list_of_tuples = list(map(tuple, radar_points_list))
            self.radar_points = np.rec.array(points_list_of_tuples, np_dtype)

        self.centroid = centroid
        self.velocity = velocity
        self.category = category
        self._vx_vy_vz = []
        self.category_confidence = category_confidence

    def __len__(self):
        return len(self.radar_points)

    @staticmethod
    def cat(list_of_clusters: list):

        if not isinstance(list_of_clusters, list):
            raise ValueError(
                f"Expected list type for list_of_clusters, instead got {list_of_clusters.__class__.__name__}"
            )

        temp = list_of_clusters[0]

        if len(list_of_clusters) == 1:
            return temp

        for i in range(1, len(list_of_clusters)):
            temp = temp + list_of_clusters[i]

        return temp

    def __add__(self, other):
        """
        adds two instances of RadarCluster3d and returns a new RadarCluster3d instance.
        Does not perform in-place addition.

        It will :
        1. concatenate radar points from both the clusters.
        2. recalculate centroid, avg_velocity, and avg_ecs from concatenated radar points.
        """

        if isinstance(other, RadarCluster3d):
            self_radar_points = self.radar_points.copy()
            new_radar_points = np.append(self_radar_points, other.radar_points)
            new_centroid = ClusterCentroidTuple(
                new_radar_points["x"].mean(),
                new_radar_points["y"].mean(),
                new_radar_points["z"].mean(),
            )
            new_mean_velocity = new_radar_points["range_rate"].mean()
            new_mean_rcs = new_radar_points["rcs"].mean()

            return RadarCluster3d(
                radar_points_list=np.rec.array(new_radar_points),
                centroid=new_centroid,
                velocity=new_mean_velocity,
                rcs=new_mean_rcs,
            )

        else:
            raise NotImplementedError()

    @property
    def vx_vy_vz(self):
        if self._vx_vy_vz:
            return self._vx_vy_vz

        x, y, z = self.centroid
        range_rate = self.velocity

        r = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan(y / x)
        elevation = np.arcsin(z / r)

        v_x = range_rate * np.cos(azimuth) * np.cos(elevation)
        v_y = range_rate * np.sin(azimuth) * np.cos(elevation)
        v_z = range_rate * np.sin(elevation)

        self._vx_vy_vz = [v_x, v_y, v_z]

        return self._vx_vy_vz

    @property  # getter
    def category_confidence(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._category_confidence

    @category_confidence.setter  # setter
    def category_confidence(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if not (value >= 0 and value < 100):
                raise ValueError(
                    f"Value of 3d object category confidence must be between [0,100] in {__class__.__name__}. But category confidence = {value} is given."
                )
            else:
                self._category_confidence = float(value)
        else:
            raise TypeError(
                f"Value of 3d object category confidence must be of type float or int in {__class__.__name__}. But category confidence = {value} of type {type(value)} is given."
            )

    @property
    def fields(self):  # only getter
        return self._fields

    @property
    def dtypes(self):  # only getter
        return self._dtypes

    @property  # getter
    def centroid(self):
        return self._centroid

    @centroid.setter  # setter
    def centroid(self, value):
        if not isinstance(value, ClusterCentroidTuple):
            raise ValueError(
                (
                    f"Value of centroid of cluster must be of type ClusterCentroidTuple in {__class__.__name__}. But value = {value} of type {type(value)} is given."
                )
            )
        self._centroid = value

    @property  # getter
    def velocity(self):
        return self._velocity

    @velocity.setter  # setter
    def velocity(self, value):
        if not isinstance(value, Real):
            raise ValueError(
                (
                    f"Value of velocity of cluster must be numeric in {__class__.__name__}. But value = {value} of type {type(value)} is given."
                )
            )
        self._velocity = value

    # @property  # getter
    # def rcs(self):
    #     return self._rcs

    # @rcs.setter  # setter
    # def rcs(self, value):
    #     if not isinstance(value, Real):
    #         raise ValueError(
    #             (
    #                 f"Value of rcs of cluster must be numeric in {__class__.__name__}. But value = {value} of type {type(value)} is given."
    #             )
    #         )
    #     self._rcs = value

    def __str__(self) -> str:
        out = "-" * 50
        out += f"\n{__class__.__name__} - [cluster centroid = {self.centroid}, avg velocity = {self.velocity}]\n"  # , avg rcs = {self.rcs}]\n"
        out += "Radar cluster points\n"
        for pt in self.radar_points:
            out += f"{pt}\n"

        return out

    __repr__ = __str__


class RadarCluster3dList:
    def __init__(
        self,
        time_stamp: TimeStamp,  # this return epoch time upto nanoseconds
        frame_id: int,
        frame_name: str = "",  # when training the model
        radar_clusters_3d: List[RadarCluster3d] = [],
    ) -> None:

        self.time_stamp = time_stamp
        self.frame_id = frame_id
        self.radar_clusters_3d = radar_clusters_3d
        self._total_clusters = 0  # this is computed property
        self.frame_name = frame_name

    @property  # getter (no setter required for this computed property)
    def total_clusters(self):
        return len(self.radar_clusters_3d)

    @property  # getter
    def frame_id(self):
        return self._frame_id

    @frame_id.setter  # setter
    def frame_id(self, value):
        if isinstance(value, int):
            if value < 0:
                raise ValueError(
                    f"Value of frame id in {__class__.__name__} must be a positive integer. But frame id = {value} is given."
                )
            else:
                self._frame_id = value
        else:
            raise TypeError(
                f"Value of frame id must be of type int in {__class__.__name__}. But frame id = {value} of type {type(value)} is given."
            )

    @property
    def time_stamp(self):
        return self._frame_name

    @time_stamp.setter
    def frame_name(self, value):
        if isinstance(value, str):
            self._frame_name = value
        else:
            raise TypeError(
                f"Value of frame name must be of type str in {__class__.__name__}. But  frame_name = {value} of type {type(value)} is given."
            )

    @property
    def time_stamp(self):
        return self._time_stamp

    @time_stamp.setter
    def time_stamp(self, value):
        if isinstance(value, TimeStamp):
            self._time_stamp = value
        else:
            raise TypeError(
                f"Value of time stamp must be of type TimeStamp in {__class__.__name__}. But time stamp = {value} of type {type(value)} is given."
            )

    def __iter__(self):
        for radar_obj in self.radar_clusters_3d:
            yield radar_obj

    def __getitem__(self, idx):

        if isinstance(idx, int):
            return self.radar_clusters_3d[idx]

        if isinstance(idx, list):
            return [self.radar_clusters_3d[i] for i in idx]

        raise NotImplementedError("")

    def __len__(self):
        return self.total_clusters

    @property  # getter
    def radar_clusters_3d(self):
        return self._radar_clusters_3d

    @radar_clusters_3d.setter
    def radar_clusters_3d(self, value):
        if isinstance(value, List):
            if len(value) == 0:  # if list is empty
                self._radar_clusters_3d = value
            else:  # if list is not empty
                for obj in value:
                    if not isinstance(
                        obj, RadarCluster3d
                    ):  # check if all instances in list are of type RadarCluster3d
                        raise TypeError(
                            f"List must contain only objects of RadarCluster3d in {__class__.__name__}. But object of type {type(obj)} is given."
                        )
                self._radar_clusters_3d = value
        else:
            raise TypeError(
                f"Values of radar cluster 3d must be given as list in {__class__.__name__}. But of type {type(value)} is given."
            )

    def __str__(self):
        str_clusters_list = ""
        for index, obj in enumerate(self.radar_clusters_3d):
            str_clusters_list += str(index + 1) + ". " + str(obj) + "\n"

        return (
            f"-------------------------------------\n"
            f"{__class__.__name__}: epoch time = {self.time_stamp}, frame id = {self.frame_id}, total objects = {self.total_clusters}\n"
            f"{str_clusters_list}"
            f"-------------------------------------\n"
        )

    __repr__ = __str__
