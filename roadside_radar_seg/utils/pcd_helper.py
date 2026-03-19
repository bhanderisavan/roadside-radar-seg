#! /usr/bin/env python3
"""
This script is supposed to run on python3
"""

from pathlib import Path
from collections import OrderedDict
import numpy.lib.recfunctions as rfn
import struct, re, math
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Union, List
import pandas as pd


class PcdHelper:
    """
    custom utilities for working with pointcloud2 data
    """

    def __init__(self):
        pass

    def dbscan(
        self,
        eps: float,
        min_points: int,
        cloud: np.recarray,
        fields: list = None,
        sorted: bool = False,
    ) -> np.recarray:
        """
        perform dbscan clustering on given cloud and append cluster id field to the cloud.
        cloud must have "index" field.
        if x_thresh and/or v_thresh is given, the dbscan is performed on preprocessed cloud.
        all background points including cluster outliers have lables -1.

        Args:
            eps: dbscan eps parameter
            min_points: dbscan min_samples parameter
            cloud : numpy structured array
            fields: fields to be used for clustering
                    if fields is None, entire cloud is used in clustering
            sorted: if True, returns sorted cloud according to cluster id [-1 to len(db.labels_)]

        returns:
            cloud with additional cluster_id field assigned to each record in input cloud

        Process
        -------
        1.  A new field with fieldname `cluster_id` is added to `cloud`. The default value is -1.
            i.e, all points are labelled as background initially. The values will be updated based on dbscan output.
        2.  Based on the values of `x_thresh` and `v_thresh`, a  new preprocessed cloud is obtained.
            If both the parameters are None/Not supplied, then preprocessed cloud = raw cloud.
        3.  The dbscan [from sklearn] with given args is performed on preprocessed cloud.
            This gives us cluster labels (i.e `cluster_id`) for each point in preprocessed cloud.
        4.  The index location of each point of preprocessed cloud in raw cloud is matched by `index` field.
            The values of `cluster_id` field in raw cloud is updated based on the labels obtained from step 3.

        TODO :  allow custom distance metric for dbscan.
        """

        if (cloud is None) or (cloud.shape[0] == 0):
            raise ValueError(f"No Cloud provided.")

        assert cloud.dtype.names is not None, "Cloud must have fieldnames"

        if "cluster_id" in cloud.dtype.names:
            raise NotImplementedError("Cannot cluster already clustered cloud.")

        if fields is None:
            fields = list(cloud.dtype.names)

        # all the field passed for clustering must be present in the cloud.
        assert all(
            [f in cloud.dtype.names for f in fields]
        ), f"cloud is missing atleast one field from {tuple(fields)}."

        row_cloud = cloud.copy()

        # cluster id array with all values = -1
        # replace actual cluster points with their ids
        dtype = np.dtype({"names": ["cluster_id"], "formats": [np.int16]})
        cl_ids = np.recarray(row_cloud.shape, dtype)
        cl_ids.fill(-1)

        raw_clustered_cloud = rfn.merge_arrays(
            (row_cloud, cl_ids), flatten=True, asrecarray=True
        )

        db = DBSCAN(eps=eps, min_samples=min_points)

        # maybe can avoid using pandas
        db.fit(pd.DataFrame.from_records(cloud[fields]))

        # sanity check
        if db.labels_.shape[0] != cloud.shape[0]:
            raise ValueError("dbscan output labels shape does not match cloud shape")

        for idx, cl_idx in zip(cloud["index"], db.labels_):
            arg = np.where(raw_clustered_cloud["index"] == idx)[0].item()
            raw_clustered_cloud[arg]["cluster_id"] = cl_idx

        if sorted:
            raw_clustered_cloud.sort("cluster_id")

        return raw_clustered_cloud

    def get_type_str(
        self, metadata, dummy_name="PAD", remove_pad=True, byte_order="little_endian"
    ):
        """
        create a type string from metadata for unpacking the bytes
        Args:
            metadata: metadata dict generated from pcd header
            remove_pad: if True, padding bytes will not be decoded while constructing type string, instead "x" will be put
            dummy_field: name of the field corresponding to pad bytes
            byte_order: in which order the bytes were encoded

        return: type string for unpacking the bytes
        """
        meta = metadata.copy()
        #         h = meta["HEIGHT"]
        #         w = meta["WIDTH"]
        # field sizes will be identical to meta["size"] if all counts are 1
        # else ith element will be the multiplication of meta["size"][i]*meta["count"][i]
        field_sizes = [
            meta["SIZE"][i] * meta["COUNT"][i] for i in range(len(meta["SIZE"]))
        ]
        # expected sizes of fields
        expected_sizes = [0, 1, 2, 4, 8]

        # Lookup table according to python struct library
        unpacking_lut = {
            "F": {2: "e", 4: "f", 8: "d"},
            "I": {1: "b", 2: "h", 4: "i", 8: "q"},
            "U": {1: "B", 2: "H", 4: "I", 8: "Q"},
            "byte_order": {"little_endian": "<", "big_endian": ">", "native": "@"},
            "pad_byte": "x",
        }

        # define the byte order to unpack
        start_byte = unpacking_lut["byte_order"][byte_order]
        # intantiate type string starting with starting byte
        type_str = start_byte + ""

        if remove_pad:
            # do not decode padded bytes
            # for ith type and ith size, get PYTHON STRUCT decoding string format, and add it to type string
            for typ, size, field in zip(meta["TYPE"], field_sizes, meta["FIELDS"]):
                if field != dummy_name:
                    # processing actual field
                    char_fmt = unpacking_lut[typ][size]
                    type_str += char_fmt
                else:
                    # pad field encountered
                    char_fmt = str(size) + unpacking_lut["pad_byte"]
                    type_str += char_fmt

            # alternative way of getting type str, more compact but less readable
            # char_str = ''.join(unpacking_lut[typ][size] if field != dummy_field else str(size)+unpacking_lut["pad_byte"] for typ, size, field in zip(md["TYPE"], field_sizes, md["FIELDS"]))
            # type_str += char_str

        else:
            # decode padded bytes - user wants to keep padded bytes
            # field byte size must be in expected sizes, otherwise it will give unpacking lut key error
            for idx, sz in enumerate(field_sizes):
                assert (
                    sz in expected_sizes
                ), f"expected byte size in: {expected_sizes}, got '{sz}' at index '{idx}'"

            # for ith type and ith size, get PYTHON STRUCT decoding string format, and add it to type string
            for typ, size in zip(meta["TYPE"], field_sizes):
                if size == 0:
                    continue
                else:
                    # cannot handle size of 15? (non existant in lut)
                    char_fmt = unpacking_lut[typ][size]
                    type_str += char_fmt
            # alternative way of getting type str, more compact but less readable
            # char_str = ''.join(unpacking_lut[typ][size] for typ, size in zip(meta["TYPE"], field_sizes))
            # type_str += char_str

        # total byte size of one point calculated from generate type string
        str_size = struct.calcsize(type_str)
        # total byte size of one point calculated from sizes of all the fields from pcd metadata
        point_size = sum(field_sizes)
        # size calculated with type string must match  the actual size, otherwise decoding will be erroneous
        assert (
            str_size == point_size
        ), f"Calculated byte size of a point from type string : {str_size} DIFFERES from actual byte size {point_size}"

        return type_str

    def _read_pcd(self, pcd_path: Union[Path, str], sensor: str) -> np.recarray:
        """reads the pcd file for sensor.

        Args:
            pcd_path (Union[Path, str]): path ot pcd file.
            sensor (str): radar or lidar.

        Returns:
            np.recarray: pcd point cloud

        NOTE: hard coded type strings for ouster lidar and ARS548 Radar.
        """

        pcd_path = Path(pcd_path)

        assert pcd_path.is_file(), f"No pcd found at {str(pcd_path)}."
        # ordered dict for reading metadata information from pcd file.
        metadata = self.pcd_metadata_template()

        with open(str(pcd_path), "rb") as pcd_file:
            for line in pcd_file:
                ln = line.strip().decode("utf-8")
                # first line, or any line with  unimortant content.
                if ln.startswith("#") or len(ln) < 2:
                    continue
                # Regular expression matching with the data of the pcd header.
                match = re.match("(\w+)\s+([\w\s\.]+)", ln)
                # no match detected, meaning the header is faulty
                if not match:
                    print(f"warning: can't understand line: {ln}")
                    continue
                # header key, and values - all are strings
                key, value = match.group(1), match.group(2)

                if key == "VERSION":
                    pass

                # viewpoint format -  translation (tx ty tz) + quaternion (qw qx qy qz)
                if key == "VIEWPOINT":
                    metadata[key] = list(map(float, value.split()))
                # these fields should be converted into int data type - only one entry
                if key in ["POINTS", "HEIGHT", "WIDTH"]:
                    metadata[key] = int(value)
                # these fields belong to int, but list of entries
                if key in ["SIZE", "COUNT"]:
                    metadata[key] = list(map(int, value.split()))
                # convert single string to list of strings
                if key in ["TYPE", "FIELDS", "DATA"]:
                    metadata[key] = value.split() if len(value.split()) > 1 else value

                # here begins the actual binary data
                if ln.startswith("DATA"):
                    break

            # starting byte index.
            start = 0
            # list of points.
            cloud = []
            # actual binary data
            binary_data = pcd_file.read()

            # list of byte sizes for each of the pointfields. - used for decoding the binary data.
            field_sizes = [
                metadata["SIZE"][i] * metadata["COUNT"][i]
                for i in range(len(metadata["SIZE"]))
            ]

            if sensor.lower() == "radar":
                # type string to get the dtype information for binary decoding of radar pcd.
                type_str = self.get_type_str(metadata)
                # list of numpy data types for each fields for converting list of points to numpy record array.
                np_dtypes = [
                    (field, np.dtype(typ_str))
                    for field, typ_str in zip(metadata["FIELDS"], type_str[1:])
                ]
            if sensor.lower() == "lidar":
                # type string to get the dtype information for binary decoding of lidar pcd.
                type_str = self.get_type_str(metadata)
                # type_str = "<fff4xfIHB1xH2xI4x4x4x"
                # lidar pcds contain extra pading bytes from the sensor. generating numpy dtypes considering that fact.
                np_dtypes = [
                    (field, np.dtype(typ))
                    for field, typ in zip(
                        metadata["FIELDS"], type_str[1:].replace("x", "")
                    )
                    if not field == "PAD"
                ]

            # binary decoding in the chunks of sum(field_sizes).
            for pts in range(metadata["POINTS"]):
                # index slice for decoding current point.
                end = start + sum(field_sizes)
                pt = struct.unpack(type_str, binary_data[start:end])
                cloud.append(pt)
                start = end
            # list of lists to numpy record array.
            cloud_np = np.rec.array(cloud, dtype=np_dtypes)

            # adding extra "index" field to keep track of point index.
            if "index" not in cloud_np.dtype.names:
                d_type = np.dtype({"names": ["index"], "formats": [np.uint32]})
                seq = np.linspace(
                    0,
                    cloud_np.shape[0],
                    cloud_np.shape[0],
                    endpoint=False,
                    dtype=d_type,
                )
                cloud_np = rfn.merge_arrays(
                    (seq, cloud_np), flatten=True, asrecarray=True
                )

        # we only have radar points annotations till 120 meters.
        # so, we clip the raw radar cloud to 120 meters in x.
        if sensor == "radar":
            cloud_np = cloud_np[cloud_np["x"] <= 120]

        return cloud_np

    def read_radar_pcd(self, radar_pcd_path) -> np.recarray:
        radar_cloud = self._read_pcd(radar_pcd_path, "radar")
        return radar_cloud

    def read_lidar_pcd(self, lidar_pcd_path) -> np.recarray:
        lidar_cloud = self._read_pcd(lidar_pcd_path, "lidar")
        return lidar_cloud

    def pcd_metadata_template(self):
        """
        returns an ordered dict template for generating  pcd header
        """
        metadata = OrderedDict(
            (
                ("VERSION", 0.7),
                ("FIELDS", []),
                ("SIZE", []),
                ("TYPE", []),
                ("COUNT", []),
                ("WIDTH", 0),
                ("HEIGHT", 0),
                ("VIEWPOINT", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                ("POINTS", 0),
                ("DATA", "binary"),
            )
        )

        return metadata

    
    def metadata_to_npdtype(self, metadata, dummy_prefix="PAD", remove_pad=True):
        """
        create a list of tuples containing field name, np_dtype
        Args:
            metadata: metadata of cloud
            dummy_name: name of the dummy field
        return:
            np_dtypes: list of tuples containing field name, np_dtype
        """
        dummy_count = 0
        # list containing numpy datatypes
        np_dtypes = []
        # metadata
        fields = []
        types = []
        sizes = []

        for f, t, s in zip(metadata["FIELDS"], metadata["TYPE"], metadata["SIZE"]):
            # only append the metadata if it is not dummy field
            if f != dummy_prefix:
                fields.append(f)
                types.append(t)
                sizes.append(s)

            else:
                # check if user wants to remove pad bytes or not, if not then append field name with _0 suffix
                if not remove_pad:
                    fields.append(dummy_prefix + f"_{dummy_count}")
                    dummy_count += 1
                    types.append(t)
                    sizes.append(s)

                # if user wants to remove pad bytes, discard the corespondig metadata
                else:
                    continue

        # pointfield to (type, size) mapping e.g 1 --> ("I", 1)
        pf_to_type_size = self.pf_to_type_size()
        # pointfield to numpy dtype mappings
        pf_to_np, _ = self.get_dtype_mappings()

        for f, t, s in zip(fields, types, sizes):

            # get the pf datatype (key) where type,size in ith iteration matches to pointfield to (type, size) mapping
            pf = [k for k, v in pf_to_type_size.items() if v == (t, s)][0]

            # append tuple conataining field name, and numpy stype
            np_dtypes.append((f, pf_to_np[pf]))

        return np_dtypes

    def _calc_vxvy(self, cloud_np):

        vx_func = lambda row: math.cos(row.azimuth_angle) * (
            row.range_rate * math.cos(row.elevation_angle)
        )
        vy_func = lambda row: math.sin(row.azimuth_angle) * (
            row.range_rate * math.cos(row.elevation_angle)
        )

        vx = np.asarray(
            list(map(vx_func, cloud_np)),
            dtype={"names": ["v_x"], "formats": [np.float32]},
        )
        vy = np.asarray(
            list(map(vy_func, cloud_np)),
            dtype={"names": ["v_y"], "formats": [np.float32]},
        )
        cloud_np = rfn.merge_arrays((cloud_np, vx, vy), flatten=True, asrecarray=True)

        return cloud_np
