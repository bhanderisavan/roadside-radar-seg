import numpy as np
import numpy.lib.recfunctions as rfn


def project_cloud_on_image(cloud: np.ndarray, projection_matrix):

    cloud_xyz = cloud[["x", "y", "z", "index"]].copy()
    cloud_xyz["index"] = 1

    projected_points = np.matmul(
        projection_matrix, rfn.structured_to_unstructured(cloud_xyz).transpose()
    )
    w = projected_points[2, :]

    projected_points = np.array(
        [
            projected_points[0, :] / projected_points[2, :],  # u = x/w
            projected_points[1, :] / projected_points[2, :],  # v = y/w
        ]
    )

    projected_points = np.transpose(projected_points)

    dtype = np.dtype(
        {
            "names": ["u", "v"],
            "formats": [np.uint32, np.float32],
        }
    )

    projected_points = rfn.unstructured_to_structured(projected_points, dtype=dtype)

    points_with_uv = rfn.merge_arrays(
        (cloud, projected_points),
        flatten=True,
        asrecarray=True,
    )

    return points_with_uv
