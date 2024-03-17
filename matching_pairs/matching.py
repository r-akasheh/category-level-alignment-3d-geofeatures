import open3d as o3d
import numpy as np
from typing import Tuple


def extract_matching_pairs(x: o3d.geometry.PointCloud, y: o3d.geometry.PointCloud, tau) -> Tuple[list, list]:
    x = np.asarray(x.points)
    y = np.asarray(y.points)
    positive_point_pairs = []
    negative_point_pairs = []

    for i in range(len(x)):
        # Compute Euclidean distance between the current point in x and all points in y
        distances = np.linalg.norm(x[i] - y, axis=1)

        # Check if any distance is less than tau
        close_points_indices = np.where(distances < tau)[0]
        negative_points_indices = np.where(distances >= tau)[0]

        # Save pairs of points with distance less than tau
        for j in close_points_indices:
            positive_point_pairs.append((x[i], y[j]))

        for j in negative_points_indices:
            negative_point_pairs.append((x[i], y[j]))

    return positive_point_pairs, negative_point_pairs


def is_positive_matching_pair(x, y, tau) -> bool:
    return np.linalg.norm(x - y, axis=1) < tau
