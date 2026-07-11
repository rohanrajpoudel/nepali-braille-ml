# -*- coding: utf-8 -*-
"""
Braille grid detection and binary representation conversion.
"""
import numpy as np
import cv2


def braille_grid_detection(points, theta, D_dis, image):
    """
    Implements the Braille Character Recognition algorithm to detect the grid for characters.

    Parameters:
    - points: list or array of shape (N, 2) containing (x, y) coordinates of detected Braille dots.
    - theta: Threshold for grouping dots into horizontal/vertical lines.
    - D_dis: Threshold distance between intersection points and detected dots.

    Returns:
    - b_cell_info: List of (x, y) tuples representing the occupied Braille grid coordinate points.
    - braille_bins: List of lists, where each sublist contains strings like '1,0,0,0,0,0' for each cell in a line.
    """
    points = np.array(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must be a list or array of (x, y) coordinates.")
    
    if len(points) == 0:
        return [], []
    
    # Step 1: Group points into horizontal lines
    sorted_idx_h = np.argsort(points[:, 1])
    sorted_points_h = points[sorted_idx_h]
    
    hor_groups = []
    current_group = [sorted_points_h[0]]
    for p in sorted_points_h[1:]:
        if abs(p[1] - current_group[-1][1]) <= theta:
            current_group.append(p)
        else:
            if len(current_group) >= 2:
                hor_groups.append(np.array(current_group))
            current_group = [p]
    if len(current_group) >= 2:
        hor_groups.append(np.array(current_group))
    
    # Fit horizontal line equations: y = m_h * x + b_h
    hor_lines = []
    hor_avg_y = []
    for group in hor_groups:
        x = group[:, 0]
        y = group[:, 1]
        m_h, b_h = np.polyfit(x, y, 1)
        hor_lines.append((m_h, b_h))
        hor_avg_y.append(np.mean(y))
    
    # Sort horizontal lines by average y (top to bottom)
    indices_h = np.argsort(hor_avg_y)
    hor_lines_sorted = [hor_lines[i] for i in indices_h]
    hor_avg_y_sorted = [hor_avg_y[i] for i in indices_h]
    
    # Step 2: Group points into vertical lines
    sorted_idx_v = np.argsort(points[:, 0])
    sorted_points_v = points[sorted_idx_v]
    
    ver_groups = []
    current_group = [sorted_points_v[0]]
    for p in sorted_points_v[1:]:
        if abs(p[0] - current_group[-1][0]) <= theta:
            current_group.append(p)
        else:
            if len(current_group) >= 2:
                ver_groups.append(np.array(current_group))
            current_group = [p]
    if len(current_group) >= 2:
        ver_groups.append(np.array(current_group))
    
    # Fit vertical line equations: x = m_v * y + b_v
    ver_lines = []
    ver_avg_x = []
    for group in ver_groups:
        y = group[:, 1]
        x = group[:, 0]
        m_v, b_v = np.polyfit(y, x, 1)
        ver_lines.append((m_v, b_v))
        ver_avg_x.append(np.mean(x))
    
    # Sort vertical lines by average x (left to right)
    indices_v = np.argsort(ver_avg_x)
    ver_lines_sorted = [ver_lines[i] for i in indices_v]
    ver_avg_x_sorted = [ver_avg_x[i] for i in indices_v]
    
    # Step 3: Calculate all intersection points
    in_points = []
    for m_h, b_h in hor_lines_sorted:
        for m_v, b_v in ver_lines_sorted:
            denom = 1 - m_h * m_v
            if abs(denom) < 1e-10:
                continue
            y = (m_h * b_v + b_h) / denom
            x = m_v * y + b_v
            in_points.append((x, y))
    in_points = np.array(in_points)
    
    # Step 4: Match to occupied points
    b_cell_info = []
    for in_p in in_points:
        distances = np.sqrt(np.sum((in_p - points)**2, axis=1))
        if np.any(distances < D_dis):
            b_cell_info.append(tuple(in_p))
    
    # Deduplicate
    if b_cell_info:
        b_cell_info = np.array(b_cell_info)
        unique_idx = np.unique(np.round(b_cell_info, decimals=5), axis=0, return_index=True)[1]
        b_cell_info = [tuple(p) for p in b_cell_info[unique_idx]]
    
    # Prepare rounded set for matching
    b_cell_rounded = set((round(x, 2), round(y, 2)) for x, y in b_cell_info)
    
    # Step 5: Group horizontal lines into text lines (clusters of 3)
    braille_bins = []
    if len(hor_avg_y_sorted) < 3:
        return b_cell_info, braille_bins
    
    diffs_h = np.diff(hor_avg_y_sorted)
    median_diff_h = np.median(diffs_h) if len(diffs_h) > 0 else 0
    line_groups = []
    current = [0]
    for i in range(len(diffs_h)):
        if median_diff_h > 0 and diffs_h[i] > 1.5 * median_diff_h:
            line_groups.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    line_groups.append(current)
    
    line_groups = [g for g in line_groups if len(g) == 3]  # Only valid 3-row groups
    
    # For vertical: assume even number of columns, drop if odd
    num_cols = len(ver_avg_x_sorted)
    num_cols -= num_cols % 2
    
    if num_cols < 2:
        return b_cell_info, braille_bins
    
    # Compute median within-cell column spacing
    within_diffs = [ver_avg_x_sorted[2 * j + 1] - ver_avg_x_sorted[2 * j] for j in range(num_cols // 2)]
    median_within = np.median(within_diffs) if within_diffs else 0
    
    # For each text line
    for group in line_groups:
        hor_lines_this = [hor_lines_sorted[idx] for idx in group]
        cell_bins = []
        
        for j in range(num_cols // 2):
            # Insert space if large gap to previous cell
            if j > 0:
                between = ver_avg_x_sorted[2 * j] - ver_avg_x_sorted[2 * j - 1]
                if median_within > 0 and between > 2 * median_within:
                    cell_bins.append('0,0,0,0,0,0')
            
            # Get line params
            m_h1, b_h1 = hor_lines_this[0]
            m_h2, b_h2 = hor_lines_this[1]
            m_h3, b_h3 = hor_lines_this[2]
            m_vl, b_vl = ver_lines_sorted[2 * j]
            m_vr, b_vr = ver_lines_sorted[2 * j + 1]
            
            # Compute intersection
            def get_intersect(m_h, b_h, m_v, b_v):
                denom = 1 - m_h * m_v
                if abs(denom) < 1e-10:
                    return None
                y = (m_h * b_v + b_h) / denom
                x = m_v * y + b_v
                return (x, y)
            
            pos1 = get_intersect(m_h1, b_h1, m_vl, b_vl)
            pos2 = get_intersect(m_h2, b_h2, m_vl, b_vl)
            pos3 = get_intersect(m_h3, b_h3, m_vl, b_vl)
            pos4 = get_intersect(m_h1, b_h1, m_vr, b_vr)
            pos5 = get_intersect(m_h2, b_h2, m_vr, b_vr)
            pos6 = get_intersect(m_h3, b_h3, m_vr, b_vr)
            
            pos_list = [pos1, pos2, pos3, pos4, pos5, pos6]
            bin_list = []
            for pos in pos_list:
                if pos is None:
                    bin_list.append(0)
                    continue
                r_pos = (round(pos[0], 2), round(pos[1], 2))
                bin_list.append(1 if r_pos in b_cell_rounded else 0)
            
            cell_bins.append(','.join(map(str, bin_list)))
        
        braille_bins.append(cell_bins)

    image = visualize_braille_grid_points(image, b_cell_info, "./temp/viz.jpg")

    return b_cell_info, braille_bins

def visualize_braille_grid_points(
    image,
    b_cell_info,
    save_path=None,
    radius=2,
    color=(0, 0, 255),
    thickness=-1,
    draw_labels=False,
):
    """
    Draw the Braille grid points that were retained for cell formation.

    Parameters
    ----------
    image : np.ndarray
        Original image (BGR, OpenCV format).

    b_cell_info : list[(x, y)]
        Output from braille_grid_detection().
        These are the occupied grid intersection points.

    save_path : str, optional
        If provided, saves the visualization.

    radius : int
        Radius of each drawn point.

    color : tuple
        BGR color of the dots.
        Default: Red.

    thickness : int
        Circle thickness.
        -1 = filled.

    draw_labels : bool
        Draw point indices for debugging.

    Returns
    -------
    np.ndarray
        Image with overlay.
    """

    vis = image.copy()

    for idx, (x, y) in enumerate(b_cell_info):
        center = (int(round(x)), int(round(y)))

        cv2.circle(
            vis,
            center,
            radius,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        if draw_labels:
            cv2.putText(
                vis,
                str(idx),
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    if save_path is not None:
        cv2.imwrite(save_path, vis)

    return vis