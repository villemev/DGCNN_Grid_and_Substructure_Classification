import json, random, os
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import math

def _get_corner_removal_indices(num_rows: int, num_cols: int, 
                                n_rows: int, n_cols: int, location: str) -> Set[int]:
    """Get linear indices for corner removal."""
    indices = set()
    
    if location == 'top_left':
        start_row, start_col = 0, 0
    elif location == 'top_right':
        start_row, start_col = 0, num_cols - n_cols
    elif location == 'bottom_left':
        start_row, start_col = num_rows - n_rows, 0
    elif location == 'bottom_right':
        start_row, start_col = num_rows - n_rows, num_cols - n_cols
    else:
        return indices
    
    # Clamp to valid ranges
    start_row = max(0, min(start_row, num_rows - 1))
    start_col = max(0, min(start_col, num_cols - 1))
    end_row = min(start_row + n_rows, num_rows)
    end_col = min(start_col + n_cols, num_cols)
    
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            indices.add(i * num_cols + j)
    
    return indices

def _get_side_middle_removal_indices(num_rows: int, num_cols: int,
                                     n_rows: int, n_cols: int, location: str) -> Set[int]:
    """Get linear indices for middle of side removal."""
    indices = set()
    
    if location == 'top':
        start_row = 0
        start_col = (num_cols - n_cols) // 2
        end_row = n_rows
        end_col = start_col + n_cols
    elif location == 'bottom':
        start_row = num_rows - n_rows
        start_col = (num_cols - n_cols) // 2
        end_row = num_rows
        end_col = start_col + n_cols
    elif location == 'left':
        start_row = (num_rows - n_rows) // 2
        start_col = 0
        end_row = start_row + n_rows
        end_col = n_cols
    elif location == 'right':
        start_row = (num_rows - n_rows) // 2
        start_col = num_cols - n_cols
        end_row = start_row + n_rows
        end_col = num_cols
    else:
        return indices
    
    # Clamp to valid ranges
    start_row = max(0, min(start_row, num_rows - 1))
    start_col = max(0, min(start_col, num_cols - 1))
    end_row = max(1, min(end_row, num_rows))
    end_col = max(1, min(end_col, num_cols))
    
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            indices.add(i * num_cols + j)
    
    return indices

def _get_rect_removal_indices(num_rows: int, num_cols: int,
                              n_rows: int, n_cols: int, 
                              offset: Tuple[int, int]) -> Set[int]:
    """Get linear indices for rectangular removal at specific offset."""
    indices = set()
    row_offset, col_offset = offset
    
    start_row = max(0, row_offset)
    start_col = max(0, col_offset)
    end_row = min(start_row + n_rows, num_rows)
    end_col = min(start_col + n_cols, num_cols)
    
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            indices.add(i * num_cols + j)
    
    return indices

def generate_random_removals(rng, num_rows, num_cols, removal_probability=0.7, max_removals=3):
    """
    Generate multiple random removals for a structure.
    
    Args:
        rng: Random number generator
        num_rows: Number of rows in grid
        num_cols: Number of columns in grid
        removal_probability: Probability of having any removals (0.0-1.0)
        max_removals: Maximum number of removal areas (1-4)
    
    Returns:
        List of removal dictionaries, each with 'type', 'size', and 'location'
    """
    removals = []
    
    if rng.random() >= removal_probability:
        return removals  # No removals
    
    # Decide how many removal areas (1 to max_removals)
    num_removals = rng.randint(1, min(max_removals, 4))
    
    # Track used locations to avoid conflicts
    used_corners = set()
    used_sides = set()
    
    for _ in range(num_removals):
        # Choose removal type with weights: corner (40%), side_middle (40%), random_rect (20%)
        removal_type = rng.choices(
            ['corner', 'side_middle', 'random_rect'], 
            weights=[0.33, 0.34, 0.33], 
            k=1
        )[0]
        
        # Random size (smaller for multiple removals)
        max_row_remove = max(1, num_rows // 3)
        max_col_remove = max(1, num_cols // 3)
        n_rows = rng.randint(1, max_row_remove)
        n_cols = rng.randint(1, max_col_remove)
        
        if removal_type == 'corner':
            # Try to find an unused corner
            available_corners = [loc for loc in ['top_left', 'top_right', 'bottom_left', 'bottom_right'] 
                                if loc not in used_corners]
            if not available_corners:
                continue  # Skip if all corners used
            location = rng.choice(available_corners)
            used_corners.add(location)
            
        elif removal_type == 'side_middle':
            # Try to find an unused side
            available_sides = [loc for loc in ['top', 'bottom', 'left', 'right'] 
                              if loc not in used_sides]
            if not available_sides:
                continue  # Skip if all sides used
            location = rng.choice(available_sides)
            used_sides.add(location)
            
        else:  # random_rect
            # Random position for rectangular removal
            max_row_offset = max(0, num_rows - n_rows)
            max_col_offset = max(0, num_cols - n_cols)
            row_offset = rng.randint(0, max_row_offset) if max_row_offset > 0 else 0
            col_offset = rng.randint(0, max_col_offset) if max_col_offset > 0 else 0
            location = None  # Not used for random_rect
        
        removal = {
            'type': removal_type,
            'size': (n_rows, n_cols),
            'location': location
        }
        
        # Add offset for random_rect
        if removal_type == 'random_rect':
            removal['offset'] = (row_offset, col_offset)
        
        removals.append(removal)
    
    return removals

def add_axis_aligned_random_columns(points, rng, R, C, xs, ys, num_grid_points, 
                                     rotation_rad, origin, random_axis_range=(0, 0)):
    """
    Add random columns aligned with either U-axis or V-axis.
    Places columns on actual grid lines with random positions along those lines.
    
    Args:
        points: List of point dictionaries to append to
        rng: Random number generator
        R: Number of rows in grid
        C: Number of columns in grid
        xs: Original x positions of columns (before rotation)
        ys: Original y positions of rows (before rotation)
        num_grid_points: Total grid points (R * C) for capping calculation
        rotation_rad: Rotation angle in radians
        origin: (ox, oy) origin point for translation
        random_axis_range: tuple (min_cols, max_cols) for number of axis-aligned columns
                          Will be split randomly between U-axis and V-axis aligned
    
    Returns:
        tuple: (num_u_axis, num_v_axis) - counts of columns added for each axis
    """
    # Calculate total number of axis-aligned columns to add
    max_random_axis = int(num_grid_points * 0.1)
    num_random_axis = rng.randint(random_axis_range[0], random_axis_range[1])
    num_random_axis = min(num_random_axis, max_random_axis)
    
    if num_random_axis == 0:
        return 0, 0
    
    ox, oy = origin
    cos_theta = math.cos(rotation_rad)
    sin_theta = math.sin(rotation_rad)
    
    # Randomly distribute between U-axis and V-axis
    num_random_u_axis = 0
    num_random_v_axis = 0
    
    for i in range(num_random_axis):
        if rng.random() < 0.5:  # 50/50 split between U and V
            # U-axis aligned: pick a random column (u index) and random y position along it
            u_val = rng.randint(0, C - 1)
            x_local = xs[u_val]  # Fixed x at this column
            y_local = rng.uniform(min(ys), max(ys))  # Random y along the column
            
            # Apply rotation and translation
            x_rot = x_local * cos_theta - y_local * sin_theta
            y_rot = x_local * sin_theta + y_local * cos_theta
            
            points.append({
                "x_mm": x_rot + ox, 
                "y_mm": y_rot + oy, 
                "u": u_val, 
                "v": -1
            })
            num_random_u_axis += 1
        else:
            # V-axis aligned: pick a random row (v index) and random x position along it
            v_val = rng.randint(0, R - 1)
            x_local = rng.uniform(min(xs), max(xs))  # Random x along the row
            y_local = ys[v_val]  # Fixed y at this row
            
            # Apply rotation and translation
            x_rot = x_local * cos_theta - y_local * sin_theta
            y_rot = x_local * sin_theta + y_local * cos_theta
            
            points.append({
                "x_mm": x_rot + ox, 
                "y_mm": y_rot + oy, 
                "u": -1, 
                "v": v_val
            })
            num_random_v_axis += 1
    
    return num_random_u_axis, num_random_v_axis

def add_shifted_points_suffix(filepath):
    """
    Add '_shifted_points' suffix before the .json extension.
    Example: 'JSON/model.json' -> 'JSON/model_shifted_points.json'
    """
    if filepath.endswith('.json'):
        return filepath[:-5] + '_shifted_points.json'
    else:
        return filepath + '_shifted_points'

def shift_random_points(struct, rng, point_count_domain, dx_domain_mm, dy_domain_mm):
    """
    Randomly shift individual points (columns).
    - Chooses K points uniformly at random (K in point_count_domain, capped by total points).
    - For each selected point, samples dx,dy and adds to x_mm,y_mm ONLY.
    - Leaves all other fields unchanged.
    - Writes a concise log in struct['moved_points'].
    """
    pts = struct["points"]
    n = len(pts)
    kmin, kmax = point_count_domain    
    if kmin > n:
        k = n    
    else:
        k = rng.randint(kmin, min(kmax, n)) if n > 0 else 0

    # choose distinct point indices
    idxs = list(range(n))
    rng.shuffle(idxs)
    chosen = idxs[:k]

    moves = []
    for i in chosen:
        p = pts[i]
        dx = rng.uniform(*dx_domain_mm)
        dy = rng.uniform(*dy_domain_mm)
        p["x_mm"] += dx
        p["y_mm"] += dy

    struct["moved_points"] = moves
    # Randomly shuffle the order of points in the structure
    rng.shuffle(struct["points"])
    return struct


# Interactive plot function
def plot_structure(idx,structures):
    s = structures[idx]
    xs = [p["x_mm"] for p in s["points"]]
    ys = [p["y_mm"] for p in s["points"]]
    us = [p["u"] for p in s["points"]]
    vs = [p["v"] for p in s["points"]]
    
    # Separate different types of points
    grid_points = [(x, y, u, v) for x, y, u, v in zip(xs, ys, us, vs) if u >= 0 and v >= 0]
    random_points = [(x, y, u, v) for x, y, u, v in zip(xs, ys, us, vs) if u == -1 and v == -1]
    u_axis_points = [(x, y, u, v) for x, y, u, v in zip(xs, ys, us, vs) if u >= 0 and v == -1]
    v_axis_points = [(x, y, u, v) for x, y, u, v in zip(xs, ys, us, vs) if u == -1 and v >= 0]
    
    plt.figure(figsize=(10,10))  # Larger figure to accommodate labels
    
    # Plot grid points in blue
    if grid_points:
        grid_xs, grid_ys, grid_us, grid_vs = zip(*grid_points)
        plt.scatter(grid_xs, grid_ys, s=80, c="blue", alpha=0.7, label="Grid points")
        
        # Add u,v labels to grid points
        for x, y, u, v in grid_points:
            plt.annotate(f'({u},{v})', (x, y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Plot fully random columns in red
    if random_points:
        rand_xs, rand_ys, _, _ = zip(*random_points)
        plt.scatter(rand_xs, rand_ys, s=80, c="red", alpha=0.7, marker='x', label="Random columns")
    
    # Plot U-axis aligned columns in orange
    if u_axis_points:
        u_xs, u_ys, u_us, _ = zip(*u_axis_points)
        plt.scatter(u_xs, u_ys, s=80, c="orange", alpha=0.7, marker='s', label="U-axis aligned")
        # Add u labels
        for x, y, u, v in u_axis_points:
            plt.annotate(f'(u={u})', (x, y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    
    # Plot V-axis aligned columns in green
    if v_axis_points:
        v_xs, v_ys, _, v_vs = zip(*v_axis_points)
        plt.scatter(v_xs, v_ys, s=80, c="green", alpha=0.7, marker='^', label="V-axis aligned")
        # Add v labels
        for x, y, u, v in v_axis_points:
            plt.annotate(f'(v={v})', (x, y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
    
    plt.gca().set_aspect("equal")
    
    # Enhanced title with statistics
    R = s.get('R', '?')
    C = s.get('C', '?')
    num_random = s.get('num_random_columns', 0)
    num_u_axis = s.get('num_random_u_axis', 0)
    num_v_axis = s.get('num_random_v_axis', 0)
    num_removed = s.get('num_removed_points', 0)  # Updated key name
    expected_grid_points = R * C if isinstance(R, int) and isinstance(C, int) else 0
    actual_grid_points = len(grid_points)
    
    title = f"Building {idx}  |  Grid {R}×{C}\n"
    title += f"Grid points: {actual_grid_points}/{expected_grid_points} "
    if num_removed > 0:
        title += f"(removed {num_removed}) "
    if num_random > 0:
        title += f"+ {num_random} random"
    if num_u_axis > 0:
        title += f" + {num_u_axis} U-axis"
    if num_v_axis > 0:
        title += f" + {num_v_axis} V-axis"
    
    plt.title(title)
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.grid(True, alpha=0.3)

    plt.legend()

def build_structure_with_rotation_exCol(rng, rows_range, cols_range, row_gap_domain_mm, col_gap_domain_mm=None, 
                                col_gap_fixed_mm=None, origin=(0.0,0.0), rotation_domain_deg=(0, 0), 
                                random_columns_range=(0, 0), remove_points_range=(0, 0),
                                random_axis_range=(0, 0)):
    """
    Generate one grid structure with dimensions in millimeters and optional rotation.
    
    Parameters:
    - rotation_domain_deg: tuple (min_deg, max_deg) for rotation angle in degrees
                          e.g., (-5, 5) for ±5 degrees, (0, 360) for full rotation
    - random_columns_range: tuple (min_cols, max_cols) for number of random extra columns
                           e.g., (0, 5) for 0-5 random columns, (10, 20) for 10-20 random columns
                           Note: Actual number will be capped at 10% of grid points (R × C)
    - remove_points_range: tuple (min_remove, max_remove) for number of grid points to remove
                           e.g., (0, 3) to remove 0-3 random grid positions
                           Note: Actual number will be capped at 20% of grid points (R × C)
    - random_axis_range: tuple (min_cols, max_cols) for axis-aligned random columns
                        These columns align with either U-axis (random u, v=-1) or V-axis (u=-1, random v)
                        Distribution between U and V is random (50/50 split)
                        e.g., (0, 6) for 0-6 axis-aligned columns
    """
    R = rng.randint(rows_range[0], rows_range[1])
    C = rng.randint(cols_range[0], cols_range[1])

    # Row spacing
    row_gaps = [rng.uniform(*row_gap_domain_mm) for _ in range(R-1)]
    ys = [0.0]
    for g in row_gaps: ys.append(ys[-1] + g)

    # Column spacing
    if col_gap_domain_mm:
        col_gaps = [rng.uniform(*col_gap_domain_mm) for _ in range(C-1)]
    else:
        gap = col_gap_fixed_mm if col_gap_fixed_mm else 1000.0  # default 1 m = 1000 mm
        col_gaps = [gap]*(C-1)
    xs = [0.0]
    for g in col_gaps: xs.append(xs[-1] + g)

    # Generate rotation angle
    rotation_deg = rng.uniform(*rotation_domain_deg)
    rotation_rad = math.radians(rotation_deg)
    
    # Apply rotation and translation
    ox, oy = origin
    cos_theta = math.cos(rotation_rad)
    sin_theta = math.sin(rotation_rad)
    
    # Rotate points around origin, then translate
    rotated_xs = []
    rotated_ys = []
    
    # Generate rotated coordinates in the same order as u,v assignment
    for y in ys:  # rows first (v index)
        for x in xs:  # columns second (u index)
            # Rotate around (0,0)
            x_rot = x * cos_theta - y * sin_theta
            y_rot = x * sin_theta + y * cos_theta
            # Then translate
            rotated_xs.append(x_rot + ox)
            rotated_ys.append(y_rot + oy)

    # Create points with u,v grid coordinates preserved
    points = []
    idx = 0
    for v, y in enumerate(ys):
        for u, x in enumerate(xs):
            points.append({
                "x_mm": rotated_xs[idx], 
                "y_mm": rotated_ys[idx], 
                "u": u, 
                "v": v
            })
            idx += 1

    # Calculate number of grid points
    num_grid_points = R * C
    
    # Remove random grid points (capped at 20% of total grid points)
    max_remove_points = int(num_grid_points * 0.2)
    num_remove_points = rng.randint(remove_points_range[0], remove_points_range[1])
    num_remove_points = min(num_remove_points, max_remove_points)  # Cap at 20% of grid
    
    if num_remove_points > 0 and len(points) > 0:
        # Randomly select indices to remove
        num_remove_points = min(num_remove_points, len(points))  # Can't remove more than exist
        indices_to_remove = rng.sample(range(len(points)), num_remove_points)
        # Remove in reverse order to preserve indices
        for idx in sorted(indices_to_remove, reverse=True):
            points.pop(idx)
    
    # Create random columns with random count from range
    # Limit to maximum 10% of grid points
    max_random_columns = int(num_grid_points * 0.1)
    num_random_columns = rng.randint(random_columns_range[0], random_columns_range[1])
    num_random_columns = min(num_random_columns, max_random_columns)  # Cap at 10% of grid
    
    for i in range(num_random_columns):        
        points.append({
                "x_mm": rng.uniform(min(rotated_xs), max(rotated_xs)), 
                "y_mm": rng.uniform(min(rotated_ys), max(rotated_ys)), 
                "u": -1, 
                "v": -1
            })
    
    # Add axis-aligned random columns (distributed between U and V axes)
    num_random_u_axis, num_random_v_axis = add_axis_aligned_random_columns(
        points, rng, R, C, xs, ys, num_grid_points, rotation_rad, origin, random_axis_range
    )

    return {
        "R": R, "C": C,
        "row_gaps_mm": row_gaps,
        "col_gaps_mm": col_gaps,
        "x_positions_mm": xs,  # original positions before rotation
        "y_positions_mm": ys,  # original positions before rotation
        "rotation_deg": rotation_deg,        
        "origin": origin,        
        "num_random_columns": num_random_columns,  # store how many random columns were added
        "num_random_u_axis": num_random_u_axis,  # columns with random u, v=-1
        "num_random_v_axis": num_random_v_axis,  # columns with u=-1, random v
        "num_removed_points": num_remove_points,  # store how many grid points were removed
        "structure_type": "ortogonal",  # Structure type classification
        "u_grid_curve_orders": [1] * C,  # All columns straight (1 = linear)
        "v_grid_curve_orders": [1] * R,  # All rows straight (1 = linear)
        "points": points  # rotated and translated positions
    }


def build_structure_with_skew(rng, rows_range, cols_range, 
                              row_gap_domain_mm, col_gap_domain_mm,
                              rotation_domain_deg=(0, 0),
                              skew_domain_deg=(0, 0),  # NEW: skew angle
                              random_columns_range=(0, 0),
                              remove_points_range=(0, 0),
                              origin=(0.0, 0.0)):
    """
    Generate grid with optional skew (shear transformation).
    skew_domain_deg: tuple (min_deg, max_deg) for skew angle
                     e.g., (-15, 15) for moderate skew
    remove_points_range: tuple (min_remove, max_remove) for number of grid points to remove
                         e.g., (0, 3) to remove 0-3 random grid positions
    """
    R = rng.randint(rows_range[0], rows_range[1])
    C = rng.randint(cols_range[0], cols_range[1])

    # Generate regular spacing
    row_gaps = [rng.uniform(*row_gap_domain_mm) for _ in range(R-1)]
    col_gaps = [rng.uniform(*col_gap_domain_mm) for _ in range(C-1)]
    
    ys = [0.0]
    for g in row_gaps: ys.append(ys[-1] + g)
    xs = [0.0]
    for g in col_gaps: xs.append(xs[-1] + g)

    # Generate skew and rotation angles
    skew_deg = rng.uniform(*skew_domain_deg)
    skew_rad = math.radians(skew_deg)
    rotation_deg = rng.uniform(*rotation_domain_deg)
    rotation_rad = math.radians(rotation_deg)
    
    ox, oy = origin
    cos_theta = math.cos(rotation_rad)
    sin_theta = math.sin(rotation_rad)
    
    # Apply transformations: skew first, then rotate, then translate
    transformed_xs = []
    transformed_ys = []
    
    for y in ys:
        for x in xs:
            # 1. Apply skew (shear in x-direction based on y)
            x_skewed = x + y * math.tan(skew_rad)
            y_skewed = y
            
            # 2. Apply rotation
            x_rot = x_skewed * cos_theta - y_skewed * sin_theta
            y_rot = x_skewed * sin_theta + y_skewed * cos_theta
            
            # 3. Apply translation
            transformed_xs.append(x_rot + ox)
            transformed_ys.append(y_rot + oy)

    # Create points
    points = []
    idx = 0
    for v, y in enumerate(ys):
        for u, x in enumerate(xs):
            points.append({
                "x_mm": transformed_xs[idx], 
                "y_mm": transformed_ys[idx], 
                "u": u, 
                "v": v
            })
            idx += 1

    # Calculate number of grid points
    num_grid_points = R * C
    
    # Remove random grid points (capped at 20% of total grid points)
    max_remove_points = int(num_grid_points * 0.2)
    num_remove_points = rng.randint(remove_points_range[0], remove_points_range[1])
    num_remove_points = min(num_remove_points, max_remove_points)  # Cap at 20% of grid
    
    if num_remove_points > 0 and len(points) > 0:
        # Randomly select indices to remove
        num_remove_points = min(num_remove_points, len(points))  # Can't remove more than exist
        indices_to_remove = rng.sample(range(len(points)), num_remove_points)
        # Remove in reverse order to preserve indices
        for idx in sorted(indices_to_remove, reverse=True):
            points.pop(idx)
    
    # Add random columns if specified
    max_random_columns = int(num_grid_points * 0.1)
    num_random_columns = rng.randint(random_columns_range[0], random_columns_range[1])
    num_random_columns = min(num_random_columns, max_random_columns)
    
    if num_random_columns > 0 and transformed_xs:
        for i in range(num_random_columns):
            # Generate random points within transformed bounds
            rand_x = rng.uniform(min(transformed_xs), max(transformed_xs))
            rand_y = rng.uniform(min(transformed_ys), max(transformed_ys))
            
            points.append({
                "x_mm": rand_x,
                "y_mm": rand_y,
                "u": -1,
                "v": -1
            })
    
    return {
        "R": R, "C": C,
        "row_gaps_mm": row_gaps,
        "col_gaps_mm": col_gaps,
        "rotation_deg": rotation_deg,
        "skew_deg": skew_deg,
        "origin": origin,        
        "num_random_columns": num_random_columns,
        "num_removed_points": num_remove_points,
        "structure_type": "skewed",  # Structure type classification
        "u_grid_curve_orders": [1] * C,  # All columns straight (1 = linear, but skewed)
        "v_grid_curve_orders": [1] * R,  # All rows straight (1 = linear, but skewed)
        "points": points

    }

def build_structure_with_arc_curvature(rng, rows_range, cols_range,
                                       row_gap_domain_mm, col_gap_domain_mm,
                                       arc_curvature_domain=(0.0, 0.0),  # Arc curvature in 1/mm
                                       rotation_domain_deg=(0, 0),
                                       random_columns_range=(0, 0),
                                       remove_points_range=(0, 0),
                                       origin=(0.0, 0.0)):
    """
    Generate grid with arc curvature - columns follow a circular arc.
    
    This creates a curved facade effect where the grid bends along the x-axis.
    
    Parameters:
    - arc_curvature_domain: tuple (min_k, max_k) for curvature coefficient
                           Positive = concave (curves inward)
                           Negative = convex (curves outward)
                           Typical values: (-0.00001, 0.00001) for 5-8m grids
    - remove_points_range: tuple (min_remove, max_remove) for number of grid points to remove
                           e.g., (0, 3) to remove 0-3 random grid positions
                           
    The curvature k relates to radius: R = 1/k
    For a 40m wide grid with k=0.00002, radius ≈ 50m
    """
    R = rng.randint(rows_range[0], rows_range[1])
    C = rng.randint(cols_range[0], cols_range[1])

    # Generate regular spacing
    row_gaps = [rng.uniform(*row_gap_domain_mm) for _ in range(R-1)]
    col_gaps = [rng.uniform(*col_gap_domain_mm) for _ in range(C-1)]
    
    ys = [0.0]
    for g in row_gaps: ys.append(ys[-1] + g)
    xs = [0.0]
    for g in col_gaps: xs.append(xs[-1] + g)

    # Generate curvature coefficient (1/radius)
    k = rng.uniform(*arc_curvature_domain)
    
    # Find x-axis center for symmetric curvature
    x_center = sum(xs) / len(xs) if xs else 0
    
    # Apply arc curvature along x-axis (like a curved facade)
    curved_xs = []
    curved_ys = []
    
    for y in ys:
        for x in xs:
            # Distance from x-axis center
            dx = x - x_center
            
            # Apply parabolic approximation of circular arc
            # For small angles: arc ≈ parabola, displacement ≈ k * dx²
            # This bends the grid in the y-direction based on x position
            y_offset = k * dx**2
            
            x_curved = x
            y_curved = y + y_offset
            
            curved_xs.append(x_curved)
            curved_ys.append(y_curved)

    # Now apply rotation and translation
    rotation_deg = rng.uniform(*rotation_domain_deg)
    rotation_rad = math.radians(rotation_deg)
    ox, oy = origin
    cos_theta = math.cos(rotation_rad)
    sin_theta = math.sin(rotation_rad)
    
    points = []
    idx = 0
    for v, y in enumerate(ys):
        for u, x in enumerate(xs):
            # Rotate curved coordinates
            x_rot = curved_xs[idx] * cos_theta - curved_ys[idx] * sin_theta
            y_rot = curved_xs[idx] * sin_theta + curved_ys[idx] * cos_theta
            
            points.append({
                "x_mm": x_rot + ox, 
                "y_mm": y_rot + oy, 
                "u": u, 
                "v": v
            })
            idx += 1
    
    # Calculate number of grid points
    num_grid_points = R * C
    
    # Remove random grid points (capped at 20% of total grid points)
    max_remove_points = int(num_grid_points * 0.2)
    num_remove_points = rng.randint(remove_points_range[0], remove_points_range[1])
    num_remove_points = min(num_remove_points, max_remove_points)  # Cap at 20% of grid
    
    if num_remove_points > 0 and len(points) > 0:
        # Randomly select indices to remove
        num_remove_points = min(num_remove_points, len(points))  # Can't remove more than exist
        indices_to_remove = rng.sample(range(len(points)), num_remove_points)
        # Remove in reverse order to preserve indices
        for idx in sorted(indices_to_remove, reverse=True):
            points.pop(idx)
    
    # Add random columns if specified
    max_random_columns = int(num_grid_points * 0.1)
    num_random_columns = rng.randint(random_columns_range[0], random_columns_range[1])
    num_random_columns = min(num_random_columns, max_random_columns)
    
    if num_random_columns > 0 and curved_xs:
        for i in range(num_random_columns):
            # Generate random points within curved bounds
            rand_x = rng.uniform(min(curved_xs), max(curved_xs))
            rand_y = rng.uniform(min(curved_ys), max(curved_ys))
            
            # Apply rotation
            x_rot = rand_x * cos_theta - rand_y * sin_theta
            y_rot = rand_x * sin_theta + rand_y * cos_theta
            
            points.append({
                "x_mm": x_rot + ox,
                "y_mm": y_rot + oy,
                "u": -1,
                "v": -1
            })

    # Calculate effective radius if k != 0
    radius_mm = abs(1.0 / k) if k != 0 else float('inf')
    
    return {
        "R": R, "C": C,
        "row_gaps_mm": row_gaps,
        "col_gaps_mm": col_gaps,
        "rotation_deg": rotation_deg,
        "curvature_k": k,
        "curvature_radius_mm": radius_mm,
        "origin": origin,        
        "num_random_columns": num_random_columns,
        "num_removed_points": num_remove_points,
        "structure_type": "fan",  # Structure type classification
        "u_grid_curve_orders": [2] * C,  # All columns curved (2 = quadratic/arc)
        "v_grid_curve_orders": [1] * R,  # All rows straight (1 = linear)
        "points": points

    }

def build_structure_with_rotation_exCol_removePart(rng, rows_range, cols_range, row_gap_domain_mm, col_gap_domain_mm=None, 
                                col_gap_fixed_mm=None, origin=(0.0,0.0), rotation_domain_deg=(0, 0), 
                                random_columns_range=(0, 0), remove_points_range=(0, 0), removal_probability=0.7, 
                                max_removals=3, random_axis_range=(0, 0)):
    """
    Generate one grid structure with dimensions in millimeters and optional rotation.
    
    Parameters:
    - rotation_domain_deg: tuple (min_deg, max_deg) for rotation angle in degrees
                          e.g., (-5, 5) for ±5 degrees, (0, 360) for full rotation
    - random_columns_range: tuple (min_cols, max_cols) for number of random extra columns
                           e.g., (0, 5) for 0-5 random columns, (10, 20) for 10-20 random columns
                           Note: Actual number will be capped at 10% of grid points (R × C)
    - remove_points_range: tuple (min_remove, max_remove) for number of grid points to remove
                           e.g., (0, 3) to remove 0-3 random grid positions
                           Note: Actual number will be capped at 20% of grid points (R × C)
    - random_axis_range: tuple (min_cols, max_cols) for axis-aligned random columns
                        These columns align with either U-axis (random u, v=-1) or V-axis (u=-1, random v)
                        Distribution between U and V is random (50/50 split)
                        e.g., (0, 6) for 0-6 axis-aligned columns
    """
    R = rng.randint(rows_range[0], rows_range[1])
    C = rng.randint(cols_range[0], cols_range[1])

    # Row spacing
    row_gaps = [rng.uniform(*row_gap_domain_mm) for _ in range(R-1)]
    ys = [0.0]
    for g in row_gaps: ys.append(ys[-1] + g)

    # Column spacing
    if col_gap_domain_mm:
        col_gaps = [rng.uniform(*col_gap_domain_mm) for _ in range(C-1)]
    else:
        gap = col_gap_fixed_mm if col_gap_fixed_mm else 1000.0  # default 1 m = 1000 mm
        col_gaps = [gap]*(C-1)
    xs = [0.0]
    for g in col_gaps: xs.append(xs[-1] + g)

    # Generate rotation angle
    rotation_deg = rng.uniform(*rotation_domain_deg)
    rotation_rad = math.radians(rotation_deg)
    
    # Apply rotation and translation
    ox, oy = origin
    cos_theta = math.cos(rotation_rad)
    sin_theta = math.sin(rotation_rad)
    
    # Rotate points around origin, then translate
    rotated_xs = []
    rotated_ys = []
    
    # Generate rotated coordinates in the same order as u,v assignment
    for y in ys:  # rows first (v index)
        for x in xs:  # columns second (u index)
            # Rotate around (0,0)
            x_rot = x * cos_theta - y * sin_theta
            y_rot = x * sin_theta + y * cos_theta
            # Then translate
            rotated_xs.append(x_rot + ox)
            rotated_ys.append(y_rot + oy)

    # Create points with u,v grid coordinates preserved
    points = []
    idx = 0
    for v, y in enumerate(ys):
        for u, x in enumerate(xs):
            points.append({
                "x_mm": rotated_xs[idx], 
                "y_mm": rotated_ys[idx], 
                "u": u, 
                "v": v
            })
            idx += 1

    # Generate points with substructure ID and optional removals (in local coordinates)
    removals = generate_random_removals(rng, R, C, removal_probability, max_removals)

    # Determine which grid points to remove
    removed_indices = set()
    
    if removals:
        for removal in removals:
            removal_type = removal.get('type')
            n_rows, n_cols = removal.get('size', (1, 1))
            
            if removal_type == 'corner':
                location = removal.get('location', 'top_left')
                indices = _get_corner_removal_indices(
                    R, C, n_rows, n_cols, location
                )
                removed_indices.update(indices)
                
            elif removal_type == 'side_middle':
                location = removal.get('location', 'top')
                indices = _get_side_middle_removal_indices(
                    R, C, n_rows, n_cols, location
                )
                removed_indices.update(indices)
                
            elif removal_type == 'random_rect':
                offset = removal.get('offset', (0, 0))
                indices = _get_rect_removal_indices(
                    R, C, n_rows, n_cols, offset
                )
                removed_indices.update(indices)

    # Filter out removed points from the substructure removals
    num_substructure_removed = len(removed_indices)
    if removed_indices:
        points = [p for i, p in enumerate(points) if i not in removed_indices]

    # Calculate number of grid points remaining after substructure removals
    num_grid_points = len(points)  # Use actual number of points remaining
    
    # Remove additional random grid points (capped at 20% of remaining grid points)
    max_remove_points = int(num_grid_points * 0.2)
    num_remove_points = rng.randint(remove_points_range[0], remove_points_range[1])
    num_remove_points = min(num_remove_points, max_remove_points)  # Cap at 20% of grid
    num_remove_points = min(num_remove_points, num_grid_points)  # Can't remove more than exist
    
    if num_remove_points > 0:
        # Randomly select indices to remove
        indices_to_remove = rng.sample(range(len(points)), num_remove_points)
        # Remove in reverse order to preserve indices
        for idx in sorted(indices_to_remove, reverse=True):
            points.pop(idx)
    
    # Create random columns with random count from range
    # Limit to maximum 10% of original grid points
    max_random_columns = int((R * C) * 0.1)
    num_random_columns = rng.randint(random_columns_range[0], random_columns_range[1])
    num_random_columns = min(num_random_columns, max_random_columns)  # Cap at 10% of grid
    
    for i in range(num_random_columns):        
        points.append({
                "x_mm": rng.uniform(min(rotated_xs), max(rotated_xs)), 
                "y_mm": rng.uniform(min(rotated_ys), max(rotated_ys)), 
                "u": -1, 
                "v": -1
            })
    
    # Add axis-aligned random columns (distributed between U and V axes)
    num_random_u_axis, num_random_v_axis = add_axis_aligned_random_columns(
        points, rng, R, C, xs, ys, (R * C), rotation_rad, origin, random_axis_range
    )

    return {
        "R": R, "C": C,
        "row_gaps_mm": row_gaps,
        "col_gaps_mm": col_gaps,
        "x_positions_mm": xs,  # original positions before rotation
        "y_positions_mm": ys,  # original positions before rotation
        "rotation_deg": rotation_deg,        
        "origin": origin,        
        "num_random_columns": num_random_columns,  # store how many random columns were added
        "num_random_u_axis": num_random_u_axis,  # columns with random u, v=-1
        "num_random_v_axis": num_random_v_axis,  # columns with u=-1, random v
        "num_removed_points": num_substructure_removed + num_remove_points,  # total grid points removed
        "num_removal_areas": len(removals),  # number of distinct removal areas
        "structure_type": "ortogonal",  # Structure type classification
        "u_grid_curve_orders": [1] * C,  # All columns straight (1 = linear)
        "v_grid_curve_orders": [1] * R,  # All rows straight (1 = linear)
        "points": points  # rotated and translated positions
    }
