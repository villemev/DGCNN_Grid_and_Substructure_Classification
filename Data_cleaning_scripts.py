"""
Data_cleaning_scripts.py
------------------------
Lightweight training-data cleaning utilities.
Only depends on json, os, numpy and collections -- NO PyTorch required.
"""
import json
import os
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Grid fitting (pure numpy, copied from Methods.py)
# ---------------------------------------------------------------------------

def fit_grid_with_median(coords, u_indices, v_indices, distance_threshold=None):
    """
    Fit a grid to points using robust median-based estimation.
    Supports variable spacing -- stores actual positions of each grid line.

    Args:
        coords:           [N, 2] array of point coordinates
        u_indices:        [N] array of u indices  (0 = outlier, 1..n = grid lines)
        v_indices:        [N] array of v indices  (0 = outlier, 1..n = grid lines)
        distance_threshold: outlier distance threshold (auto if None)

    Returns:
        dict with 'center', 'u_axis', 'v_axis', 'u_spacing', 'v_spacing',
                  'u_positions', 'v_positions', 'inlier_mask', 'residuals'
    """
    coords    = np.array(coords)
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)

    grid_mask = (u_indices > 0) & (v_indices > 0)

    if grid_mask.sum() < 4:
        return {
            'center': np.mean(coords, axis=0),
            'u_axis': np.array([1.0, 0.0]),
            'v_axis': np.array([0.0, 1.0]),
            'u_spacing': 1.0,
            'v_spacing': 1.0,
            'u_positions': {},
            'v_positions': {},
            'inlier_mask': grid_mask,
            'residuals': np.zeros(len(coords)),
        }

    grid_coords = coords[grid_mask]
    grid_u      = u_indices[grid_mask]
    grid_v      = v_indices[grid_mask]

    center   = np.median(grid_coords, axis=0)
    unique_u = np.unique(grid_u)
    unique_v = np.unique(grid_v)

    if len(unique_u) >= 2 and len(unique_v) >= 2:
        u_directions = []
        for v_val in unique_v:
            v_pts   = grid_coords[grid_v == v_val]
            v_u_idx = grid_u[grid_v == v_val]
            if len(v_pts) >= 2:
                s  = np.argsort(v_u_idx)
                sp = v_pts[s];  su = v_u_idx[s]
                for i in range(len(sp) - 1):
                    step = su[i+1] - su[i]
                    if step != 0:
                        u_directions.append((sp[i+1] - sp[i]) / step)

        v_directions = []
        for u_val in unique_u:
            u_pts   = grid_coords[grid_u == u_val]
            u_v_idx = grid_v[grid_u == u_val]
            if len(u_pts) >= 2:
                s  = np.argsort(u_v_idx)
                sp = u_pts[s];  sv = u_v_idx[s]
                for i in range(len(sp) - 1):
                    step = sv[i+1] - sv[i]
                    if step != 0:
                        v_directions.append((sp[i+1] - sp[i]) / step)

        if u_directions:
            u_ax_raw = np.median(np.array(u_directions), axis=0)
            u_spacing = np.linalg.norm(u_ax_raw)
            u_axis = u_ax_raw / u_spacing if u_spacing > 1e-6 else np.array([1.0, 0.0])
        else:
            u_axis = np.array([1.0, 0.0]);  u_spacing = 1.0

        if v_directions:
            v_ax_raw = np.median(np.array(v_directions), axis=0)
            v_spacing = np.linalg.norm(v_ax_raw)
            v_axis = v_ax_raw / v_spacing if v_spacing > 1e-6 else np.array([0.0, 1.0])
        else:
            v_axis = np.array([0.0, 1.0]);  v_spacing = 1.0
    else:
        centered = grid_coords - center
        cov = np.cov(centered.T)
        _, eigvecs = np.linalg.eigh(cov)
        u_axis = eigvecs[:, 1];  v_axis = eigvecs[:, 0]
        u_proj = centered @ u_axis;  v_proj = centered @ v_axis
        u_spacing = float(np.median(np.abs(np.diff(np.sort(u_proj))))) if len(u_proj) > 1 else 1.0
        v_spacing = float(np.median(np.abs(np.diff(np.sort(v_proj))))) if len(v_proj) > 1 else 1.0

    cos_uv = float(np.dot(u_axis, v_axis))
    denom  = 1.0 - cos_uv ** 2
    if abs(denom) < 1e-8:
        cos_uv = 0.0;  denom = 1.0

    u_coord_lists = defaultdict(list)
    v_coord_lists = defaultdict(list)
    for pt, ui, vi in zip(grid_coords, grid_u, grid_v):
        c  = pt - center
        up = c @ u_axis;  vp = c @ v_axis
        a  = (up - cos_uv * vp) / denom
        b  = (vp - cos_uv * up) / denom
        u_coord_lists[int(ui)].append(a)
        v_coord_lists[int(vi)].append(b)

    u_positions = {k: float(np.median(v)) for k, v in u_coord_lists.items()}
    v_positions = {k: float(np.median(v)) for k, v in v_coord_lists.items()}

    residuals = np.zeros(len(coords))
    for i, (coord, ui, vi) in enumerate(zip(coords, u_indices, v_indices)):
        if ui > 0 and vi > 0:
            a = u_positions.get(int(ui), (ui - 1) * u_spacing)
            b = v_positions.get(int(vi), (vi - 1) * v_spacing)
            expected = center + a * u_axis + b * v_axis
        elif ui == 0 and vi > 0:
            b = v_positions.get(int(vi), (vi - 1) * v_spacing)
            expected = center + b * v_axis
        elif ui > 0 and vi == 0:
            a = u_positions.get(int(ui), (ui - 1) * u_spacing)
            expected = center + a * u_axis
        else:
            expected = center
        residuals[i] = np.linalg.norm(coord - expected)

    if distance_threshold is None:
        gr = residuals[grid_mask]
        if len(gr) > 0:
            mad = np.median(np.abs(gr - np.median(gr)))
            distance_threshold = np.median(gr) + 5 * mad
        else:
            distance_threshold = np.inf

    return {
        'center': center,
        'u_axis': u_axis,
        'v_axis': v_axis,
        'u_spacing': u_spacing,
        'v_spacing': v_spacing,
        'u_positions': u_positions,
        'v_positions': v_positions,
        'inlier_mask': residuals < distance_threshold,
        'residuals': residuals,
        'threshold': distance_threshold,
    }


def classify_points_by_grid_geometry(coords, grid_fit, tolerance_mm=50):
    """
    Classify points by perpendicular distance to fitted grid lines.

    Returns dict with:
        'categories'         -- array of 'grid_cross' / 'u_axis' / 'v_axis' / 'outlier'
        'distance_to_u_line' -- [N] distances to nearest U-line
        'distance_to_v_line' -- [N] distances to nearest V-line
        'nearest_u_idx'      -- [N] nearest U-line index (-1 if none)
        'nearest_v_idx'      -- [N] nearest V-line index (-1 if none)
    """
    coords      = np.array(coords)
    n           = len(coords)
    center      = grid_fit['center']
    u_axis      = grid_fit['u_axis']
    v_axis      = grid_fit['v_axis']
    u_positions = grid_fit.get('u_positions', {})
    v_positions = grid_fit.get('v_positions', {})

    if len(u_positions) == 0 or len(v_positions) == 0:
        return {
            'categories':         np.array(['outlier'] * n),
            'distance_to_u_line': np.full(n, np.inf),
            'distance_to_v_line': np.full(n, np.inf),
            'nearest_u_idx':      np.full(n, -1, dtype=int),
            'nearest_v_idx':      np.full(n, -1, dtype=int),
        }

    u_idxs = sorted(u_positions.keys())
    v_idxs = sorted(v_positions.keys())

    dist_u   = np.zeros(n);  near_u = np.zeros(n, dtype=int)
    dist_v   = np.zeros(n);  near_v = np.zeros(n, dtype=int)
    categories = []

    for i, coord in enumerate(coords):
        # Distance to nearest U-line (lines parallel to v_axis)
        best_du = np.inf;  best_ui = -1
        for ui in u_idxs:
            lp  = center + u_positions[ui] * u_axis
            vec = coord - lp
            d   = np.linalg.norm(vec - np.dot(vec, v_axis) * v_axis)
            if d < best_du:
                best_du = d;  best_ui = ui
        dist_u[i] = best_du;  near_u[i] = best_ui

        # Distance to nearest V-line (lines parallel to u_axis)
        best_dv = np.inf;  best_vi = -1
        for vi in v_idxs:
            lp  = center + v_positions[vi] * v_axis
            vec = coord - lp
            d   = np.linalg.norm(vec - np.dot(vec, u_axis) * u_axis)
            if d < best_dv:
                best_dv = d;  best_vi = vi
        dist_v[i] = best_dv;  near_v[i] = best_vi

        on_u = dist_u[i] < tolerance_mm
        on_v = dist_v[i] < tolerance_mm
        if on_u and on_v:
            categories.append('grid_cross')
        elif on_u:
            categories.append('u_axis')
        elif on_v:
            categories.append('v_axis')
        else:
            categories.append('outlier')

    return {
        'categories':         np.array(categories),
        'distance_to_u_line': dist_u,
        'distance_to_v_line': dist_v,
        'nearest_u_idx':      near_u,
        'nearest_v_idx':      near_v,
    }


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_training_json_axis_conflicts(input_json_path, threshold_mm,
                                       output_json_path=None, skip_curved=True):
    """
    Remove ambiguous points from a training JSON using axis-proximity rules.

    Removal rules (raw label convention: -1 = outlier / unassigned):
      - Outlier   (u=-1, v=-1)  removed if closer than threshold to ANY U- or V-line.
      - U-axis pt (u>=0, v=-1)  removed if closer than threshold to a V-line.
      - V-axis pt (u=-1, v>=0)  removed if closer than threshold to a U-line.

    Args:
        input_json_path:  Path to the source JSON file.
        threshold_mm:     Distance threshold in mm (must be > 0).
        output_json_path: Output path.  None → '<name>_axis_cleaned.json'.
        skip_curved:      If True, 'fan' / 'arc' / 'curved' structures are kept untouched.

    Returns:
        dict with keys: 'input_json_path', 'output_json_path', 'threshold_mm',
                        'stats' (global counts), 'per_structure_stats' (list of dicts).
    """
    if threshold_mm <= 0:
        raise ValueError("threshold_mm must be > 0")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'structures' not in data or not isinstance(data['structures'], list):
        raise ValueError("Input JSON must contain a top-level 'structures' list")

    if output_json_path is None:
        root, ext = os.path.splitext(input_json_path)
        output_json_path = f"{root}_axis_cleaned{ext or '.json'}"

    global_stats = dict(
        structures_total=0, structures_cleaned=0,
        points_before=0, points_after=0, removed_total=0,
        removed_outlier_near_axis=0,
        removed_u_axis_near_v_axis=0,
        removed_v_axis_near_u_axis=0,
        skipped_structures=0,
    )
    per_structure_stats = []
    curved_tags = {'fan', 'arc', 'curved'}

    for struct_idx, struct in enumerate(data['structures']):
        points   = struct.get('points', [])
        n_before = len(points)
        global_stats['structures_total'] += 1
        global_stats['points_before']    += n_before

        def _skip(reason):
            global_stats['skipped_structures'] += 1
            global_stats['points_after']       += n_before
            per_structure_stats.append(dict(
                structure=struct_idx, points_before=n_before, points_after=n_before,
                removed_total=0, removed_outlier_near_axis=0,
                removed_u_axis_near_v_axis=0, removed_v_axis_near_u_axis=0,
                skipped=True, skip_reason=reason,
            ))

        if n_before == 0:
            _skip('no points');  continue

        struct_type = str(struct.get('structure_type', '')).lower()
        if skip_curved and struct_type in curved_tags:
            _skip(f"curved structure_type='{struct_type}'");  continue

        # Build arrays
        coords = [];  u_raw = [];  v_raw = []
        for pt in points:
            if 'x_mm' in pt:
                coords.append([float(pt['x_mm']), float(pt['y_mm'])])
            else:
                coords.append([float(pt['x']),    float(pt['y'])])
            u_raw.append(int(pt.get('u', -1)))
            v_raw.append(int(pt.get('v', -1)))

        coords = np.asarray(coords, dtype=np.float32)
        u_raw  = np.asarray(u_raw,  dtype=np.int32)
        v_raw  = np.asarray(v_raw,  dtype=np.int32)

        # Fit grid  (cleaner uses 0=outlier convention internally)
        grid_fit = fit_grid_with_median(coords, u_raw + 1, v_raw + 1)
        if not grid_fit.get('u_positions') or not grid_fit.get('v_positions'):
            _skip('insufficient grid lines for geometric cleaning');  continue

        geo   = classify_points_by_grid_geometry(coords, grid_fit, tolerance_mm=threshold_mm)
        du    = geo['distance_to_u_line']
        dv    = geo['distance_to_v_line']

        is_outlier = (u_raw == -1) & (v_raw == -1)
        is_u_axis  = (u_raw >= 0)  & (v_raw == -1)
        is_v_axis  = (u_raw == -1) & (v_raw >= 0)

        rm_outlier = is_outlier & ((du < threshold_mm) | (dv < threshold_mm))
        rm_u_axis  = is_u_axis  & (dv < threshold_mm)
        rm_v_axis  = is_v_axis  & (du < threshold_mm)
        remove     = rm_outlier | rm_u_axis | rm_v_axis

        n_rm_out = int(rm_outlier.sum())
        n_rm_u   = int(rm_u_axis.sum())
        n_rm_v   = int(rm_v_axis.sum())
        n_rm     = int(remove.sum())

        if n_rm > 0:
            struct['points'] = [pt for i, pt in enumerate(points) if not remove[i]]
            global_stats['structures_cleaned'] += 1

        n_after = len(struct['points'])
        global_stats['points_after']               += n_after
        global_stats['removed_total']              += n_rm
        global_stats['removed_outlier_near_axis']  += n_rm_out
        global_stats['removed_u_axis_near_v_axis'] += n_rm_u
        global_stats['removed_v_axis_near_u_axis'] += n_rm_v

        per_structure_stats.append(dict(
            structure=struct_idx, points_before=n_before, points_after=n_after,
            removed_total=n_rm, removed_outlier_near_axis=n_rm_out,
            removed_u_axis_near_v_axis=n_rm_u, removed_v_axis_near_u_axis=n_rm_v,
            skipped=False,
        ))

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return {
        'input_json_path':    input_json_path,
        'output_json_path':   output_json_path,
        'threshold_mm':       float(threshold_mm),
        'stats':              global_stats,
        'per_structure_stats': per_structure_stats,
    }
