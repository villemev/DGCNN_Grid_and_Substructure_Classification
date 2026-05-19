import random
import math
rng = random.Random(42)

# Import structure generation functions from Training_sett_scripts
from Training_sett_scripts import (
    build_structure_with_rotation_exCol,
    build_structure_with_skew,
    build_structure_with_arc_curvature,
    build_structure_with_rotation_exCol_removePart,
)

def build_manual_structure(rng, num_substructures=2, rows_range=(3,12), cols_range=(2,15), 
                         row_gap_domain_mm=(5000,8000), col_gap_domain_mm=(5000,8000), 
                         rotation_domain_deg=(-45, 45), removal_probability=0.7, 
                         same_spacing_probability=0, random_columns_range=(0, 3),
                         remove_points_range=(0, 3), align_grid_spacing=True,
                         align_skew_rotation=True, skew_domain_deg=(-15, 15),
                         vertical_shift_range=(0, 3),
                         structure_type_weights=None):
    """
    Generate one multi-structure with multiple non-overlapping substructures using
    structure generation functions from Training_sett_scripts.
    
    Parameters:
    - num_substructures: Number of substructures to generate
    - rows_range: Range for number of rows (R)
    - cols_range: Range for number of columns (C)
    - row_gap_domain_mm: Range for row spacing in millimeters
    - col_gap_domain_mm: Range for column spacing in millimeters
    - rotation_domain_deg: Range for rotation of entire assembly in degrees
    - removal_probability: Probability of removing sections (for removePart structures)
    - same_spacing_probability: Probability that substructure uses same spacing as main
    - random_columns_range: Range for number of random extra columns
    - remove_points_range: Range for number of random points to remove
    - align_grid_spacing: If True, all substructures use the same fixed grid spacing (one value from domain)
    - align_skew_rotation: If True, skewed structures are rotated to align column direction with other structures
    - skew_domain_deg: Range for skew angle in degrees for skewed structures
    - vertical_shift_range: Range for vertical shift in units of column spacing (can be negative to positive)
    - structure_type_weights: Dict with weights for each structure type, e.g. {'ortogonal': 0.4, 'skewed': 0.3, 'fan': 0.2, 'ortogonal_removePart': 0.1}
                              If None, all types have equal probability
    
    Returns a single structure dict that contains multiple substructures in the same coordinate space.
    Each point has a substructure_id to identify which substructure it belongs to.
    """
    
    # Choose structure types to use
    structure_generators = [
        ('ortogonal', build_structure_with_rotation_exCol),
        ('skewed', build_structure_with_skew),
        ('fan', build_structure_with_arc_curvature),
        ('ortogonal_removePart', build_structure_with_rotation_exCol_removePart)
    ]
    
    # Set up weights for structure type selection
    if structure_type_weights is None:
        # Equal probability for all types
        weights = [1.0, 1.0, 1.0, 1.0]
    else:
        # Use provided weights
        weights = [
            structure_type_weights.get('ortogonal', 0.0),
            structure_type_weights.get('skewed', 0.0),
            structure_type_weights.get('fan', 0.0),
            structure_type_weights.get('ortogonal_removePart', 0.0)
        ]
    
    all_points = []
    substructure_info = []
    
    # Generate MAIN structure first (substructure_id = 0)
    main_structure_type, main_generator = rng.choices(structure_generators, weights=weights, k=1)[0]
    
    # If align_grid_spacing is True, pick unique spacing for this substructure
    if align_grid_spacing:
        main_row_gap = rng.uniform(*row_gap_domain_mm)
        main_col_gap = rng.uniform(*col_gap_domain_mm)
        main_row_domain = (main_row_gap, main_row_gap)
        main_col_domain = (main_col_gap, main_col_gap)
    else:
        main_row_domain = row_gap_domain_mm
        main_col_domain = col_gap_domain_mm
    
    # Build main structure with no rotation (will rotate entire assembly later)
    main_skew_angle = 0.0  # Track skew angle for alignment
    
    if main_structure_type == 'skewed':
        # Generate unique skew angle for this structure
        main_skew_angle = rng.uniform(*skew_domain_deg)
        
        # If align_skew_rotation is True, apply counter-rotation to align columns
        # Use half the skew angle for correct geometric alignment
        main_local_rotation = main_skew_angle if align_skew_rotation else 0.0
        
        main_struct = main_generator(
            rng=rng,
            rows_range=rows_range,
            cols_range=cols_range,
            row_gap_domain_mm=main_row_domain,
            col_gap_domain_mm=main_col_domain,
            rotation_domain_deg=(main_local_rotation, main_local_rotation),
            skew_domain_deg=(main_skew_angle, main_skew_angle),
            random_columns_range=random_columns_range,
            remove_points_range=remove_points_range,
            origin=(0.0, 0.0)
        )
    elif main_structure_type == 'fan':
        main_struct = main_generator(
            rng=rng,
            rows_range=rows_range,
            cols_range=cols_range,
            row_gap_domain_mm=main_row_domain,
            col_gap_domain_mm=main_col_domain,
            arc_curvature_domain=(-0.00001, 0.00001),
            rotation_domain_deg=(0, 0),  # No rotation yet
            random_columns_range=random_columns_range,
            remove_points_range=remove_points_range,
            origin=(0.0, 0.0)
        )
    elif main_structure_type == 'ortogonal_removePart':
        main_struct = main_generator(
            rng=rng,
            rows_range=rows_range,
            cols_range=cols_range,
            row_gap_domain_mm=main_row_domain,
            col_gap_domain_mm=main_col_domain,
            rotation_domain_deg=(0, 0),  # No rotation yet
            random_columns_range=random_columns_range,
            remove_points_range=remove_points_range,
            removal_probability=removal_probability,
            origin=(0.0, 0.0)
        )
    else:  # ortogonal
        main_struct = main_generator(
            rng=rng,
            rows_range=rows_range,
            cols_range=cols_range,
            row_gap_domain_mm=main_row_domain,
            col_gap_domain_mm=main_col_domain,
            rotation_domain_deg=(0, 0),  # No rotation yet
            random_columns_range=random_columns_range,
            remove_points_range=remove_points_range,
            origin=(0.0, 0.0)
        )
    
    # Add substructure_id to main structure points
    for point in main_struct["points"]:
        point["substructure_id"] = 0
    
    all_points.extend(main_struct["points"])
    
    # Calculate rightmost extent for positioning next substructure
    rightmost_x = max(p["x_mm"] for p in main_struct["points"]) if main_struct["points"] else 0
    
    # Store main structure info
    substructure_info.append({
        "substructure_id": 0,
        "R": main_struct["R"],
        "C": main_struct["C"],
        "structure_type": main_struct["structure_type"],
        "rotation_deg": 0.0,
        "origin": (0.0, 0.0),
        "points_count": len(main_struct["points"]),
        "is_main": True
    })
    
    total_substructures_placed = 1
    
    # Generate additional substructures
    for sub_id in range(1, num_substructures):
        # Choose structure type for this substructure
        structure_type, generator = rng.choices(structure_generators, weights=weights, k=1)[0]
        
        # Decide if using same spacing as main
        use_same_spacing = rng.random() < same_spacing_probability
        
        # Generate unique spacing for this substructure if align_grid_spacing is True
        if align_grid_spacing:
            sub_row_gap_value = rng.uniform(*row_gap_domain_mm)
            sub_col_gap_value = rng.uniform(*col_gap_domain_mm)
            sub_row_domain = (sub_row_gap_value, sub_row_gap_value)
            sub_col_domain = (sub_col_gap_value, sub_col_gap_value)
        else:
            sub_row_domain = row_gap_domain_mm
            sub_col_domain = col_gap_domain_mm
        
        if use_same_spacing:
            # Use same or similar dimensions
            sub_row_range = (max(rows_range[0], main_struct["R"] - 2), 
                           min(rows_range[1], main_struct["R"] + 2))
            sub_col_range = (max(cols_range[0], main_struct["C"] - 2), 
                           min(cols_range[1], main_struct["C"] + 2))
        else:
            sub_row_range = rows_range
            sub_col_range = cols_range
        
        # Calculate gap between structures (average of main structure's column gaps)
        avg_gap = sum(main_struct["col_gaps_mm"]) / len(main_struct["col_gaps_mm"]) if main_struct["col_gaps_mm"] else 5000
        
        # Apply vertical shift (in units of column spacing)
        vertical_shift_units = rng.uniform(*vertical_shift_range)
        vertical_shift_mm = vertical_shift_units * avg_gap
        
        origin_x = rightmost_x + avg_gap
        origin_y = vertical_shift_mm
        
        # Build substructure with appropriate parameters
        sub_skew_angle = 0.0  # Track skew angle for this substructure
        
        if structure_type == 'skewed':
            # Generate unique skew angle for this substructure
            sub_skew_angle = rng.uniform(*skew_domain_deg)
            
            # If align_skew_rotation is True, apply counter-rotation to align columns
            # Use half the skew angle for correct geometric alignment
            sub_local_rotation = sub_skew_angle if align_skew_rotation else 0.0
            
            sub_struct = generator(
                rng=rng,
                rows_range=sub_row_range,
                cols_range=sub_col_range,
                row_gap_domain_mm=sub_row_domain,
                col_gap_domain_mm=sub_col_domain,
                rotation_domain_deg=(sub_local_rotation, sub_local_rotation),
                skew_domain_deg=(sub_skew_angle, sub_skew_angle),
                random_columns_range=random_columns_range,
                remove_points_range=remove_points_range,
                origin=(origin_x, origin_y)
            )
        elif structure_type == 'fan':
            sub_struct = generator(
                rng=rng,
                rows_range=sub_row_range,
                cols_range=sub_col_range,
                row_gap_domain_mm=sub_row_domain,
                col_gap_domain_mm=sub_col_domain,
                arc_curvature_domain=(-0.00001, 0.00001),
                rotation_domain_deg=(0, 0),  # No rotation yet
                random_columns_range=random_columns_range,
                remove_points_range=remove_points_range,
                origin=(origin_x, origin_y)
            )
        elif structure_type == 'ortogonal_removePart':
            sub_struct = generator(
                rng=rng,
                rows_range=sub_row_range,
                cols_range=sub_col_range,
                row_gap_domain_mm=sub_row_domain,
                col_gap_domain_mm=sub_col_domain,
                rotation_domain_deg=(0, 0),  # No rotation yet
                random_columns_range=random_columns_range,
                remove_points_range=remove_points_range,
                removal_probability=removal_probability,
                origin=(origin_x, origin_y)
            )
        else:  # ortogonal
            sub_struct = generator(
                rng=rng,
                rows_range=sub_row_range,
                cols_range=sub_col_range,
                row_gap_domain_mm=sub_row_domain,
                col_gap_domain_mm=sub_col_domain,
                rotation_domain_deg=(0, 0),  # No rotation yet
                random_columns_range=random_columns_range,
                remove_points_range=remove_points_range,
                origin=(origin_x, origin_y)
            )
        
        # Add substructure_id to points
        for point in sub_struct["points"]:
            point["substructure_id"] = sub_id
        
        all_points.extend(sub_struct["points"])
        
        # Update rightmost extent
        if sub_struct["points"]:
            rightmost_x = max(p["x_mm"] for p in sub_struct["points"])
        
        # Store substructure info
        substructure_info.append({
            "substructure_id": sub_id,
            "R": sub_struct["R"],
            "C": sub_struct["C"],
            "structure_type": sub_struct["structure_type"],
            "rotation_deg": 0.0,
            "origin": (origin_x, origin_y),
            "points_count": len(sub_struct["points"]),
            "uses_same_spacing": use_same_spacing
        })
        
        total_substructures_placed += 1
    
    # Now apply rotation to the entire assembled structure
    rotation_main = rng.uniform(*rotation_domain_deg)
    rotation_rad = math.radians(rotation_main)
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)
    
    for point in all_points:
        x_local = point["x_mm"]
        y_local = point["y_mm"]
        
        # Apply rotation around origin (0, 0)
        x_rotated = x_local * cos_r - y_local * sin_r
        y_rotated = x_local * sin_r + y_local * cos_r
        
        point["x_mm"] = x_rotated
        point["y_mm"] = y_rotated
    
    # Update substructure origins after rotation
    for sub_info in substructure_info:
        origin_x, origin_y = sub_info["origin"]
        
        x_rotated = origin_x * cos_r - origin_y * sin_r
        y_rotated = origin_x * sin_r + origin_y * cos_r
        
        sub_info["origin"] = (x_rotated, y_rotated)
        sub_info["rotation_deg"] = rotation_main
    
    # Return the complete multi-structure
    return {
        "R": main_struct["R"],
        "C": main_struct["C"],
        "row_gaps_mm": main_struct["row_gaps_mm"],
        "col_gaps_mm": main_struct["col_gaps_mm"],
        "x_positions_mm": main_struct.get("x_positions_mm", []),
        "y_positions_mm": main_struct.get("y_positions_mm", []),
        "rotation_deg": rotation_main,
        "origin": [0, 0],
        "points": all_points,
        "substructures": substructure_info,
        "num_substructures": total_substructures_placed,
        "structure_type": "multi_structure"
    }