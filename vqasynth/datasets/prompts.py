# Requires pcd objects, produced by pointcloud.py

def tall_choice(pcd_a, pcd_b):
    # Compute heights
    height_a = pcd_a.get_axis_aligned_bounding_box().get_extent()[1]
    height_b = pcd_b.get_axis_aligned_bounding_box().get_extent()[1]
    
    return "A" if height_a > height_b else "B" if height_b > height_a else "Equal"

def gap(pcd_a, pcd_b):
    # This requires a more complex spatial analysis to find the closest points or edges between objects.
    # Placeholder for simplicity
    return "Gap estimation not implemented"

def tall_short_classify(pcd_a, pcd_b):
    height_a = pcd_a.get_axis_aligned_bounding_box().get_extent()[1]
    height_b = pcd_b.get_axis_aligned_bounding_box().get_extent()[1]
    
    return "Tall-Short" if height_a > height_b else "Short-Tall" if height_b > height_a else "Equal"

def above_predicate(pcd_a, pcd_b):
    min_z_a = pcd_a.get_axis_aligned_bounding_box().get_min_bound()[2]
    max_z_b = pcd_b.get_axis_aligned_bounding_box().get_max_bound()[2]
    
    return min_z_a > max_z_b

def below_predicate(pcd_a, pcd_b):
    max_z_a = pcd_a.get_axis_aligned_bounding_box().get_max_bound()[2]
    min_z_b = pcd_b.get_axis_aligned_bounding_box().get_min_bound()[2]
    
    return max_z_a < min_z_b

def short_choice(pcd_a, pcd_b):
    return "A" if not tall_choice(pcd_a, pcd_b) == "A" else "B"

def below_diff(pcd_a, pcd_b):
    min_z_a = pcd_a.get_axis_aligned_bounding_box().get_min_bound()[2]
    min_z_b = pcd_b.get_axis_aligned_bounding_box().get_min_bound()[2]
    
    return abs(min_z_a - min_z_b)

def tall_predicate(pcd_a, pcd_b):
    return tall_choice(pcd_a, pcd_b) == "A"

def short_predicate(pcd_a, pcd_b):
    return short_choice(pcd_a, pcd_b) == "A"

def above_diff(pcd_a, pcd_b):
    max_z_a = pcd_a.get_axis_aligned_bounding_box().get_max_bound()[2]
    max_z_b = pcd_b.get_axis_aligned_bounding_box().get_max_bound()[2]
    
    return abs(max_z_a - max_z_b)

def vertical_dist(pcd_a, pcd_b):
    center_a = pcd_a.get_center()[2]
    center_b = pcd_b.get_center()[2]
    
    return abs(center_a - center_b)

def horizontal_dist(pcd_a, pcd_b):
    center_a = pcd_a.get_center()
    center_b = pcd_b.get_center()
    
    return np.sqrt((center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2)

def above_below_classify(pcd_a, pcd_b):
    if above_predicate(pcd_a, pcd_b):
        return "Above"
    elif below_predicate(pcd_a, pcd_b):
        return "Below"
    else:
        return "Uncertain"

def height(pcd):
    return pcd.get_axis_aligned_bounding_box().get_extent()[1]

def width(pcd):
    return pcd.get_axis_aligned_bounding_box().get_extent()[0]

def above_choice(pcd_a, pcd_b):
    return "A" if above_predicate(pcd_a, pcd_b) else "B" if below_predicate(pcd_a, pcd_b) else "Equal"

def below_choice(pcd_a, pcd_b):
    return "B" if below_predicate(pcd_a, pcd_b) else "A" if above_predicate(pcd_a, pcd_b) else "Equal"

def elevation(pcd):
    # Assuming the reference plane is Z=0
    min_z = pcd.get_axis_aligned_bounding_box().get_min_bound()[2]
    return min_z
