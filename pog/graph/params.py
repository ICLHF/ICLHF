# Parameter file for pog.graph
OFFSET = 0.003  # Offset (z-axis) between two objects (avoids collision between parent and child)

# for pog.graph.chromosome
FRICTION_ANGLE_THRESH = 0.1
TF_DIFF_THRESH = 0.01

# {support up: support down}
PairedSurface = {
    "box_aff_pz": "box_aff_nz",
    "box_aff_nz": "box_aff_pz",
    "box_aff_px": "box_aff_nx",
    "box_aff_nx": "box_aff_px",
    "box_aff_py": "box_aff_ny",
    "box_aff_ny": "box_aff_py",
    "cylinder_aff_nz": "cylinder_aff_pz",
    "cylinder_aff_pz": "cylinder_aff_nz",
    "shelf_aff_pz_top": "shelf_aff_nz",
    "shelf_aff_pz_bottom": "shelf_aff_nz",
    "shelf_outer_top": "shelf_outer_bottom",
    "cabinet_inner_bottom": "cabinet_outer_bottom",
    "cabinet_inner_middle": "cabinet_outer_bottom",
    "drawer_inner_bottom": "drawer_outer_bottom",
}

ContainmentSurface = [
    "shelf_aff_pz_bottom",
    "cabinet_inner_bottom",
    "cabinet_inner_middle",
    "drawer_inner_bottom",
]
