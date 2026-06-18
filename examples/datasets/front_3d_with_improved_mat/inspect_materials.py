import blenderproc as bproc

"""
inspect_materials.py
=====================

Loads a 3D-FRONT scene (no camera/rendering needed) and prints a summary of
every material's Roughness and Metallic inputs - whether each is a constant
value or driven by a texture/node, and flags any material where no top-level
Principled BSDF could be found at all (which would silently get NO AOV node
in render_front3d_multipass.py, i.e. render as black in those passes).

Pass --cc_material_path to also apply CC0 floor/wall materials exactly like
render_front3d_multipass.py does, so you can verify those show up as
texture-driven entries in the table.

Usage:
  blenderproc run inspect_materials.py [FRONT_JSON] [FUTURE_MODEL_DIR] [FRONT_TEXTURE_DIR] \
      [--cc_material_path CC_TEXTURES_DIR]

  All positional/optional arguments default to Felix's actual dataset paths
  if omitted - pass your own values to override any of them.
"""

import argparse
import os
import random

import bpy
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("front_json", nargs="?",
                     default=r"E:\3D-Front\3D-FRONT\00ad8345-45e0-45b3-867d-4a3c88c2517a.json")
parser.add_argument("future_folder", nargs="?", default=r"E:\3D-Front\3D-FUTURE-model")
parser.add_argument("front_texture", nargs="?", default=r"E:\3D-Front\3D-FRONT-texture")
parser.add_argument("--cc_material_path", default=r"E:\3D-Front\cctextures",
                     help="Optional path to a cctextures folder, applied the same way as in "
                          "render_front3d_multipass.py (floors + occasional marble walls). "
                          "Pass --cc_material_path none to disable.")
parser.add_argument("--metal_material_list",
                     default=r"C:\Users\felix\Documents\BlenderProc\examples\datasets\front_3d_with_improved_mat\metal_material_paths.txt",
                     help="Same confirmed-metallic asset list used by render_front3d_multipass.py "
                          "for floor assignment. Pass --metal_material_list none to fall back to "
                          "the general 'Metal' category instead.")
args = parser.parse_args()

if args.cc_material_path and args.cc_material_path.strip().lower() in ("none", ""):
    args.cc_material_path = None
if args.metal_material_list and args.metal_material_list.strip().lower() in ("none", ""):
    args.metal_material_list = None

bproc.init()

mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

loaded_objects = bproc.loader.load_front3d(
    json_path=args.front_json,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_texture,
    label_mapping=mapping,
)

if args.cc_material_path:
    def asset_name_from_path(path):
        path = path.strip()
        if not path:
            return None
        if os.path.splitext(path)[1]:
            path = os.path.dirname(path)
        return os.path.basename(path.rstrip("\\/")) or None

    metal_materials = None
    if args.metal_material_list and os.path.exists(args.metal_material_list):
        with open(args.metal_material_list, "r", encoding="utf-8") as f:
            metal_asset_names = sorted({
                name for name in (asset_name_from_path(line) for line in f) if name
            })
        print(f"Read {len(metal_asset_names)} confirmed-metallic asset name(s) from {args.metal_material_list}")
        if metal_asset_names:
            metal_materials = bproc.loader.load_ccmaterials(args.cc_material_path, metal_asset_names)
            print(f"  -> matched {len(metal_materials)} loaded CC material(s)")

    if not metal_materials:
        print("No confirmed-metallic list available/matched - falling back to the 'Metal' category.")
        metal_materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Metal"])

    general_cc_materials = bproc.loader.load_ccmaterials(
        args.cc_material_path, ["Bricks", "Wood", "Carpet", "Tile", "Marble"]
    )

    floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
    for floor in floors:
        for i in range(len(floor.get_materials())):
            if metal_materials and np.random.uniform(0, 1) <= 0.95:
                floor.set_material(i, random.choice(metal_materials))

    marble_materials = [m for m in general_cc_materials if "Marble" in m.get_name()]
    if marble_materials:
        walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
        for wall in walls:
            for i in range(len(wall.get_materials())):
                if np.random.uniform(0, 1) <= 0.1:
                    wall.set_material(i, random.choice(marble_materials))


def find_principled_bsdf(node_tree, depth=0):
    """Search the top level of a node tree for a Principled BSDF. Also peeks
    one level into any node groups, and reports if it had to do so (this is
    only used here for diagnostic reporting - render_front3d_multipass.py
    deliberately does NOT follow into groups, since AOV nodes can't be wired
    across separate node trees)."""
    for n in node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n, depth
    if depth == 0:
        for n in node_tree.nodes:
            if n.type == "GROUP" and n.node_tree is not None:
                found, d = find_principled_bsdf(n.node_tree, depth=1)
                if found is not None:
                    return found, d
    return None, depth


def describe_socket(socket):
    if not socket.is_linked:
        return f"constant={socket.default_value:.3f}"
    src_node = socket.links[0].from_node
    if src_node.type == "TEX_IMAGE":
        img = src_node.image
        name = img.name if img is not None else "?"
        size = f"{img.size[0]}x{img.size[1]}" if img is not None else "?"
        return f"texture-driven (image '{name}', {size})"
    return f"node-driven (via {src_node.type} node '{src_node.name}')"


rows = []
no_bsdf_found = []

for mat in bpy.data.materials:
    if not mat.use_nodes or mat.node_tree is None:
        continue
    bsdf, depth = find_principled_bsdf(mat.node_tree)
    if bsdf is None:
        no_bsdf_found.append(mat.name)
        continue
    flag = "  (nested in group!)" if depth > 0 else ""
    rows.append((
        mat.name + flag,
        describe_socket(bsdf.inputs["Roughness"]),
        describe_socket(bsdf.inputs["Metallic"]),
    ))

print(f"\n{'Material':45s} | {'Roughness':35s} | {'Metallic'}")
print("-" * 130)
for name, rough, metal in sorted(rows):
    print(f"{name[:45]:45s} | {rough[:35]:35s} | {metal}")

print(f"\nMaterials with a usable Principled BSDF: {len(rows)}")
nested = sum(1 for n, _, _ in rows if "nested in group" in n)
if nested:
    print(f"  - {nested} of those had it nested inside a node group (one level deep)")

const_metal = [r for r in rows if r[2].startswith("constant")]
nonzero_const_metal = [r for r in const_metal if not r[2].startswith("constant=0.0")]
print(f"  - constant metallic values: {len(const_metal)} ({len(nonzero_const_metal)} are non-zero)")
print(f"  - texture/node-driven metallic: {len(rows) - len(const_metal)}")

if no_bsdf_found:
    print(f"\n*** {len(no_bsdf_found)} materials have NO usable Principled BSDF found "
          f"(would render as black in roughness/metallic AOV passes): ***")
    for name in no_bsdf_found:
        print(f"  - {name}")