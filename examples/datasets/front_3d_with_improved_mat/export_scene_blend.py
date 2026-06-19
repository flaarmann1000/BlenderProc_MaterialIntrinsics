import blenderproc as bproc

r"""
export_scene_blend.py
=======================

Loads a 3D-FRONT scene with BlenderProc - using the same scene-loading and
material-assignment logic as render_front3d_multipass.py (including the
optional CC0 floor/wall materials and the confirmed-metallic floor list) -
and saves the result as a .blend file, so it can be opened directly in
Blender for manual inspection (e.g. checking materials, room layout, or
debugging why a render looked off, without having to re-run a full render).

No camera poses or lights are added - just the raw furnished scene as
BlenderProc loaded it.

Usage:
  blenderproc run export_scene_blend.py \
      [SCENE_ID] [FUTURE_MODEL_DIR] [FRONT_TEXTURE_DIR] [OUTPUT_BLEND_PATH] \
      [--front_json_dir FRONT_JSON_DIR] \
      [--cc_material_path CC_TEXTURES_DIR] [--metal_material_list PATH]

  SCENE_ID is just the scene's id (the .json filename without extension,
  e.g. 00ad8345-45e0-45b3-867d-4a3c88c2517a) - the actual file is looked up
  as <front_json_dir>/<scene_id>.json. A full path or filename also works
  (only the id portion is used).

  All positional/optional arguments default to Felix's actual dataset paths
  if omitted - pass your own values to override any of them. If
  OUTPUT_BLEND_PATH is a directory (or omitted), the file is named
  <scene_id>.blend and placed there; if it ends in .blend, that exact path
  is used.

Example:
  blenderproc run export_scene_blend.py
  blenderproc run export_scene_blend.py 00ad8345-45e0-45b3-867d-4a3c88c2517a --cc_material_path none
"""

import argparse
import os
import random

import bpy
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("scene_id", nargs="?",
                     default="00ad8345-45e0-45b3-867d-4a3c88c2517a",
                     help="3D-FRONT scene id (the .json filename without extension). "
                          "Looked up as <front_json_dir>/<scene_id>.json")
parser.add_argument("--front_json_dir", default=r"E:\3D-Front\3D-FRONT",
                     help="Folder containing 3D-FRONT scene .json files")
parser.add_argument("future_folder", nargs="?", default=r"E:\3D-Front\3D-FUTURE-model",
                     help="Path to the 3D-FUTURE-model folder")
parser.add_argument("front_texture", nargs="?", default=r"E:\3D-Front\3D-FRONT-texture",
                     help="Path to the 3D-FRONT-texture folder")
parser.add_argument("output_path", nargs="?", default="output",
                     help="Output .blend file path, or a directory to place "
                          "<scene_id>.blend into")
parser.add_argument("--cc_material_path", default=r"E:\3D-Front\cctextures",
                     help="Path to a cctextures folder for floor/wall PBR materials, applied "
                          "the same way as render_front3d_multipass.py. "
                          "Pass --cc_material_path none to disable.")
parser.add_argument("--metal_material_list",
                     default=r"C:\Users\felix\Documents\BlenderProc\examples\datasets\front_3d_with_improved_mat\metal_material_paths.txt",
                     help="Same confirmed-metallic asset list used by render_front3d_multipass.py "
                          "for floor assignment. Pass --metal_material_list none to fall back to "
                          "the general 'Metal' category.")
args = parser.parse_args()

if args.cc_material_path and args.cc_material_path.strip().lower() in ("none", ""):
    args.cc_material_path = None
if args.metal_material_list and args.metal_material_list.strip().lower() in ("none", ""):
    args.metal_material_list = None

# scene_id may be passed bare ("00ad8345-...") or as a path/filename someone
# copy-pasted (".../00ad8345-....json") - strip down to just the id either way.
scene_id = os.path.splitext(os.path.basename(args.scene_id))[0]

# Resolve paths to absolute now, before bproc.init() can change the working
# directory (see render_front3d_multipass.py for why this matters).
args.front_json_dir = os.path.abspath(args.front_json_dir)
args.future_folder = os.path.abspath(args.future_folder)
args.front_texture = os.path.abspath(args.front_texture)
args.output_path = os.path.abspath(args.output_path)
if args.cc_material_path:
    args.cc_material_path = os.path.abspath(args.cc_material_path)
if args.metal_material_list:
    args.metal_material_list = os.path.abspath(args.metal_material_list)

front_json = os.path.join(args.front_json_dir, f"{scene_id}.json")
if not os.path.exists(front_json):
    raise FileNotFoundError(
        f"No scene .json found for scene_id '{scene_id}' at expected path: {front_json}\n"
        f"Check --front_json_dir (currently: {args.front_json_dir}) and the scene_id spelling."
    )

if args.output_path.lower().endswith(".blend"):
    blend_path = args.output_path
else:
    blend_path = os.path.join(args.output_path, f"{scene_id}.blend")
os.makedirs(os.path.dirname(blend_path), exist_ok=True)


bproc.init()

# ---------------------------------------------------------------------------
# Load the scene (identical logic to render_front3d_multipass.py)
# ---------------------------------------------------------------------------
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

loaded_objects = bproc.loader.load_front3d(
    json_path=front_json,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_texture,
    label_mapping=mapping,
    ceiling_light_strength=0.0,
    lamp_light_strength=0.0,
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

# ---------------------------------------------------------------------------
# Save as .blend, packing all external textures into the file itself so it's
# fully self-contained and portable (won't break if moved away from the
# original 3D-FUTURE-model/3D-FRONT-texture/cctextures folders).
# ---------------------------------------------------------------------------
print("Packing external resources (textures) into the .blend file...")
bpy.ops.file.pack_all()

print(f"Saving to: {blend_path}")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print("Done.")
