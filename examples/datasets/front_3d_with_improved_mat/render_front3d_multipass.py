import blenderproc as bproc

"""
render_front3d_multipass.py
============================

Renders a 3D-FRONT scene with BlenderProc: N camera views, each rendered
under M randomly generated lighting conditions (1-3 point/area lights per
condition, randomized position/color/energy/size), plus a lighting-
independent G-buffer captured once per view. Output layout:

  <output_dir>/<scene_id>/
    light_conditions.json
      Manifest describing the exact parameters (type/location/energy/color/
      size) of every light in every light_XXX condition, for reproducibility.
    <cam_nr>/                  (one folder per view, cam_nr = 0, 1, 2, ...)
      albedo.png                - diffuse base color   (lighting-independent)
      normals.png                - surface normals      (lighting-independent)
      roughness.png              - 16-bit material roughness (lighting-independent)
      metallic.png                - 16-bit material metallic  (lighting-independent)
      light_000.png ... light_MMM.png  - beauty render under each lighting condition

scene_id is derived from the input scene .json's filename (without extension).

Albedo/normals/roughness/metallic only depend on geometry and materials, not
on lighting, so they're rendered once for all views in a single dedicated
call before the M lighting conditions are rendered, then moved from a flat
temp folder into their final per-view folders.

NOTE on file naming: albedo/normals/roughness/metallic are all written
DIRECTLY to disk by manual compositor "File Output" nodes, bypassing
BlenderProc's own reload-by-registered-path mechanism (enable_diffuse_
color_output / enable_normals_output). This sidesteps a recurring Blender
issue on this machine where Blender appends an unexpected "_L" view suffix
to compositor-written filenames, which otherwise makes BlenderProc crash
with FileNotFoundError when it tries to reload them by their expected name.
The move-into-view-folders step is tolerant of that suffix (or any other
unexpected text Blender appends) since it only matches on the leading
channel name and frame digits.

Usage:
  blenderproc run render_front3d_multipass.py \
      [FRONT_JSON] [FUTURE_MODEL_DIR] [FRONT_TEXTURE_DIR] [OUTPUT_DIR] \
      [--cc_material_path CC_TEXTURES_DIR] [--metal_material_list PATH] \
      [--num_camera_poses N] [--num_light_setups M] [--seed N] [--resolution N]

  All positional/optional arguments default to Felix's actual dataset paths
  if omitted - pass your own values to override any of them.
"""

import argparse
import colorsys
import json
import os
import random
import re
import shutil

import bpy
import numpy as np
from PIL import Image

from blenderproc.python.utility.Utility import Utility


# ---------------------------------------------------------------------------
# CLI - defaults point at Felix's actual dataset paths; override as needed.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("front_json", nargs="?",
                     default=r"E:\3D-Front\3D-FRONT\00ad8345-45e0-45b3-867d-4a3c88c2517a.json",
                     help="Path to the 3D-FRONT scene .json file")
parser.add_argument("future_folder", nargs="?",
                     default=r"E:\3D-Front\3D-FUTURE-model",
                     help="Path to the 3D-FUTURE-model folder")
parser.add_argument("front_texture", nargs="?",
                     default=r"E:\3D-Front\3D-FRONT-texture",
                     help="Path to the 3D-FRONT-texture folder")
parser.add_argument("output_dir", nargs="?", default="output",
                     help="Where to write the dataset")
parser.add_argument("--cc_material_path", default=r"E:\3D-Front\cctextures",
                     help="Path to a cctextures folder for floor/wall PBR materials. "
                          "Pass --cc_material_path none to disable.")
parser.add_argument("--metal_material_list",
                     default=r"C:\Users\felix\Documents\BlenderProc\examples\datasets\front_3d_with_improved_mat\metal_material_paths.txt",
                     help="Text file listing paths to CC0 assets confirmed to have a real "
                          "metallic map (one path per line, file or folder paths both work - "
                          "only the asset folder name is used). The floor is assigned materials "
                          "from this list specifically, so the metallic AOV always has genuine "
                          "non-zero ground truth instead of relying on a random draw across "
                          "mostly-non-metallic categories. Pass --metal_material_list none to "
                          "fall back to drawing from the general 'Metal' category instead.")
parser.add_argument("--num_camera_poses", type=int, default=2,
                     help="Number of views (N) to sample in the scene")
parser.add_argument("--num_light_setups", type=int, default=3,
                     help="Number of random lighting conditions (M) to generate; each is "
                          "rendered across all N views")
parser.add_argument("--seed", type=int, default=None,
                     help="Optional random seed, for reproducible camera/light sampling")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                     help="If true (default), delete any existing <scene_id> output folder "
                          "before rendering, so old runs never mix with the new one. Pass "
                          "--no-overwrite to keep/merge into an existing folder instead.")
parser.add_argument("--resolution", type=int, default=124,
                     help="Square render resolution")
args = parser.parse_args()

if args.cc_material_path and args.cc_material_path.strip().lower() in ("none", ""):
    args.cc_material_path = None
if args.metal_material_list and args.metal_material_list.strip().lower() in ("none", ""):
    args.metal_material_list = None

# Resolve every path argument to an absolute path *now*, before bproc.init()
# or scene loading run - Blender/BlenderProc can change the process's
# working directory internally during initialization, so a relative path
# like the "output" default would otherwise resolve inconsistently
# depending on when it's used later in the script (this is what caused
# output to split between the local folder and C:\output).
args.front_json = os.path.abspath(args.front_json)
args.future_folder = os.path.abspath(args.future_folder)
args.front_texture = os.path.abspath(args.front_texture)
args.output_dir = os.path.abspath(args.output_dir)
if args.cc_material_path:
    args.cc_material_path = os.path.abspath(args.cc_material_path)
if args.metal_material_list:
    args.metal_material_list = os.path.abspath(args.metal_material_list)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

bproc.init()

# Defensive - see module docstring. Doesn't appear to be the actual cause of
# the "_L" suffix issue on this machine, but harmless to keep set.
bpy.context.scene.render.use_multiview = False

# ---------------------------------------------------------------------------
# Load the scene
# ---------------------------------------------------------------------------
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

loaded_objects = bproc.loader.load_front3d(
    json_path=args.front_json,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_texture,
    label_mapping=mapping,
    # Disable the dataset's built-in ceiling/lamp emission - we add our own
    # controllable lights per light setup further below instead.
    ceiling_light_strength=0.0,
    lamp_light_strength=0.0,
)

# Optional: replace floor/wall materials with higher quality CC0 textures.
#
# Most cctextures categories (wood, carpet, tile, marble, brick, fabric, ...)
# are dielectric and simply don't ship a metalness map at all - there's
# nothing spatially varying to encode when the value is uniformly zero. So
# rather than draw the floor randomly from a broad mix where 5/6 categories
# would silently give constant=0 metallic, we read a list of assets already
# confirmed to carry a real metalness map and draw the floor from that list
# specifically. This guarantees the metallic AOV has genuine non-zero ground
# truth, instead of depending on random luck. Walls still use the original
# broad pool for occasional marble-style materials, since they aren't the
# focus of metallic verification.
if args.cc_material_path:
    def asset_name_from_path(path):
        path = path.strip()
        if not path:
            return None
        if os.path.splitext(path)[1]:  # looks like a file path -> use its parent folder
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

    # Walls only occasionally get a marble-style material, matching
    # BlenderProc's own front_3d_with_improved_mat example.
    marble_materials = [m for m in general_cc_materials if "Marble" in m.get_name()]
    if marble_materials:
        walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
        for wall in walls:
            for i in range(len(wall.get_materials())):
                if np.random.uniform(0, 1) <= 0.1:
                    wall.set_material(i, random.choice(marble_materials))

mesh_objects = [o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)]

# ---------------------------------------------------------------------------
# Camera pose sampling (same logic as BlenderProc's official front_3d example)
# ---------------------------------------------------------------------------
bproc.camera.set_resolution(args.resolution, args.resolution)

point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)
bvh_tree = bproc.object.create_bvh_tree_multi_objects(mesh_objects)

proximity_checks = {"min": 1.0, "avg": {"min": 2.5, "max": 3.5}, "no_background": True}
poses, tries = 0, 0
while poses < args.num_camera_poses and tries < 10000:
    tries += 1
    height = np.random.uniform(1.4, 1.8)
    location = point_sampler.sample(height)
    rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, 2 * np.pi])
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

    if bproc.camera.scene_coverage_score(cam2world_matrix) > 0.4 \
            and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
        bproc.camera.add_camera_pose(cam2world_matrix)
        poses += 1

print(f"Sampled {poses} camera poses after {tries} tries")
if poses == 0:
    raise RuntimeError("Could not find any valid camera pose - try lowering the coverage threshold.")


# ---------------------------------------------------------------------------
# Material AOVs for roughness & metallic.
# Blender's Cycles has no built-in render pass for these (unlike normals or
# diffuse color), so we manually expose each material's Roughness/Metallic
# Principled BSDF input as an Arbitrary Output Variable (AOV) and route it
# to disk via the compositor.
# ---------------------------------------------------------------------------
def find_top_level_principled_bsdf(node_tree):
    """Only matches a Principled BSDF at the top level of this node tree.
    (A nested one inside a node group can't be wired to an AOV node created
    in the top-level tree - Blender doesn't allow linking across separate
    node trees - so such materials are skipped rather than mishandled.
    Confirmed unnecessary for both OBJ-imported and CC materials, which
    BlenderProc always builds at the top level - see chat.)"""
    for n in node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    return None


def setup_material_aovs():
    view_layer = bpy.context.view_layer
    existing = {a.name for a in view_layer.aovs}
    for name in ("Roughness", "Metallic"):
        if name not in existing:
            aov = view_layer.aovs.add()
            aov.name = name
            aov.type = "VALUE"

    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        bsdf = find_top_level_principled_bsdf(mat.node_tree)
        if bsdf is None:
            continue

        for input_name in ("Roughness", "Metallic"):
            socket = bsdf.inputs[input_name]
            aov_node = nodes.new("ShaderNodeOutputAOV")
            aov_node.aov_name = input_name

            if socket.is_linked:
                # A texture (or other node) drives this input - tap the same source.
                src = socket.links[0].from_socket
                links.new(src, aov_node.inputs["Value"])
            else:
                # It's a constant value - feed that constant into the AOV.
                val_node = nodes.new("ShaderNodeValue")
                val_node.outputs[0].default_value = socket.default_value
                links.new(val_node.outputs[0], aov_node.inputs["Value"])


def enable_aov_file_output(output_dir, aov_name, file_prefix):
    """Writes a single-channel AOV pass directly to disk as a 16-bit PNG
    (16-bit avoids the 8-bit quantization that would visibly band roughness
    gradients). Bypasses BlenderProc's registered-output reload mechanism
    entirely - we never ask BlenderProc to read this file back."""
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    render_layer_node = Utility.get_the_one_node_with_type(tree.nodes, "CompositorNodeRLayers")
    socket = render_layer_node.outputs[aov_name]

    out_node = tree.nodes.new("CompositorNodeOutputFile")
    out_node.base_path = output_dir
    out_node.format.file_format = "PNG"
    out_node.format.color_mode = "BW"
    out_node.format.color_depth = "16"
    out_node.file_slots.values()[0].path = file_prefix
    tree.links.new(socket, out_node.inputs[0])
    return out_node


def enable_diffuse_output_simple(output_dir, file_prefix="albedo_"):
    """Writes the diffuse color (albedo) pass directly to disk, bypassing
    BlenderProc's enable_diffuse_color_output (whose reload step is what
    crashes with the "_L" suffix issue on this machine)."""
    bpy.context.view_layer.use_pass_diffuse_color = True
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    render_layer_node = Utility.get_the_one_node_with_type(tree.nodes, "CompositorNodeRLayers")
    out_node = tree.nodes.new("CompositorNodeOutputFile")
    out_node.base_path = output_dir
    out_node.format.file_format = "PNG"
    out_node.file_slots.values()[0].path = file_prefix
    tree.links.new(render_layer_node.outputs["DiffCol"], out_node.inputs[0])
    return out_node


def enable_normals_output_simple(output_dir, file_prefix="normals_"):
    """Writes a simple world-space normal map directly to disk, bypassing
    BlenderProc's own enable_normals_output (which also relies on the
    reload mechanism that's crashing here). This is a simpler remap than
    BlenderProc's camera-space version (just normal*0.5+0.5 -> 0..1), but
    perfectly valid ground truth for most downstream uses."""
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    render_layer_node = Utility.get_the_one_node_with_type(tree.nodes, "CompositorNodeRLayers")

    mix_scale = tree.nodes.new("CompositorNodeMixRGB")
    mix_scale.blend_type = "MULTIPLY"
    mix_scale.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    tree.links.new(render_layer_node.outputs["Normal"], mix_scale.inputs[1])

    mix_add = tree.nodes.new("CompositorNodeMixRGB")
    mix_add.blend_type = "ADD"
    mix_add.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    tree.links.new(mix_scale.outputs["Image"], mix_add.inputs[1])

    out_node = tree.nodes.new("CompositorNodeOutputFile")
    out_node.base_path = output_dir
    out_node.format.file_format = "PNG"
    out_node.format.color_depth = "16"
    out_node.file_slots.values()[0].path = file_prefix
    tree.links.new(mix_add.outputs["Image"], out_node.inputs[0])
    return out_node


setup_material_aovs()

scene_id = os.path.splitext(os.path.basename(args.front_json))[0]
scene_dir = os.path.join(args.output_dir, scene_id)
if args.overwrite and os.path.exists(scene_dir):
    print(f"--overwrite is set: removing existing {scene_dir}")
    shutil.rmtree(scene_dir)
os.makedirs(scene_dir, exist_ok=True)
print(f"Writing output to: {scene_dir}")

gbuffer_tmp_dir = os.path.join(scene_dir, "_gbuffer_tmp")
os.makedirs(gbuffer_tmp_dir, exist_ok=True)

# All four of these ride along "for free" with one render call (Cycles
# computes them as extra channels of the same pass for every view at once).
# Blender's compositor can only write into one shared base_path per call,
# appending a frame number per view - so they land in a flat temp folder
# first, then get moved into their final <scene_id>/<cam_nr>/ folders below.
_roughness_out = enable_aov_file_output(gbuffer_tmp_dir, "Roughness", "roughness_")
_metallic_out = enable_aov_file_output(gbuffer_tmp_dir, "Metallic", "metallic_")
_albedo_out = enable_diffuse_output_simple(gbuffer_tmp_dir, "albedo_")
_normals_out = enable_normals_output_simple(gbuffer_tmp_dir, "normals_")

print("Rendering G-buffers (albedo/normals/roughness/metallic) for all views...")
# "colors" from this throwaway call is irrelevant (no lighting condition
# applied yet) and return_data=False skips reloading it entirely - the
# gbuffer files themselves are written as a side effect of the call.
bproc.renderer.render(return_data=False)


def move_gbuffers_into_view_folders(tmp_dir, dest_scene_dir, channel_prefixes):
    """Moves channel_FRAME[_suffix].ext files from a flat temp folder into
    dest_scene_dir/<cam_nr>/channel.ext. Tolerant of any unexpected filename
    suffix Blender may add (e.g. the "_L" view-suffix quirk seen on this
    machine), since the regex only anchors on the leading channel name and
    frame digits, not on what (if anything) follows them."""
    pattern = re.compile(r"^(" + "|".join(channel_prefixes) + r")_(\d{4}).*\.(png|exr)$")
    moved = 0
    for fname in os.listdir(tmp_dir):
        m = pattern.match(fname)
        if not m:
            continue
        channel, frame_str, ext = m.group(1), m.group(2), m.group(3)
        cam_dir = os.path.join(dest_scene_dir, str(int(frame_str)))
        os.makedirs(cam_dir, exist_ok=True)
        shutil.move(os.path.join(tmp_dir, fname), os.path.join(cam_dir, f"{channel}.{ext}"))
        moved += 1
    return moved


num_moved = move_gbuffers_into_view_folders(
    gbuffer_tmp_dir, scene_dir, ["albedo", "normals", "roughness", "metallic"]
)
print(f"Moved {num_moved} G-buffer file(s) into per-view folders under {scene_dir}")
try:
    os.rmdir(gbuffer_tmp_dir)
except OSError:
    print(f"Note: {gbuffer_tmp_dir} wasn't empty after moving - check it for unexpectedly named files.")

# Disconnect the gbuffer output nodes now that they've done their one-time
# job - otherwise they'd keep firing (and recreating gbuffer_tmp_dir) during
# every one of the M lighting-condition renders below, which is both wasted
# I/O and would leave redundant leftover files since nothing moves them
# again after this point.
_gbuffer_compositor_tree = bpy.context.scene.node_tree
for _node in (_roughness_out, _metallic_out, _albedo_out, _normals_out):
    _gbuffer_compositor_tree.nodes.remove(_node)


# ---------------------------------------------------------------------------
# Random lighting conditions (M total). Each condition is 1-3 point/area
# lights with randomized position, color, energy, and size:
#   - POINT lights: "size" is the soft-shadow radius (0 = sharp point source,
#     larger = softer/dimmer, see set_radius()).
#   - AREA lights: "size" is the physical width/height of the emitting plane
#     (set directly on the underlying Blender light data - BlenderProc's
#     Light wrapper has no dedicated method for this, only for the point-
#     light-style radius).
# Colors are sampled in HSV with capped saturation, to keep them as plausible
# (if varied) light colors rather than fully random/garish RGB combinations.
# ---------------------------------------------------------------------------
bbox = np.array([o.get_bound_box() for o in mesh_objects]).reshape(-1, 3)
room_min = bbox.min(axis=0)
room_max = bbox.max(axis=0)
ceiling_z = bbox[:, 2].max()


def sample_random_light_setup():
    num_lights = np.random.randint(1, 4)  # 1-3 lights per condition
    specs = []
    for _ in range(num_lights):
        light_type = random.choice(["POINT", "AREA"])
        location = [
            float(np.random.uniform(room_min[0], room_max[0])),
            float(np.random.uniform(room_min[1], room_max[1])),
            float(np.random.uniform(ceiling_z * 0.5, ceiling_z - 0.05)),
        ]
        hue = float(np.random.uniform(0, 1))
        saturation = float(np.random.uniform(0.0, 0.5))
        color = list(colorsys.hsv_to_rgb(hue, saturation, 1.0))
        energy = float(np.random.uniform(100, 1000))

        spec = dict(type=light_type, location=location, energy=energy, color=color)
        if light_type == "AREA":
            spec["size"] = float(np.random.uniform(0.2, 1.5))
        else:
            spec["size"] = float(np.random.uniform(0.02, 0.4))
        specs.append(spec)
    return specs


def create_lights_from_spec(light_specs):
    created = []
    for spec in light_specs:
        light = bproc.types.Light()
        light.set_type(spec["type"])
        light.set_location(spec["location"])
        light.set_energy(spec["energy"])
        light.set_color(spec["color"])
        if spec["type"] == "AREA":
            light.blender_obj.data.size = spec["size"]
        else:
            light.set_radius(spec["size"])
        created.append(light)
    return created


light_setups = [sample_random_light_setup() for _ in range(args.num_light_setups)]

manifest_path = os.path.join(scene_dir, "light_conditions.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(
        {f"light_{idx:03d}": specs for idx, specs in enumerate(light_setups)},
        f, indent=2
    )
print(f"Generated {len(light_setups)} light condition(s); manifest written to {manifest_path}")


# ---------------------------------------------------------------------------
# Render: M lighting conditions, each across all N views, written directly
# into <scene_id>/<cam_nr>/light_XXX.png. Only "colors" is reloaded via
# BlenderProc's own mechanism (which works fine).
# ---------------------------------------------------------------------------
def to_uint8_image(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.min() < 0:  # looks like a [-1, 1]-encoded map
        arr = (arr + 1.0) / 2.0
    return np.clip(arr, 0, 1) * 255


def save_png(arr, path):
    Image.fromarray(to_uint8_image(arr).astype(np.uint8)).save(path)


for idx, light_specs in enumerate(light_setups):
    light_name = f"light_{idx:03d}"
    created_lights = create_lights_from_spec(light_specs)

    data = bproc.renderer.render(load_keys={"colors"})

    for i, img in enumerate(data["colors"]):
        cam_dir = os.path.join(scene_dir, str(i))
        os.makedirs(cam_dir, exist_ok=True)
        save_png(img, os.path.join(cam_dir, f"{light_name}.png"))

    for light in created_lights:
        light.delete()

    print(f"Rendered {light_name} ({len(light_specs)} light(s)) across {poses} view(s)")

print("\nDone.")
print(f"{len(light_setups)} light condition(s) x {poses} view(s) = "
      f"{len(light_setups) * poses} total renderings")
print(f"Output layout: {scene_dir}/<cam_nr>/{{albedo,normals,roughness,metallic}}.png "
      f"+ light_000.png ... light_{len(light_setups) - 1:03d}.png")
print("Light condition manifest:", manifest_path)