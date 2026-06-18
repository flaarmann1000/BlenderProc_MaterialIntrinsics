import blenderproc as bproc

"""
render_front3d_multipass.py
============================

Renders a 3D-FRONT scene with BlenderProc: N camera views, each with its own
lighting-independent G-buffer plus M lighting conditions. For each view,
every light condition is generated, rendered, and checked: if the resulting
image's average brightness falls outside [brightness_min, brightness_max]
(default 0.2-0.8), a new random light condition is generated and re-rendered,
up to --max_light_retries attempts (after which the last attempt is kept
anyway, so every view always ends up with exactly M images). This is done
render-wise rather than pregenerating M lighting conditions for the whole
scene up front, since the same light setup can be well-exposed from one
viewpoint and not another. Output layout:

  <output_dir>/<scene_id>/
    light_conditions.json
      Manifest of every accepted (or given-up-on) light condition per view:
      its exact parameters (type/location/energy/color/size per light),
      the resulting brightness, and how many attempts it took.
    <cam_nr>/                  (one folder per view, cam_nr = 0, 1, 2, ...)
      albedo.png                  - diffuse base color   (lighting-independent)
      normals.png                 - surface normals      (lighting-independent)
      roughness.png               - 16-bit material roughness (lighting-independent)
      metallic.png                - 16-bit material metallic  (lighting-independent)
      light_000.png ... light_MMM.png  - beauty render under each lighting condition

scene_id is derived from the input scene .json's filename (without extension).

Note: this renders one camera pose at a time (via bproc.utility.
reset_keyframes() + a single add_camera_pose() per view) rather than
batching all N views into one render() call per lighting condition, since
each light condition now needs to be individually checked and possibly
retried before being accepted. This is slower per image than batched
rendering, especially if many lighting attempts get rejected.

A view's G-buffer is also checked after rendering it: if the roughness or
metallic map turns out essentially flat across the whole frame (std below
--min_material_std, e.g. a close-up of a single material), the camera pose
is discarded and a fresh one sampled instead, up to --max_view_retries
attempts (after which the last attempt is kept anyway).

NOTE on file naming: albedo/normals/roughness/metallic are all written
DIRECTLY to disk by manual compositor "File Output" nodes, bypassing
BlenderProc's own reload-by-registered-path mechanism (enable_diffuse_
color_output / enable_normals_output). This sidesteps a recurring Blender
issue on this machine where Blender appends an unexpected "_L" view suffix
to compositor-written filenames, which otherwise makes BlenderProc crash
with FileNotFoundError when it tries to reload them by their expected name.
The rename-in-place step is tolerant of that suffix (or any other
unexpected text Blender appends) since it only matches on the leading
channel name and frame digits.

Usage:
  blenderproc run render_front3d_multipass.py \
      [FRONT_JSON] [FUTURE_MODEL_DIR] [FRONT_TEXTURE_DIR] [OUTPUT_DIR] \
      [--cc_material_path CC_TEXTURES_DIR] [--metal_material_list PATH] \
      [--num_camera_poses N] [--num_light_setups M] [--seed N] [--resolution N] \
      [--brightness_min F] [--brightness_max F] [--max_light_retries N] \
      [--min_material_std F] [--max_view_retries N]

  All positional/optional arguments default to Felix's actual dataset paths
  if omitted - pass your own values to override any of them.

blenderproc run render_front3d_multipass.py --num_camera_poses 4 --num_light_setups 16 --resolution 512

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
parser.add_argument("--brightness_min", type=float, default=0.2,
                     help="Minimum acceptable average image brightness (0-1 luminance)")
parser.add_argument("--brightness_max", type=float, default=0.8,
                     help="Maximum acceptable average image brightness (0-1 luminance)")
parser.add_argument("--max_light_retries", type=int, default=15,
                     help="Max attempts to find a light condition within the brightness range "
                          "for a given (view, light slot) before giving up and keeping the "
                          "last attempt anyway")
parser.add_argument("--min_material_std", type=float, default=0.01,
                     help="Minimum standard deviation (0-1 normalized) required in BOTH the "
                          "roughness and metallic G-buffer maps for a view to be kept - a value "
                          "below this on either map means that map is essentially flat/uniform "
                          "('boring') across the whole view, e.g. a close-up of a single material.")
parser.add_argument("--max_view_retries", type=int, default=10,
                     help="Max attempts to find a camera pose whose G-buffer isn't 'boring' "
                          "(see --min_material_std) before giving up and keeping the last "
                          "attempt anyway")
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
# Camera pose sampler (same coverage/obstacle logic as BlenderProc's
# official front_3d example). Returns one candidate pose at a time rather
# than collecting all N up front, since a pose can now be rejected *after*
# rendering its G-buffer (if the material maps turn out too uniform/
# "boring" - see the per-view loop below) and a fresh pose sampled instead.
# ---------------------------------------------------------------------------
bproc.camera.set_resolution(args.resolution, args.resolution)

point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)
bvh_tree = bproc.object.create_bvh_tree_multi_objects(mesh_objects)

proximity_checks = {"min": 1.0, "avg": {"min": 2.5, "max": 3.5}, "no_background": True}


def sample_camera_pose(max_tries=10000):
    for _ in range(max_tries):
        height = np.random.uniform(1.4, 1.8)
        location = point_sampler.sample(height)
        rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, 2 * np.pi])
        candidate = bproc.math.build_transformation_mat(location, rotation)
        if bproc.camera.scene_coverage_score(candidate) > 0.4 \
                and bproc.camera.perform_obstacle_in_view_check(candidate, proximity_checks, bvh_tree):
            return candidate
    raise RuntimeError(f"Could not find a valid camera pose after {max_tries} tries.")



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
                src = socket.links[0].from_socket
                links.new(src, aov_node.inputs["Value"])
            else:
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


def rename_gbuffer_files(view_dir, channel_prefixes):
    """Renames channel_FRAME[_suffix].ext into a clean channel.ext name
    (frame is always 0000 here, since keyframes get reset per view).
    Tolerant of any unexpected suffix Blender may add (e.g. the "_L"
    view-suffix quirk seen on this machine), since the regex only anchors
    on the leading channel name and frame digits."""
    pattern = re.compile(r"^(" + "|".join(channel_prefixes) + r")_(\d{4}).*\.(png|exr)$")
    for fname in os.listdir(view_dir):
        m = pattern.match(fname)
        if not m:
            continue
        channel, _frame, ext = m.group(1), m.group(2), m.group(3)
        src = os.path.join(view_dir, fname)
        dest = os.path.join(view_dir, f"{channel}.{ext}")
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.move(src, dest)


def capture_gbuffers_for_view(view_dir):
    """Creates the gbuffer compositor output nodes fresh, renders this
    view's albedo/normals/roughness/metallic, renames the files, then
    removes the nodes again - so they don't keep firing (and re-cluttering
    this folder) during the light-retry renders that follow."""
    nodes = [
        enable_aov_file_output(view_dir, "Roughness", "roughness_"),
        enable_aov_file_output(view_dir, "Metallic", "metallic_"),
        enable_diffuse_output_simple(view_dir, "albedo_"),
        enable_normals_output_simple(view_dir, "normals_"),
    ]
    bproc.renderer.render(return_data=False)
    rename_gbuffer_files(view_dir, ["albedo", "normals", "roughness", "metallic"])

    tree = bpy.context.scene.node_tree
    for node in nodes:
        tree.nodes.remove(node)


# ---------------------------------------------------------------------------
# Random lighting condition generator. Each condition is 1-3 point/area
# lights with randomized position, color, energy, and size:
#   - POINT lights: "size" is the soft-shadow radius (0 = sharp point
#     source, larger = softer/dimmer, see set_radius()).
#   - AREA lights: "size" is the physical width/height of the emitting
#     plane (set directly on the underlying Blender light data -
#     BlenderProc's Light wrapper has no dedicated method for this, only
#     for the point-light-style radius).
# Colors are sampled in HSV with capped saturation, to keep them as
# plausible (if varied) light colors rather than fully random/garish RGB.
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


def average_brightness(img):
    """Mean perceptual luminance of an image, normalized to [0, 1]."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.5:  # detect 0-255 range vs already-0-1 range
        arr = arr / 255.0
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        luminance = arr
    return float(luminance.mean())


def channel_std(view_dir, channel):
    """Standard deviation (0-1 normalized) of a single-channel gbuffer PNG
    already written to view_dir by capture_gbuffers_for_view."""
    arr = np.array(Image.open(os.path.join(view_dir, f"{channel}.png"))).astype(np.float32)
    max_val = 65535.0 if arr.max() > 255.5 else 255.0  # 16-bit vs 8-bit PNG
    return float((arr / max_val).std())


def gbuffer_variation_ok(view_dir, min_std):
    """Rejects a view whose roughness or metallic map is essentially flat
    across the whole frame (e.g. a close-up of a single material) - such a
    view carries little useful material information."""
    roughness_std = channel_std(view_dir, "roughness")
    metallic_std = channel_std(view_dir, "metallic")
    ok = roughness_std >= min_std and metallic_std >= min_std
    return ok, roughness_std, metallic_std


# ---------------------------------------------------------------------------
# Per-view rendering: for each view, capture its G-buffer once, then fill
# all M light slots, regenerating + re-rendering any light condition whose
# resulting average brightness falls outside [brightness_min, brightness_max]
# (up to max_light_retries attempts before giving up and keeping the last
# attempt anyway, so every view always ends up with exactly M images).
#
# This replaces the earlier approach of pregenerating M lighting conditions
# once for the whole scene and applying them across all views - validity is
# now checked render-by-render instead, since the same light setup can be
# well-exposed from one viewpoint and not another.
# ---------------------------------------------------------------------------
manifest = {}

for view_idx in range(args.num_camera_poses):
    view_dir = os.path.join(scene_dir, str(view_idx))
    os.makedirs(view_dir, exist_ok=True)

    view_tries = 0
    view_ok = False
    roughness_std = metallic_std = None
    while not view_ok and view_tries < args.max_view_retries:
        view_tries += 1
        cam2world_matrix = sample_camera_pose()

        bproc.utility.reset_keyframes()
        bproc.camera.add_camera_pose(cam2world_matrix)
        capture_gbuffers_for_view(view_dir)

        view_ok, roughness_std, metallic_std = gbuffer_variation_ok(view_dir, args.min_material_std)
        if not view_ok:
            print(f"view {view_idx}: gbuffer too 'boring' (roughness_std={roughness_std:.4f}, "
                  f"metallic_std={metallic_std:.4f}, attempt {view_tries}) - resampling a new view")

    view_status = "accepted" if view_ok else f"gave up after {view_tries} attempt(s), kept last"
    print(f"view {view_idx}: roughness_std={roughness_std:.4f}, metallic_std={metallic_std:.4f} ({view_status})")

    view_manifest = {
        "_view": {
            "tries": view_tries,
            "accepted": view_ok,
            "roughness_std": roughness_std,
            "metallic_std": metallic_std,
        }
    }
    for light_idx in range(args.num_light_setups):
        light_name = f"light_{light_idx:03d}"
        accepted = False
        attempt = 0
        light_spec = None
        brightness = None

        while not accepted and attempt < args.max_light_retries:
            attempt += 1
            light_spec = sample_random_light_setup()
            created_lights = create_lights_from_spec(light_spec)

            data = bproc.renderer.render(load_keys={"colors"})
            img = data["colors"][0]
            brightness = average_brightness(img)
            accepted = args.brightness_min <= brightness <= args.brightness_max

            if accepted or attempt == args.max_light_retries:
                save_png(img, os.path.join(view_dir, f"{light_name}.png"))

            for light in created_lights:
                light.delete()

        status = "accepted" if accepted else f"gave up after {attempt} attempt(s), kept last"
        print(f"view {view_idx} {light_name}: brightness={brightness:.3f} ({status})")

        view_manifest[light_name] = {
            "light_spec": light_spec,
            "brightness": brightness,
            "attempts": attempt,
            "accepted": accepted,
        }

    manifest[str(view_idx)] = view_manifest
    print(f"View {view_idx}/{args.num_camera_poses - 1} done -> {view_dir}")

manifest_path = os.path.join(scene_dir, "light_conditions.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nDone.")
print(f"{args.num_camera_poses} view(s) x {args.num_light_setups} light condition(s) = "
      f"{args.num_camera_poses * args.num_light_setups} total renderings")
print(f"Output layout: {scene_dir}/<cam_nr>/{{albedo,normals,roughness,metallic}}.png "
      f"+ light_000.png ... light_{args.num_light_setups - 1:03d}.png")
print("Light condition manifest:", manifest_path)