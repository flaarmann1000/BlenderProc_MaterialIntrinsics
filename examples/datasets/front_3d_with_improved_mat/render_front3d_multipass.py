import blenderproc as bproc

"""
render_front3d_multipass.py
============================

Renders a 3D-FRONT scene with BlenderProc: N camera views, each with its own
lighting-independent G-buffer plus exactly three lighting conditions:

  sun             – 1-3 random SUN (directional) lights, shadows disabled
  point_shadow    – 1-3 random POINT lights, shadows enabled
  point_no_shadow – same POINT light setup re-rendered with shadows disabled

The sun and point setups are sampled independently per view. The two point
renders share the same light configuration so shadow/no-shadow images are
paired pixel-accurate. Each setup is brightness-checked with a cheap preview
render and re-sampled up to --max_light_retries times if the result falls
outside [brightness_min, brightness_max]. Output layout:

  <output_dir>/<scene_id>/
    light_conditions.json
      Manifest of the accepted (or given-up-on) light condition per view:
      its exact parameters, the resulting brightness, and how many attempts.
    <cam_nr>/                  (one folder per view, cam_nr = 0, 1, 2, ...)
      albedo.png / .exr            - raw Base Color via AOV (both linear; PNG written via PIL, no sRGB)
      normals.png / .exr           - camera-space normals (PNG: uint8 encoded; EXR: raw [-1,1] float32)
      roughness.png / .exr         - material roughness   (lighting-independent)
      metallic.png / .exr          - material metallic    (lighting-independent)
      sun_<i>.png / _srgb.png / .exr           - SUN lights, no shadows        (i = 0..num_light_setups-1)
      point_shadow_<i>.png / _srgb.png / .exr  - POINT lights, shadows on      (i = 0..num_light_setups-1)
      point_no_shadow_<i>.png / _srgb.png / .exr - same POINT lights, shadows off (paired with point_shadow_<i>)
      (linear .png = raw [0-1] clipped; _srgb.png = sRGB-encoded for display; .exr = linear float32)

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

Each lighting condition's brightness check uses a cheap low-sample preview
render (--preview_samples, default 10) rather than a full-quality render,
since most candidates get rejected; only the accepted condition (or the
last attempt, if --max_light_retries is exhausted) is re-rendered once at
full quality (--samples, default 512) before being saved.

Each random lighting condition is made of 1-3 POINT and/or SUN (directional)
lights rather than POINT/AREA - SUN lights behave like real sunlight, with
direction set by rotation rather than position, and energy on a much
smaller W/m^2 scale than POINT/AREA's Watts.

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
      [--num_camera_poses N] [--seed N] [--resolution N] \
      [--brightness_min F] [--brightness_max F] [--max_light_retries N] \
      [--min_material_std F] [--max_view_retries N] \
      [--samples N] [--preview_samples N]

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
parser.add_argument("--num_light_setups", type=int, default=16,
                     help="Number of distinct lighting configurations to render per view per "
                          "lighting type (sun / point). Each setup is independently sampled "
                          "and brightness-checked. Outputs are named sun_0, sun_1, ... and "
                          "point_shadow_0, point_no_shadow_0, etc.")
parser.add_argument("--seed", type=int, default=None,
                     help="Optional random seed, for reproducible camera/light sampling")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                     help="If true (default), delete any existing <scene_id> output folder "
                          "before rendering, so old runs never mix with the new one. Pass "
                          "--no-overwrite to keep/merge into an existing folder instead.")
parser.add_argument("--resolution", type=int, default=124,
                     help="Square render resolution")
parser.add_argument("--brightness_min", type=float, default=0.3,
                     help="Minimum acceptable average image brightness (0-1 luminance)")
parser.add_argument("--brightness_max", type=float, default=0.7,
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
parser.add_argument("--samples", type=int, default=512,
                     help="Max Cycles samples per pixel for FINAL/accepted renderings "
                          "(BlenderProc's own default is 1024 - this lowers it for speed).")
parser.add_argument("--preview_samples", type=int, default=10,
                     help="Max Cycles samples per pixel used while testing whether a candidate "
                          "lighting condition falls in the target brightness range. Kept low "
                          "since these are throwaway checks - only the final accepted render "
                          "(or the last attempt, if --max_light_retries is exhausted) is "
                          "re-rendered at --samples quality.")
parser.add_argument("--accepted_setups", default=None,
                     help="Path to an accepted_setups.json produced by a previous run of this "
                          "script. When given, the scene is loaded identically but all camera "
                          "sampling / gbuffer checks / brightness checks are skipped: only the "
                          "exact camera poses and light specs recorded in the JSON are used. "
                          "scene_dir is inferred from the JSON file's parent directory, so "
                          "--output_dir is ignored. The neighbouring light_conditions.json is "
                          "left untouched.")
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
bproc.renderer.set_max_amount_of_samples(args.samples)

# Disable all display/view transforms so every output (G-buffers AND beauty
# renders) is saved in raw linear light with no tonemapping or gamma applied.
bpy.context.scene.view_settings.view_transform = "Raw"

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
    if "Albedo" not in existing:
        aov = view_layer.aovs.add()
        aov.name = "Albedo"
        aov.type = "COLOR"
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

        # Albedo AOV: wire Base Color directly (preserves texture links) so the
        # output is the raw base color without DiffCol's (1-metallic) weighting.
        color_socket = bsdf.inputs["Base Color"]
        albedo_aov = nodes.new("ShaderNodeOutputAOV")
        albedo_aov.aov_name = "Albedo"
        if color_socket.is_linked:
            links.new(color_socket.links[0].from_socket, albedo_aov.inputs["Color"])
        else:
            rgb_node = nodes.new("ShaderNodeRGB")
            rgb_node.outputs[0].default_value = color_socket.default_value
            links.new(rgb_node.outputs[0], albedo_aov.inputs["Color"])


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
    slot = out_node.file_slots.values()[0]
    slot.path = file_prefix
    slot.save_as_render = False
    tree.links.new(socket, out_node.inputs[0])

    exr_node = tree.nodes.new("CompositorNodeOutputFile")
    exr_node.base_path = output_dir
    exr_node.format.file_format = "OPEN_EXR"
    exr_node.format.color_mode = "RGB"
    exr_node.format.color_depth = "32"
    exr_slot = exr_node.file_slots.values()[0]
    exr_slot.path = file_prefix
    exr_slot.save_as_render = False
    tree.links.new(socket, exr_node.inputs[0])
    return [out_node, exr_node]


def enable_albedo_output(output_dir, file_prefix="albedo_"):
    """Writes the raw Base Color via the Albedo AOV (no metallic weighting).
    Only writes EXR here; the linear PNG is produced in
    capture_gbuffers_for_view() by loading this EXR back via bpy.data.images
    (bypasses Blender's sRGB compositor encoding for PNG)."""
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    render_layer_node = Utility.get_the_one_node_with_type(tree.nodes, "CompositorNodeRLayers")

    exr_node = tree.nodes.new("CompositorNodeOutputFile")
    exr_node.base_path = output_dir
    exr_node.format.file_format = "OPEN_EXR"
    exr_node.format.color_mode = "RGB"
    exr_node.format.color_depth = "32"
    exr_slot = exr_node.file_slots.values()[0]
    exr_slot.path = file_prefix
    exr_slot.save_as_render = False
    tree.links.new(render_layer_node.outputs["Albedo"], exr_node.inputs[0])
    return [exr_node]


def enable_normals_output_simple(output_dir, file_prefix="normals_"):
    """Writes world-space normals to disk via the built-in Normal render pass,
    remapped to [0,1] (n*0.5+0.5). The caller is responsible for transforming
    to camera space afterwards using transform_normals_to_camera_space()."""
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
    slot = out_node.file_slots.values()[0]
    slot.path = file_prefix
    slot.save_as_render = False   # write linear values — skip sRGB display transform
    tree.links.new(mix_add.outputs["Image"], out_node.inputs[0])
    return [mix_scale, mix_add, out_node]


def transform_normals_to_camera_space(view_dir, cam2world_matrix):
    """Loads the world-space normals PNG written by Blender, transforms each
    normal to camera space using the known cam2world rotation, and overwrites
    the file as an 8-bit RGB PNG (sufficient precision for normal maps).

    Math: cam2world_matrix[:3,:3] maps camera→world, so its transpose maps
    world→camera. For row-vector normals: n_cam = n_world @ R_cam2world,
    which is equivalent to n_cam = (R_cam2world.T @ n_world.T).T."""
    normals_path = os.path.join(view_dir, "normals.png")
    img = np.array(Image.open(normals_path)).astype(np.float32)
    max_val = 65535.0 if img.max() > 255.5 else 255.0
    n_world = img / max_val * 2.0 - 1.0          # (H, W, 3) in [-1, 1]

    R = cam2world_matrix[:3, :3]                  # camera→world rotation
    H, W = n_world.shape[:2]
    n_cam = n_world.reshape(-1, 3) @ R            # world→camera (row vectors)
    n_cam /= np.linalg.norm(n_cam, axis=1, keepdims=True).clip(min=1e-8)
    n_cam = n_cam.reshape(H, W, 3)

    encoded = ((n_cam + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(encoded).save(normals_path)
    save_exr(n_cam, normals_path.replace(".png", ".exr"))


setup_material_aovs()

scene_id = os.path.splitext(os.path.basename(args.front_json))[0]
if args.accepted_setups:
    # Re-render mode: scene_dir is where the JSON lives; never delete it.
    scene_dir = os.path.dirname(os.path.abspath(args.accepted_setups))
else:
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


def capture_gbuffers_for_view(view_dir, cam2world_matrix):
    """Creates the gbuffer compositor output nodes fresh, renders this
    view's albedo/normals/roughness/metallic, renames the files, then
    removes the nodes again - so they don't keep firing (and re-cluttering
    this folder) during the light-retry renders that follow.
    Normals are captured world-space and then transformed to camera space
    in Python (avoids Blender compositor clamping of negative components)."""
    node_groups = [
        enable_aov_file_output(view_dir, "Roughness", "roughness_"),
        enable_aov_file_output(view_dir, "Metallic", "metallic_"),
        enable_albedo_output(view_dir, "albedo_"),
        enable_normals_output_simple(view_dir, "normals_"),
    ]
    bproc.renderer.render(return_data=False)
    rename_gbuffer_files(view_dir, ["albedo", "normals", "roughness", "metallic"])
    transform_normals_to_camera_space(view_dir, cam2world_matrix)

    # Linear albedo PNG: load the EXR (scene-linear) and save via PIL (no sRGB).
    _alb_exr = os.path.join(view_dir, "albedo.exr")
    _alb_img = bpy.data.images.load(_alb_exr)
    _alb_img.colorspace_settings.name = "Non-Color"
    _alb_px = np.empty(_alb_img.size[0] * _alb_img.size[1] * 4, dtype=np.float32)
    _alb_img.pixels.foreach_get(_alb_px)
    _alb_arr = _alb_px.reshape(_alb_img.size[1], _alb_img.size[0], 4)[::-1, :, :3]
    save_png(_alb_arr.copy(), os.path.join(view_dir, "albedo.png"))
    bpy.data.images.remove(_alb_img)

    tree = bpy.context.scene.node_tree
    for group in node_groups:
        for node in (group if isinstance(group, list) else [group]):
            tree.nodes.remove(node)


# ---------------------------------------------------------------------------
# Light samplers — one per type.  Colors use capped-saturation HSV to stay
# plausible rather than garish.  SUN lights are directional (position is
# irrelevant in Blender; only rotation matters).  POINT lights are placed
# randomly within the room volume near the ceiling.
# ---------------------------------------------------------------------------
bbox = np.array([o.get_bound_box() for o in mesh_objects]).reshape(-1, 3)
room_min = bbox.min(axis=0)
room_max = bbox.max(axis=0)
ceiling_z = bbox[:, 2].max()


def _random_color():
    return list(colorsys.hsv_to_rgb(
        float(np.random.uniform(0, 1)),
        float(np.random.uniform(0.0, 0.5)),
        1.0,
    ))


def sample_sun_light_setup():
    return [
        dict(
            type="SUN",
            color=_random_color(),
            rotation_euler=[
                float(np.random.uniform(0, np.pi)),
                float(np.random.uniform(0, 2 * np.pi)),
                float(np.random.uniform(0, 2 * np.pi)),
            ],
            energy=float(np.random.uniform(0.5, 4.0)),
            size=float(np.random.uniform(0.01, 0.3)),
        )
        for _ in range(np.random.randint(1, 4))
    ]


def sample_point_light_setup():
    return [
        dict(
            type="POINT",
            color=_random_color(),
            location=[
                float(np.random.uniform(room_min[0], room_max[0])),
                float(np.random.uniform(room_min[1], room_max[1])),
                float(np.random.uniform(ceiling_z * 0.5, ceiling_z - 0.05)),
            ],
            energy=float(np.random.uniform(100, 1000)),
            size=float(np.random.uniform(0.02, 0.4)),
        )
        for _ in range(np.random.randint(1, 4))
    ]


def create_lights_from_spec(light_specs):
    created = []
    for spec in light_specs:
        light = bproc.types.Light()
        light.set_type(spec["type"])
        light.set_energy(spec["energy"])
        light.set_color(spec["color"])
        if spec["type"] == "SUN":
            light.blender_obj.rotation_euler = spec["rotation_euler"]
            light.blender_obj.data.angle = spec["size"]
        else:
            light.set_location(spec["location"])
            light.set_radius(spec["size"])
        created.append(light)
    return created


def set_shadow(lights, enabled):
    for light in lights:
        d = light.blender_obj.data
        d.use_shadow = enabled
        if hasattr(d, "cycles") and hasattr(d.cycles, "cast_shadow"):
            d.cycles.cast_shadow = enabled


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


def save_exr(arr, path):
    """Save a float32 numpy array as a 32-bit EXR via Blender's image API.
    Pixels are stored as linear floats; no encoding or clamping is applied.
    Uint8 [0-255] input (as returned by BlenderProc's colors key) is
    normalized to [0, 1] before writing."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() > 1.5:   # uint8 [0-255] → normalize to [0, 1]
        arr = arr / 255.0
    H, W = arr.shape[:2]
    if arr.ndim == 2:
        rgba = np.stack([arr, arr, arr, np.ones((H, W), dtype=np.float32)], axis=-1)
    elif arr.shape[2] == 3:
        rgba = np.concatenate([arr, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
    else:
        rgba = arr.copy()
    img = bpy.data.images.new(os.path.basename(path), width=W, height=H, float_buffer=True)
    img.pixels.foreach_set(rgba[::-1].reshape(-1).tolist())
    img.file_format = "OPEN_EXR"
    img.filepath_raw = path
    img.save()
    bpy.data.images.remove(img)


def save_srgb_png(arr, path):
    """Apply the sRGB transfer function to a linear image and save as PNG."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() > 1.5:   # uint8 [0-255] → normalize to [0, 1]
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    srgb = np.where(arr <= 0.0031308,
                    12.92 * arr,
                    1.055 * arr ** (1.0 / 2.4) - 0.055)
    Image.fromarray((srgb * 255.0).astype(np.uint8)).save(path)


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
# Per-view rendering loop.
# ---------------------------------------------------------------------------

def render_light_condition(sampler, use_shadow_for_preview):
    """Sample a light setup, brightness-check with a preview render, then
    render at full quality.  Returns (light_spec, brightness, attempts,
    accepted, created_lights) with the lights still live in the scene so
    the caller can flip shadows and render again before deleting them."""
    accepted = False
    attempt = 0
    light_spec = None
    brightness = None
    created_lights = []

    while not accepted and attempt < args.max_light_retries:
        for l in created_lights:
            l.delete()
        attempt += 1
        light_spec = sampler()
        created_lights = create_lights_from_spec(light_spec)
        set_shadow(created_lights, use_shadow_for_preview)

        bproc.renderer.set_max_amount_of_samples(args.preview_samples)
        preview = bproc.renderer.render(load_keys={"colors"})
        brightness = average_brightness(preview["colors"][0])
        accepted = args.brightness_min <= brightness <= args.brightness_max

    return light_spec, brightness, attempt, accepted, created_lights


if args.accepted_setups:
    # ── Re-render from accepted_setups.json ──────────────────────────────────
    with open(args.accepted_setups, encoding="utf-8") as _f:
        _setups = json.load(_f)
    print(f"Re-render mode: {len(_setups)} view(s) from {args.accepted_setups}")

    for _view_key, _setup in _setups.items():
        view_idx = int(_view_key)
        view_dir = os.path.join(scene_dir, str(view_idx))
        os.makedirs(view_dir, exist_ok=True)

        cam2world_matrix = np.array(_setup["cam2world"], dtype=np.float32)
        bproc.utility.reset_keyframes()
        bproc.camera.add_camera_pose(cam2world_matrix)
        capture_gbuffers_for_view(view_dir, cam2world_matrix)

        for light_idx, sun_entry in enumerate(_setup["sun"]):
            sun_lights = create_lights_from_spec(sun_entry["light_spec"])
            set_shadow(sun_lights, False)
            bproc.renderer.set_max_amount_of_samples(args.samples)
            sun_data = bproc.renderer.render(load_keys={"colors"})
            save_png(sun_data["colors"][0], os.path.join(view_dir, f"sun_{light_idx}.png"))
            save_srgb_png(sun_data["colors"][0], os.path.join(view_dir, f"sun_{light_idx}_srgb.png"))
            save_exr(sun_data["colors"][0], os.path.join(view_dir, f"sun_{light_idx}.exr"))
            for l in sun_lights:
                l.delete()
            print(f"view {view_idx} sun[{light_idx}]: re-rendered "
                  f"(stored brightness={sun_entry['brightness']:.3f})")

        for light_idx, pt_entry in enumerate(_setup["point"]):
            pt_lights = create_lights_from_spec(pt_entry["light_spec"])
            bproc.renderer.set_max_amount_of_samples(args.samples)

            set_shadow(pt_lights, True)
            pt_shadow_data = bproc.renderer.render(load_keys={"colors"})
            save_png(pt_shadow_data["colors"][0], os.path.join(view_dir, f"point_shadow_{light_idx}.png"))
            save_srgb_png(pt_shadow_data["colors"][0], os.path.join(view_dir, f"point_shadow_{light_idx}_srgb.png"))
            save_exr(pt_shadow_data["colors"][0], os.path.join(view_dir, f"point_shadow_{light_idx}.exr"))

            set_shadow(pt_lights, False)
            pt_noshadow_data = bproc.renderer.render(load_keys={"colors"})
            save_png(pt_noshadow_data["colors"][0], os.path.join(view_dir, f"point_no_shadow_{light_idx}.png"))
            save_srgb_png(pt_noshadow_data["colors"][0], os.path.join(view_dir, f"point_no_shadow_{light_idx}_srgb.png"))
            save_exr(pt_noshadow_data["colors"][0], os.path.join(view_dir, f"point_no_shadow_{light_idx}.exr"))

            for l in pt_lights:
                l.delete()
            print(f"view {view_idx} point[{light_idx}]: re-rendered "
                  f"(stored brightness={pt_entry['brightness']:.3f})")

        print(f"View {view_idx} done -> {view_dir}")

    print("\nDone (re-render mode).")
    print(f"Re-rendered {len(_setups)} view(s) -> {scene_dir}")

else:
    # ── Normal sampling mode ──────────────────────────────────────────────
    manifest: dict = {}
    accepted_setups: dict = {}

    for view_idx in range(args.num_camera_poses):
        view_dir = os.path.join(scene_dir, str(view_idx))
        os.makedirs(view_dir, exist_ok=True)

        # ── Camera pose (retry if G-buffer is too uniform) ────────────────
        cam2world_matrix = None
        view_tries = 0
        view_ok = False
        roughness_std = metallic_std = None
        while not view_ok and view_tries < args.max_view_retries:
            view_tries += 1
            cam2world_matrix = sample_camera_pose()
            bproc.utility.reset_keyframes()
            bproc.camera.add_camera_pose(cam2world_matrix)
            capture_gbuffers_for_view(view_dir, cam2world_matrix)
            view_ok, roughness_std, metallic_std = gbuffer_variation_ok(view_dir, args.min_material_std)
            if not view_ok:
                print(f"view {view_idx}: gbuffer too 'boring' (roughness_std={roughness_std:.4f}, "
                      f"metallic_std={metallic_std:.4f}, attempt {view_tries}) - resampling")

        view_status = "accepted" if view_ok else f"gave up after {view_tries} attempt(s), kept last"
        print(f"view {view_idx}: roughness_std={roughness_std:.4f}, metallic_std={metallic_std:.4f} ({view_status})")

        view_manifest: dict = {
            "_view": {
                "tries": view_tries,
                "accepted": view_ok,
                "roughness_std": roughness_std,
                "metallic_std": metallic_std,
            }
        }

        # ── SUN lights, no shadows ────────────────────────────────────────
        sun_results = []
        for light_idx in range(args.num_light_setups):
            sun_spec, sun_brightness, sun_attempts, sun_accepted, sun_lights = \
                render_light_condition(sample_sun_light_setup, use_shadow_for_preview=False)
            set_shadow(sun_lights, False)
            bproc.renderer.set_max_amount_of_samples(args.samples)
            sun_data = bproc.renderer.render(load_keys={"colors"})
            save_png(sun_data["colors"][0], os.path.join(view_dir, f"sun_{light_idx}.png"))
            save_srgb_png(sun_data["colors"][0], os.path.join(view_dir, f"sun_{light_idx}_srgb.png"))
            save_exr(sun_data["colors"][0], os.path.join(view_dir, f"sun_{light_idx}.exr"))
            for l in sun_lights:
                l.delete()
            status = "accepted" if sun_accepted else f"gave up after {sun_attempts} attempt(s)"
            print(f"view {view_idx} sun[{light_idx}]: brightness={sun_brightness:.3f} ({status})")
            sun_results.append({
                "light_spec": sun_spec, "brightness": sun_brightness,
                "attempts": sun_attempts, "accepted": sun_accepted,
            })

        view_manifest["sun"] = sun_results

        # ── POINT lights — paired shadow / no-shadow ──────────────────────
        pt_results = []
        for light_idx in range(args.num_light_setups):
            pt_spec, pt_brightness, pt_attempts, pt_accepted, pt_lights = \
                render_light_condition(sample_point_light_setup, use_shadow_for_preview=True)
            bproc.renderer.set_max_amount_of_samples(args.samples)

            set_shadow(pt_lights, True)
            pt_shadow_data = bproc.renderer.render(load_keys={"colors"})
            save_png(pt_shadow_data["colors"][0], os.path.join(view_dir, f"point_shadow_{light_idx}.png"))
            save_srgb_png(pt_shadow_data["colors"][0], os.path.join(view_dir, f"point_shadow_{light_idx}_srgb.png"))
            save_exr(pt_shadow_data["colors"][0], os.path.join(view_dir, f"point_shadow_{light_idx}.exr"))

            set_shadow(pt_lights, False)
            pt_noshadow_data = bproc.renderer.render(load_keys={"colors"})
            save_png(pt_noshadow_data["colors"][0], os.path.join(view_dir, f"point_no_shadow_{light_idx}.png"))
            save_srgb_png(pt_noshadow_data["colors"][0], os.path.join(view_dir, f"point_no_shadow_{light_idx}_srgb.png"))
            save_exr(pt_noshadow_data["colors"][0], os.path.join(view_dir, f"point_no_shadow_{light_idx}.exr"))

            for l in pt_lights:
                l.delete()
            status = "accepted" if pt_accepted else f"gave up after {pt_attempts} attempt(s)"
            print(f"view {view_idx} point[{light_idx}]: brightness={pt_brightness:.3f} ({status})")
            pt_results.append({
                "light_spec": pt_spec, "brightness": pt_brightness,
                "attempts": pt_attempts, "accepted": pt_accepted,
            })

        view_manifest["point"] = pt_results

        manifest[str(view_idx)] = view_manifest

        sun_accepted_list = [r for r in sun_results if r["accepted"]]
        pt_accepted_list  = [r for r in pt_results  if r["accepted"]]
        if sun_accepted_list and pt_accepted_list and cam2world_matrix is not None:
            accepted_setups[str(view_idx)] = {
                "cam2world": cam2world_matrix.tolist(),
                "sun":   [{"light_spec": r["light_spec"], "brightness": r["brightness"]}
                          for r in sun_accepted_list],
                "point": [{"light_spec": r["light_spec"], "brightness": r["brightness"]}
                          for r in pt_accepted_list],
            }

        print(f"View {view_idx}/{args.num_camera_poses - 1} done -> {view_dir}")

    manifest_path = os.path.join(scene_dir, "light_conditions.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    accepted_path = os.path.join(scene_dir, "accepted_setups.json")
    with open(accepted_path, "w", encoding="utf-8") as f:
        json.dump(accepted_setups, f, indent=2)

    print("\nDone.")
    print(f"{args.num_camera_poses} view(s), {args.num_light_setups} light setups each "
          f"-> {args.num_camera_poses * args.num_light_setups * 3} total renderings")
    print(f"Accepted setups: {len(accepted_setups)}/{args.num_camera_poses} views "
          f"-> {accepted_path}")
    print(f"Output: {scene_dir}/<cam_nr>/{{albedo,normals,roughness,metallic,"
          f"sun,point_shadow,point_no_shadow}}_*.{{png,exr}}")
    print("Light condition manifest:", manifest_path)