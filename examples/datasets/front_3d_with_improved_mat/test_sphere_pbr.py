import blenderproc as bproc

"""
test_sphere_pbr.py
==================
Renders a UV sphere with known PBR material values under a single white SUN
light placed at 45° azimuth to the right of the camera axis and 45° elevation.

Material: albedo=(0.7, 0.3, 0.3)  metallic=0.5  roughness=0.7

Outputs:
  albedo.png / .exr         - diffuse color        (both linear; PNG written via PIL, no sRGB)
  roughness.png / .exr      - roughness map         (linear, no color transform)
  metallic.png / .exr       - metallic map          (linear, no color transform)
  normals.png / .exr        - camera-space normals  (PNG: uint8 [0,255]; EXR: raw [-1,1])
  rendering_linear.png      - beauty render, linear [0-1] clipped
  rendering_srgb.png        - beauty render, sRGB-encoded for display
  rendering.exr             - beauty render, 32-bit linear float

Run with:
  blenderproc run test_sphere_pbr.py [output_dir]
"""

import math
import os
import re
import shutil

import bpy
import numpy as np
from PIL import Image

from blenderproc.python.utility.Utility import Utility

# ── Args ──────────────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output_dir", nargs="?", default="output_sphere_pbr")
args = parser.parse_args()
output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# ── Init ──────────────────────────────────────────────────────────────────────
bproc.init()
bproc.camera.set_resolution(512, 512)
bproc.renderer.set_max_amount_of_samples(256)
# No tonemapping / display transform on any output
bpy.context.scene.view_settings.view_transform = "Raw"
bpy.context.scene.render.use_multiview = False

# ── Material constants ────────────────────────────────────────────────────────
ALBEDO    = (0.7, 0.3, 0.3)
METALLIC  = 0
ROUGHNESS = 1

# ── Sphere ────────────────────────────────────────────────────────────────────
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0),
                                     segments=128, ring_count=64)
bpy.ops.object.shade_smooth()
sphere_obj = bpy.context.active_object

mat = bpy.data.materials.new("sphere_mat")
mat.use_nodes = True
nodes_mat = mat.node_tree.nodes
links_mat = mat.node_tree.links
bsdf = nodes_mat["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (*ALBEDO, 1.0)
bsdf.inputs["Metallic"].default_value   = METALLIC
bsdf.inputs["Roughness"].default_value  = ROUGHNESS
sphere_obj.data.materials.append(mat)

# ── Material AOVs (albedo + roughness + metallic) ────────────────────────────
view_layer = bpy.context.view_layer

# Albedo AOV: raw Base Color, bypassing the Principled BSDF so the (1-metallic)
# weighting that Cycles applies to the DiffCol pass is not included.
aov_albedo = view_layer.aovs.add()
aov_albedo.name = "Albedo"
aov_albedo.type = "COLOR"

for aov_name in ("Roughness", "Metallic"):
    aov = view_layer.aovs.add()
    aov.name = aov_name
    aov.type = "VALUE"

# Albedo: ShaderNodeRGB outputs the constant in scene-linear space so the EXR
# contains the raw (R, G, B) tuple set in Python without any colour management.
rgb_node = nodes_mat.new("ShaderNodeRGB")
rgb_node.outputs[0].default_value = (*ALBEDO, 1.0)
albedo_aov_node = nodes_mat.new("ShaderNodeOutputAOV")
albedo_aov_node.aov_name = "Albedo"
links_mat.new(rgb_node.outputs[0], albedo_aov_node.inputs["Color"])

for input_name in ("Roughness", "Metallic"):
    val_node = nodes_mat.new("ShaderNodeValue")
    val_node.outputs[0].default_value = bsdf.inputs[input_name].default_value
    aov_node = nodes_mat.new("ShaderNodeOutputAOV")
    aov_node.aov_name = input_name
    links_mat.new(val_node.outputs[0], aov_node.inputs["Value"])

# ── Camera ────────────────────────────────────────────────────────────────────
# Camera at (0, -3, 0) looking at origin — forward = +Y world, right = +X world
cam_pos  = np.array([0.0, -3.0, 0.0])
target   = np.array([0.0,  0.0, 0.0])
world_up = np.array([0.0,  0.0, 1.0])
fwd   = target - cam_pos;  fwd   /= np.linalg.norm(fwd)
right = np.cross(fwd, world_up);  right /= np.linalg.norm(right)
up    = np.cross(right, fwd);     up    /= np.linalg.norm(up)
R = np.stack([right, up, -fwd], axis=1)
cam2world = np.eye(4, dtype=np.float32)
cam2world[:3, :3] = R
cam2world[:3,  3] = cam_pos
bproc.camera.add_camera_pose(cam2world)

# ── SUN light ────────────────────────────────────────────────────────────────
# Azimuth 45° to the right of camera axis (+Y) → horizontal direction (0.707, 0.707, 0).
# Elevation 45° above horizontal.
# Combined sun_from = (0.5, 0.5, 0.707), light travels toward (-0.5, -0.5, -0.707).
#
# Blender SUN shines along lamp local -Z. We need lamp -Z = (-0.5, -0.5, -0.707).
# With Blender XYZ Euler (rx, ry, rz):  lamp -Z = (-sin(rz)*sin(rx),  cos(rz)*sin(rx), -cos(rx))
# Solving: rx = 45° (π/4),  ry = 0,  rz = 135° (3π/4).
sun_data = bpy.data.lights.new("sun", "SUN")
sun_data.color   = (1.0, 1.0, 1.0)
sun_data.energy  = 5.0
sun_obj = bpy.data.objects.new("sun", sun_data)
bpy.context.scene.collection.objects.link(sun_obj)
# sun_obj.rotation_euler = (math.pi / 4, 0.0, 3 * math.pi / 4)
sun_obj.rotation_euler = (math.pi/4,0.0,math.pi / 4)

# ── Helper: save functions ────────────────────────────────────────────────────
def _to_linear_float(arr):
    """Normalize to [0,1] float32. Handles uint8 and float [0-255] inputs."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def save_png_linear(arr, path):
    arr = _to_linear_float(arr)
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def save_srgb_png(arr, path):
    arr = np.clip(_to_linear_float(arr), 0.0, 1.0)
    srgb = np.where(arr <= 0.0031308,
                    12.92 * arr,
                    1.055 * arr ** (1.0 / 2.4) - 0.055)
    Image.fromarray((srgb * 255.0).astype(np.uint8)).save(path)


def save_exr(arr, path):
    """Write raw float32 EXR via Blender's image API (Y-flipped, RGBA)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() > 1.5:          # uint8 [0-255] → normalize to [0, 1]
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
    img.file_format  = "OPEN_EXR"
    img.filepath_raw = path
    img.save()
    bpy.data.images.remove(img)

# ── Compositor: G-buffer file output nodes ────────────────────────────────────
bpy.context.view_layer.use_pass_normal        = True
bpy.context.scene.render.use_compositing      = True
bpy.context.scene.use_nodes                   = True
tree = bpy.context.scene.node_tree
rl = Utility.get_the_one_node_with_type(tree.nodes, "CompositorNodeRLayers")


def _add_file_out(socket, prefix, fmt, color_mode, color_depth):
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.base_path              = output_dir
    node.format.file_format     = fmt
    node.format.color_mode      = color_mode
    node.format.color_depth     = color_depth
    slot                        = node.file_slots.values()[0]
    slot.path                   = prefix
    slot.save_as_render         = False
    tree.links.new(socket, node.inputs[0])
    return node


# Albedo — raw Base Color via AOV (scene-linear, no metallic weighting).
# Only EXR here; the linear PNG is written in the post-render section by
# loading this EXR back via bpy.data.images (bypasses Blender's sRGB encoding).
_add_file_out(rl.outputs["Albedo"],   "albedo_",    "OPEN_EXR", "RGB", "32")

# Roughness
_add_file_out(rl.outputs["Roughness"], "roughness_", "PNG",     "BW",  "16")
_add_file_out(rl.outputs["Roughness"], "roughness_", "OPEN_EXR","RGB", "32")

# Metallic
_add_file_out(rl.outputs["Metallic"],  "metallic_",  "PNG",     "BW",  "16")
_add_file_out(rl.outputs["Metallic"],  "metallic_",  "OPEN_EXR","RGB", "32")

# Normals — world-space encoded (n*0.5+0.5), transformed to camera space in Python
mix_mul = tree.nodes.new("CompositorNodeMixRGB")
mix_mul.blend_type = "MULTIPLY"
mix_mul.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
tree.links.new(rl.outputs["Normal"], mix_mul.inputs[1])
mix_add = tree.nodes.new("CompositorNodeMixRGB")
mix_add.blend_type = "ADD"
mix_add.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
tree.links.new(mix_mul.outputs["Image"], mix_add.inputs[1])
_add_file_out(mix_add.outputs["Image"], "normals_world_", "PNG", "RGB", "16")

# ── Render ────────────────────────────────────────────────────────────────────
render_data = bproc.renderer.render(load_keys={"colors"})

# Rename Blender's frame-suffixed outputs (e.g. albedo_0000.png → albedo.png)
pattern = re.compile(
    r"^(albedo|roughness|metallic|normals_world)_(\d{4}).*\.(png|exr)$")
for fname in os.listdir(output_dir):
    m = pattern.match(fname)
    if not m:
        continue
    channel, _, ext = m.group(1), m.group(2), m.group(3)
    src  = os.path.join(output_dir, fname)
    dest = os.path.join(output_dir, f"{channel}.{ext}")
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.move(src, dest)

# ── World → camera-space normals ──────────────────────────────────────────────
normals_world_path = os.path.join(output_dir, "normals_world.png")
img = np.array(Image.open(normals_world_path)).astype(np.float32)
max_val = 65535.0 if img.max() > 255.5 else 255.0
n_world = img / max_val * 2.0 - 1.0          # (H, W, 3) in [-1, 1]
H, W = n_world.shape[:2]
n_cam = n_world.reshape(-1, 3) @ cam2world[:3, :3]
n_cam /= np.linalg.norm(n_cam, axis=1, keepdims=True).clip(min=1e-8)
n_cam = n_cam.reshape(H, W, 3)

encoded = ((n_cam + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
Image.fromarray(encoded).save(os.path.join(output_dir, "normals.png"))
save_exr(n_cam, os.path.join(output_dir, "normals.exr"))
os.remove(normals_world_path)

# ── Linear albedo PNG (bypasses Blender's sRGB compositor encoding) ───────────
# bpy.data.images.load() reads EXR pixels as raw scene-linear floats.
# save_png_linear writes them via PIL (clip+scale, no sRGB curve applied).
_alb_exr_path = os.path.join(output_dir, "albedo.exr")
_alb_img = bpy.data.images.load(_alb_exr_path)
_alb_img.colorspace_settings.name = "Non-Color"   # prevent internal re-conversion
_alb_px = np.empty(_alb_img.size[0] * _alb_img.size[1] * 4, dtype=np.float32)
_alb_img.pixels.foreach_get(_alb_px)
# Blender stores rows bottom-up; reshape + flip + drop alpha → (H, W, 3)
_alb_arr = _alb_px.reshape(_alb_img.size[1], _alb_img.size[0], 4)[::-1, :, :3]
save_png_linear(_alb_arr.copy(), os.path.join(output_dir, "albedo.png"))
bpy.data.images.remove(_alb_img)

# ── Beauty renders ────────────────────────────────────────────────────────────
colors = render_data["colors"][0]
save_png_linear(colors, os.path.join(output_dir, "rendering_linear.png"))
save_srgb_png  (colors, os.path.join(output_dir, "rendering_srgb.png"))
save_exr       (colors, os.path.join(output_dir, "rendering.exr"))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nOutputs written to: {output_dir}/")
for f in sorted(os.listdir(output_dir)):
    print(f"  {f}")
