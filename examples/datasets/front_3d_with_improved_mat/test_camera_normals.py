import blenderproc as bproc

"""
test_camera_normals.py
======================
Validates the world→camera-space normal transform by rendering a sphere
with a camera that is clearly rotated away from the world axes.

Expected output
---------------
  normals_world.png   – world-space: center will NOT be blue (the camera
                        faces a diagonal world direction, so the camera-
                        facing surface normal is not (0,0,1) in world space)
  normals_camera.png  – camera-space: center MUST be blue (~128,128,255),
                        regardless of camera orientation

Run with:
    blenderproc run test_camera_normals.py [output_dir]
"""

import argparse
import os
import re
import shutil


import bpy
import numpy as np
from PIL import Image

from blenderproc.python.utility.Utility import Utility

parser = argparse.ArgumentParser()
parser.add_argument("output_dir", nargs="?", default="output_normals_test")
args = parser.parse_args()
output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

bproc.init()
bproc.camera.set_resolution(512, 512)
bproc.renderer.set_max_amount_of_samples(1)   # normal pass is noise-free

# ── Scene: UV sphere ──────────────────────────────────────────────────────────
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0),
                                     segments=128, ring_count=64)
bpy.ops.object.shade_smooth()

# Minimal sun so Cycles doesn't warn about missing lights (normal pass ignores it)
sun = bpy.data.lights.new("sun", "SUN")
sun_obj = bpy.data.objects.new("sun", sun)
bpy.context.scene.collection.objects.link(sun_obj)

# ── Camera: intentionally off world-axes ──────────────────────────────────────
# (2, -2, 1.5) looking at the origin — clearly diagonal in every world axis.
# If the transform is correct the sphere's front-facing center must be blue.
cam_pos  = np.array([2.0, -2.0, 1.5])
target   = np.array([0.0,  0.0, 0.0])
world_up = np.array([0.0,  0.0, 1.0])

fwd   = target - cam_pos;  fwd   /= np.linalg.norm(fwd)
right = np.cross(fwd, world_up);  right /= np.linalg.norm(right)
up    = np.cross(right, fwd);     up    /= np.linalg.norm(up)

# cam2world: columns are camera X/Y/Z axes expressed in world coordinates.
# Camera local +Z = −fwd  (camera looks in its own −Z direction).
R = np.stack([right, up, -fwd], axis=1)   # (3, 3)
cam2world = np.eye(4, dtype=np.float32)
cam2world[:3, :3] = R
cam2world[:3,  3] = cam_pos

bproc.camera.add_camera_pose(cam2world)

# ── Compositor: capture world-space normals (n*0.5+0.5 → [0,1]) ──────────────
bpy.context.view_layer.use_pass_normal = True
bpy.context.scene.render.use_compositing = True
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree

rl = Utility.get_the_one_node_with_type(tree.nodes, "CompositorNodeRLayers")

mul = tree.nodes.new("CompositorNodeMixRGB")
mul.blend_type = "MULTIPLY"
mul.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
tree.links.new(rl.outputs["Normal"], mul.inputs[1])

add = tree.nodes.new("CompositorNodeMixRGB")
add.blend_type = "ADD"
add.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
tree.links.new(mul.outputs["Image"], add.inputs[1])

out = tree.nodes.new("CompositorNodeOutputFile")
out.base_path = output_dir
out.format.file_format = "PNG"
out.format.color_depth = "16"
slot = out.file_slots.values()[0]
slot.path = "normals_world_"
slot.save_as_render = False   # write linear values — skip sRGB display transform
tree.links.new(add.outputs["Image"], out.inputs[0])

bproc.renderer.render(return_data=False)

# Rename frame-suffixed file Blender writes (e.g. normals_world_0000.png)
for fname in os.listdir(output_dir):
    m = re.match(r"^(normals_world)_(\d{4}).*\.(png|exr)$", fname)
    if m:
        shutil.move(os.path.join(output_dir, fname),
                    os.path.join(output_dir, "normals_world.png"))

# ── Python transform: world → camera space ────────────────────────────────────
# cam2world[:3,:3] rotates camera→world, so for row-vector normals:
#   n_cam = n_world @ R_cam2world
# This is exactly what transform_normals_to_camera_space() does in the main script.
world_path = os.path.join(output_dir, "normals_world.png")
img = np.array(Image.open(world_path)).astype(np.float32)
max_val = 65535.0 if img.max() > 255.5 else 255.0
n_world = img / max_val * 2.0 - 1.0                     # (H, W, 3) in [-1, 1]

H, W = n_world.shape[:2]
n_cam = n_world.reshape(-1, 3) @ cam2world[:3, :3]      # world→camera (row vecs)
n_cam /= np.linalg.norm(n_cam, axis=1, keepdims=True).clip(min=1e-8)
n_cam = n_cam.reshape(H, W, 3)

encoded = ((n_cam + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
Image.fromarray(encoded).save(os.path.join(output_dir, "normals_camera.png"))

# ── Sanity check ──────────────────────────────────────────────────────────────
cy, cx = H // 2, W // 2
c_world = n_world[cy, cx]
c_cam   = n_cam[cy, cx]

print("\n── Normal map validation ─────────────────────────────────────────────")
print(f"Center pixel world-space  normal: X={c_world[0]:+.3f}  Y={c_world[1]:+.3f}  Z={c_world[2]:+.3f}")
print(f"Center pixel camera-space normal: X={c_cam[0]:+.3f}  Y={c_cam[1]:+.3f}  Z={c_cam[2]:+.3f}")
print(f"\nZ (camera-space) should be ≈ +1.0  →  actual: {c_cam[2]:.4f}")
print("PASS ✓" if c_cam[2] > 0.99 else "FAIL ✗  (check cam2world construction)")
print(f"\nOutputs written to: {output_dir}/")
print("  normals_world.png   – world-space reference (center NOT blue)")
print("  normals_camera.png  – camera-space result   (center MUST be blue)")
