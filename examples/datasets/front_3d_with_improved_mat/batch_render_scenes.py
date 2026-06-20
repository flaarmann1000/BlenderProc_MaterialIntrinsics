"""
batch_render_scenes.py
========================

Iterates over 3D-FRONT scene .json files and runs the full
render_front3d_multipass.py pipeline on each one that passes a
"well-furnished" filter, until --num_scenes scenes have been successfully
rendered (or --max_scenes_checked candidates have been examined).

Why "well-furnished" rather than "decent metallic": metallic values only
ever come from the CC0 floor material swap (--cc_material_path /
--metal_material_list in render_front3d_multipass.py), which is identical
machinery regardless of which scene you pick - it is NOT a property of the
scene itself. What does vary per scene is how much actual furniture it has,
which affects how reliably each view passes render_front3d_multipass.py's
own --min_material_std "boring view" check and how visually varied the
output ends up being. So this script filters on furniture amount (reusing
the same material-count heuristic as batch_inspect_scenes.py) as the
practical proxy, and leaves per-view metallic/roughness quality to
render_front3d_multipass.py's own retry logic, which already handles that.

This is a *regular* Python script - run it with plain `python`, not
`blenderproc run`. It launches two kinds of subprocesses per candidate
scene: a quick `blenderproc run inspect_materials.py ...` check, and (if
that scene passes) a full `blenderproc run render_front3d_multipass.py ...`
render.

Usage:
  python batch_render_scenes.py [FRONT_JSON_DIR] [FUTURE_MODEL_DIR] [FRONT_TEXTURE_DIR] [OUTPUT_DIR] \
      [--cc_material_path CC_TEXTURES_DIR] [--metal_material_list PATH] \
      [--num_scenes N] [--max_scenes_checked N] [--min_materials N] \
      [--inspect_script PATH] [--render_script PATH] \
      [-- <any extra args, forwarded as-is to render_front3d_multipass.py>]

  All positional/optional arguments default to Felix's actual dataset paths
  if omitted. Anything after a literal "--" is forwarded verbatim to every
  render_front3d_multipass.py call, e.g.:

    python batch_render_scenes.py --num_scenes 5
"""

import argparse
import glob
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("front_json_dir", nargs="?", default=r"E:\3D-Front\3D-FRONT",
                     help="Folder containing 3D-FRONT scene .json files")
parser.add_argument("future_folder", nargs="?", default=r"E:\3D-Front\3D-FUTURE-model",
                     help="Path to the 3D-FUTURE-model folder")
parser.add_argument("front_texture", nargs="?", default=r"E:\3D-Front\3D-FRONT-texture",
                     help="Path to the 3D-FRONT-texture folder")
parser.add_argument("output_dir", nargs="?", default=r"E:\3D-Front\output",
                     help="Output directory passed through to render_front3d_multipass.py "
                          "(each scene gets its own <output_dir>/<scene_id>/ subfolder there)")
parser.add_argument("--cc_material_path", default=r"E:\3D-Front\cctextures",
                     help="Path to a cctextures folder. Pass --cc_material_path none to disable.")
parser.add_argument("--metal_material_list",
                     default=r"C:\Users\felix\Documents\BlenderProc\examples\datasets\front_3d_with_improved_mat\metal_material_paths.txt",
                     help="Confirmed-metallic asset list, forwarded to both scripts.")
parser.add_argument("--num_scenes", type=int, default=5,
                     help="Number of well-furnished scenes to fully render")
parser.add_argument("--max_scenes_checked", type=int, default=50,
                     help="Max number of candidate scenes to examine before giving up, "
                          "even if --num_scenes hasn't been reached yet")
parser.add_argument("--min_materials", type=int, default=70,
                     help="Same heuristic as batch_inspect_scenes.py: a bare room shell "
                          "(walls/ceiling/floor only, no furniture) typically shows ~50 "
                          "materials, so scenes at or above this threshold are treated as "
                          "well-furnished and get fully rendered.")
parser.add_argument("--inspect_script", default="inspect_materials.py",
                     help="Path to inspect_materials.py")
parser.add_argument("--render_script", default="render_front3d_multipass.py",
                     help="Path to render_front3d_multipass.py")
parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True,
                     help="Shuffle scene order before checking (default True), so repeated "
                          "runs don't always re-examine the same alphabetically-first scenes")
parser.add_argument("--seed", type=int, default=None, help="Random seed for --shuffle")
args, extra_render_args = parser.parse_known_args()

if args.cc_material_path and args.cc_material_path.strip().lower() in ("none", ""):
    args.cc_material_path = None
if args.metal_material_list and args.metal_material_list.strip().lower() in ("none", ""):
    args.metal_material_list = None

# argparse.parse_known_args() leaves a literal "--" (if present) as the
# first leftover token - strip it so it isn't forwarded as a stray argument.
if extra_render_args and extra_render_args[0] == "--":
    extra_render_args = extra_render_args[1:]

json_files = sorted(glob.glob(os.path.join(args.front_json_dir, "*.json")))
print(f"Found {len(json_files)} scene file(s) in {args.front_json_dir}")
if not json_files:
    raise SystemExit("No .json files found - check front_json_dir.")

if args.shuffle:
    import random
    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(json_files)

rendered = []
checked = 0

for json_path in json_files:
    if len(rendered) >= args.num_scenes or checked >= args.max_scenes_checked:
        break
    checked += 1
    scene_id = os.path.splitext(os.path.basename(json_path))[0]
    print(f"\n[{checked}/{args.max_scenes_checked}] Checking {scene_id} "
          f"({len(rendered)}/{args.num_scenes} rendered so far) ...")

    inspect_cmd = ["blenderproc", "run", args.inspect_script,
                   json_path, args.future_folder, args.front_texture]
    if args.cc_material_path:
        inspect_cmd += ["--cc_material_path", args.cc_material_path]
    if args.metal_material_list:
        inspect_cmd += ["--metal_material_list", args.metal_material_list]

    proc = subprocess.run(inspect_cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr

    import re
    m_total = re.search(r"Materials with a usable Principled BSDF: (\d+)", output)
    if not m_total:
        print("  Could not parse inspect output (script may have crashed) - skipping. Last lines:")
        print("\n".join(output.splitlines()[-10:]))
        continue

    total_materials = int(m_total.group(1))
    well_furnished = total_materials >= args.min_materials
    print(f"  total materials: {total_materials} (well-furnished: {well_furnished})")

    if not well_furnished:
        continue

    print(f"  -> rendering {scene_id} ...")
    render_cmd = ["blenderproc", "run", args.render_script,
                  json_path, args.future_folder, args.front_texture, args.output_dir]
    if args.cc_material_path:
        render_cmd += ["--cc_material_path", args.cc_material_path]
    if args.metal_material_list:
        render_cmd += ["--metal_material_list", args.metal_material_list]
    render_cmd += extra_render_args

    print(f"  $ {' '.join(render_cmd)}")
    render_proc = subprocess.run(render_cmd)
    if render_proc.returncode != 0:
        print(f"  WARNING: render_front3d_multipass.py exited with code "
              f"{render_proc.returncode} for {scene_id} - check its output above.")
        continue

    rendered.append(scene_id)
    print(f"  done: {scene_id}")

print(f"\nChecked {checked} scene(s), rendered {len(rendered)}/{args.num_scenes} requested.")
if rendered:
    print("Rendered scene ids:")
    for scene_id in rendered:
        print(f"  - {scene_id}")
if len(rendered) < args.num_scenes:
    print(f"WARNING: only found {len(rendered)} well-furnished scene(s) within "
          f"--max_scenes_checked={args.max_scenes_checked}. Increase that limit, "
          f"or lower --min_materials, to get more.")
