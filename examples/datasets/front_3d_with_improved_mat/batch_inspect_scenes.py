"""
batch_inspect_scenes.py
=========================

Iterates over 3D-FRONT scene .json files, running inspect_materials.py on
each (as a separate BlenderProc subprocess), and reports which scenes
actually contain a substantial amount of furniture - i.e. more materials
than just the room shell (walls/ceiling/floor), which is what an
under-furnished or empty scene would show.

Note: this no longer searches for "non-trivial metallic" scenes. Metallic
values only come from CC0 materials applied to floors/walls (--cc_material_path),
and that assignment is random per-run, not tied to which scene's furniture
you pick - so every scene looks the same on that front once CC materials are
applied. What still varies meaningfully per scene is how much actual
furniture it contains, which is what this script now reports.

This is a *regular* Python script - run it with plain `python`, not
`blenderproc run`. It launches `blenderproc run inspect_materials.py ...`
as a subprocess for each scene it checks.

Usage:
  python batch_inspect_scenes.py [FRONT_JSON_DIR] [FUTURE_MODEL_DIR] [FRONT_TEXTURE_DIR] \
      [--cc_material_path CC_TEXTURES_DIR] \
      [--max_scenes N] [--min_materials N] [--stop_after_found N] [--inspect_script PATH]

  All positional/optional arguments default to Felix's actual dataset paths
  if omitted - pass your own values to override any of them.
"""

import argparse
import glob
import os
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("front_json_dir", nargs="?", default=r"E:\3D-Front\3D-FRONT",
                     help="Folder containing 3D-FRONT scene .json files")
parser.add_argument("future_folder", nargs="?", default=r"E:\3D-Front\3D-FUTURE-model",
                     help="Path to the 3D-FUTURE-model folder")
parser.add_argument("front_texture", nargs="?", default=r"E:\3D-Front\3D-FRONT-texture",
                     help="Path to the 3D-FRONT-texture folder")
parser.add_argument("--cc_material_path", default=r"E:\3D-Front\cctextures",
                     help="Path to a cctextures folder, passed through to inspect_materials.py. "
                          "Pass --cc_material_path none to disable.")
parser.add_argument("--max_scenes", type=int, default=10,
                     help="Max number of scenes to check before giving up")
parser.add_argument("--min_materials", type=int, default=70,
                     help="A bare room shell (walls/ceiling/floor only) typically shows ~50 "
                          "materials with no furniture at all - scenes at or above this "
                          "threshold are flagged as having real furniture content.")
parser.add_argument("--stop_after_found", type=int, default=1,
                     help="Stop once this many well-furnished scenes are found")
parser.add_argument("--inspect_script", default="inspect_materials.py",
                     help="Path to inspect_materials.py")
args = parser.parse_args()

if args.cc_material_path and args.cc_material_path.strip().lower() in ("none", ""):
    args.cc_material_path = None

json_files = sorted(glob.glob(os.path.join(args.front_json_dir, "*.json")))
print(f"Found {len(json_files)} scene files in {args.front_json_dir}")
if not json_files:
    raise SystemExit("No .json files found - check the front_json_dir path.")

found = 0
checked = 0
results = []

for json_path in json_files:
    if checked >= args.max_scenes or found >= args.stop_after_found:
        break
    checked += 1
    scene_id = os.path.splitext(os.path.basename(json_path))[0]
    print(f"\n[{checked}/{args.max_scenes}] Checking {scene_id} ...")

    cmd = ["blenderproc", "run", args.inspect_script, json_path, args.future_folder, args.front_texture]
    if args.cc_material_path:
        cmd += ["--cc_material_path", args.cc_material_path]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr

    m_total = re.search(r"Materials with a usable Principled BSDF: (\d+)", output)
    m_const = re.search(r"constant metallic values: (\d+) \((\d+) are non-zero\)", output)
    m_tex = re.search(r"texture/node-driven metallic: (\d+)", output)

    if not m_total:
        print("  Could not parse output (script may have crashed). Last lines:")
        print("\n".join(output.splitlines()[-15:]))
        continue

    total = int(m_total.group(1))
    nonzero_const = int(m_const.group(2)) if m_const else 0
    textured = int(m_tex.group(1)) if m_tex else 0
    well_furnished = total >= args.min_materials

    print(f"  total materials: {total}  |  non-zero constant metallic: {nonzero_const}  |  "
          f"texture/node-driven: {textured}  |  well-furnished: {well_furnished}")
    results.append((scene_id, total, well_furnished))

    if well_furnished:
        found += 1
        print(f"  >>> FOUND: {scene_id} has {total} materials (>= {args.min_materials}) <<<")

print(f"\nChecked {checked} scene(s), found {found} well-furnished (>= {args.min_materials} materials).")
print("\nSummary (scene_id, total materials, well-furnished):")
for scene_id, total, well_furnished in results:
    marker = "  <-- use this one" if well_furnished else ""
    print(f"  {scene_id}: {total} materials{marker}")