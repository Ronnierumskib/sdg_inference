import os
import re
import json
import subprocess
import shutil
from pathlib import Path

# ==========================
# USER CONFIGURATION
# ==========================

# VM connection
VM_USER = "root"
VM_IP = "69.63.236.190"
VM_PORT = 26249

# Local paths (WSL)
WINDOWS_BASE_PATH = Path("/mnt/c/isaac-sim/_out_basic_writer")
LOCAL_JSON_PATH = Path("image2image_road.json") 

# VM paths
VM_BASE_PATH = "/workspace/cosmos-transfer2.5"
VM_ASSETS_PATH = f"{VM_BASE_PATH}/assets/image_example"
VM_OUTPUT_PATH = f"{VM_BASE_PATH}/outputs/image2image"

# Scene configuration
SCENE_START = 0
SCENE_END = 0   # inclusive

# ==========================
# HELPERS
# ==========================

def run(cmd):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def ssh(cmd):
    run(
        f"ssh -p {VM_PORT} {VM_USER}@{VM_IP} "
        f"\"{cmd}\""
    )

def scp_to_vm(local, remote):
    run(
        f"scp -P {VM_PORT} -r {local} "
        f"{VM_USER}@{VM_IP}:{remote}"
    )

def scp_from_vm(remote, local):
    run(
        f"scp -P {VM_PORT} -r "
        f"{VM_USER}@{VM_IP}:{remote} {local}"
    )

# ==========================
# MAIN LOGIC
# ==========================

def main():
    all_video_paths = []
    image_map = []  # (scene, filename)

    # 1. Collect images
    rgb_pattern = re.compile(r"^rgb_\d{4}\.png$")

    for scene in range(SCENE_START, SCENE_END + 1):
        scene_dir = WINDOWS_BASE_PATH / f"scene_{scene}"
        if not scene_dir.exists():
            print(f"[WARN] Missing {scene_dir}, skipping")
            continue

        for img in sorted(scene_dir.iterdir()):
            if img.is_file() and rgb_pattern.match(img.name):
                rel_path = f"scene_{scene}/{img.name}"
                all_video_paths.append(rel_path)
                image_map.append((scene, img.name))

    if not all_video_paths:
        raise RuntimeError("No images found!")

    # 2. Prepare JSON
    with open(LOCAL_JSON_PATH, "r") as f:
        cfg = json.load(f)

    cfg["video_path"] = all_video_paths
    cfg["num_video_frames_per_chunk"] = 1
    cfg["max_frames"] = 1
    temp_json = Path("image2image_road_temp.json")
    with open(temp_json, "w") as f:
        json.dump(cfg, f, indent=4)

    # 3. Prepare VM directories
    ssh(f"mkdir -p {VM_ASSETS_PATH}")
    ssh(f"mkdir -p {VM_OUTPUT_PATH}")

    for scene in range(SCENE_START, SCENE_END + 1):
        ssh(f"mkdir -p {VM_ASSETS_PATH}/scene_{scene}")

    # 4. Upload JSON
    scp_to_vm(temp_json, f"{VM_ASSETS_PATH}/image2image_road.json")

    # 5. Upload images
    for scene, filename in image_map:
        local_img = WINDOWS_BASE_PATH / f"scene_{scene}" / filename
        remote_dir = f"{VM_ASSETS_PATH}/scene_{scene}"
        scp_to_vm(local_img, remote_dir)

    # 6. Run inference
    ssh(
        f"cd {VM_BASE_PATH} && "
        f"python examples/inference.py "
        f"-i assets/image_example/image2image_road.json "
        f"-o outputs/image2image"
    )

    # 7. Download results and place back
    for scene, filename in image_map:
        inferred_local = (
            WINDOWS_BASE_PATH
            / f"scene_{scene}"
            / filename.replace(".png", "_inferred.png")
        )

        remote_out = (
            f"{VM_OUTPUT_PATH}/scene_{scene}/{filename}"
        )

        scp_from_vm(remote_out, inferred_local)

    print("All scenes processed successfully")

if __name__ == "__main__":
    main()