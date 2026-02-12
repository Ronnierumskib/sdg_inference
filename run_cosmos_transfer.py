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
VM_IP = "51.154.62.90"
VM_PORT = 61080

# Local paths (WSL)
WINDOWS_BASE_PATH = Path("/mnt/c/isaac-sim/_out_basic_writer")
LOCAL_JSON_PATH = Path("image2image_road.json") 
STAGING_DIR = Path("/tmp/cosmos_upload")

# VM paths
VM_BASE_PATH = "/workspace/cosmos-transfer2.5"
VM_ASSETS_PATH = f"{VM_BASE_PATH}/assets/image_example"
VM_OUTPUT_PATH = f"{VM_BASE_PATH}/outputs/image2image"

# Scene configuration
SCENE_START = 3
SCENE_END = 3   # inclusive

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
                image_map.append((scene, img.name))

    for scene, filename in image_map:
        image_name = filename.replace(".png", "")
        inferred_name = image_name + "_inferred"

        # ---- Update JSON per image ----
        with open(LOCAL_JSON_PATH, "r") as f:
            cfg = json.load(f)

        cfg["video_path"] = f"scene_{scene}/{filename}"
        cfg["name"] = inferred_name
        cfg["num_video_frames_per_chunk"] = 1
        cfg["max_frames"] = 1

        temp_json = Path("image2image_road_temp.json")
        with open(temp_json, "w") as f:
            json.dump(cfg, f, indent=4)

        # ---- Prepare VM dirs ----
        ssh(f"mkdir -p {VM_ASSETS_PATH}/scene_{scene}")
        ssh(f"mkdir -p {VM_BASE_PATH}/scene_{scene}")

        # ---- Upload image + JSON ----
        scp_to_vm(temp_json, f"{VM_ASSETS_PATH}/image2image_road.json")
        scp_to_vm(
            WINDOWS_BASE_PATH / f"scene_{scene}" / filename,
            f"{VM_ASSETS_PATH}/scene_{scene}"
        )

        # ---- Run inference for THIS image ----
        ssh(
            f"cd {VM_BASE_PATH} && source .venv/bin/activate && python examples/inference.py -i assets/image_example/image2image_road.json -o scene_{scene}"
        )

        # ---- Download result ----
        remote_out = (
            f"{VM_BASE_PATH}/scene_{scene}/{inferred_name}.jpg"
        )
        local_out = (
            WINDOWS_BASE_PATH
            / f"scene_{scene}"
#            / f"{inferred_name}.jpg"
        )

        scp_from_vm(remote_out, local_out)

    print("All scenes processed successfully")

if __name__ == "__main__":
    main()