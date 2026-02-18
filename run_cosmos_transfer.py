import re
import json
import subprocess
import shutil
from pathlib import Path

# ==========================
# USER CONFIGURATION
# ==========================

VM_USER = "root"
VM_IP = "58.123.93.163"
VM_PORT = 40264

WINDOWS_BASE_PATH = Path("/mnt/c/isaac-sim/_out_basic_writer")
LOCAL_JSON_PATH = Path("image2image_road.json")
STAGING_DIR = Path("/tmp/cosmos_batch_upload")

VM_BASE_PATH = "/workspace/cosmos-transfer2.5"
VM_ASSETS_PATH = f"{VM_BASE_PATH}/assets/image_example"
VM_OUTPUT_PATH = f"{VM_BASE_PATH}/outputs/image2image"

SCENE_START = 1
SCENE_END = 8  # inclusive

# ==========================
# HELPERS
# ==========================

def run(cmd):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def ssh(cmd):
    run(f"ssh -p {VM_PORT} {VM_USER}@{VM_IP} \"{cmd}\"")

def scp_to_vm(local, remote):
    run(f"scp -P {VM_PORT} -r {local} {VM_USER}@{VM_IP}:{remote}")

def scp_from_vm(remote, local):
    run(f"scp -P {VM_PORT} -r {VM_USER}@{VM_IP}:{remote} {local}")

def ssh_agent_setup():
    #run("eval "$(ssh-agent -s)"")
    print("lorem ipsum")
# ==========================
# MAIN LOGIC
# ==========================

def main():
    rgb_pattern = re.compile(r"^rgb_\d{4}\.png$")

    # (scene, filename, image_name, inferred_name)
    image_map = []

    # 1. Collect all images
    for scene in range(SCENE_START, SCENE_END + 1):
        scene_dir = WINDOWS_BASE_PATH / f"scene_{scene}"
        if not scene_dir.exists():
            print(f"[WARN] Missing {scene_dir}, skipping")
            continue
        for img in sorted(scene_dir.iterdir()):
            if img.is_file() and rgb_pattern.match(img.name):
                image_name = img.stem
                inferred_name = f"scene_{scene}_{image_name}_inferred"
                image_map.append((scene, img.name, image_name, inferred_name))

    if not image_map:
        print("[ERROR] No images found.")
        return

    # 2. Build local staging directory mirroring the VM asset structure
    #    STAGING_DIR/
    #      scene_3/
    #        rgb_0001.png
    #        rgb_0002.png
    #        ...
    #      batch_3_rgb_0001.json
    #      batch_3_rgb_0002.json
    #      ...
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    with open(LOCAL_JSON_PATH, "r") as f:
        base_cfg = json.load(f)

    vm_json_paths = []

    for scene, filename, image_name, inferred_name in image_map:
        # Copy image into staging
        scene_staging = STAGING_DIR / f"scene_{scene}"
        scene_staging.mkdir(exist_ok=True)
        shutil.copy(WINDOWS_BASE_PATH / f"scene_{scene}" / filename, scene_staging / filename)

        # Write per-image JSON into staging
        cfg = base_cfg.copy()
        cfg["video_path"] = f"scene_{scene}/{filename}"
        cfg["name"] = inferred_name
        cfg["num_video_frames_per_chunk"] = 1
        cfg["max_frames"] = 1

        json_filename = f"batch_{scene}_{image_name}.json"
        with open(STAGING_DIR / json_filename, "w") as f:
            json.dump(cfg, f, indent=4)

        vm_json_paths.append(f"{VM_ASSETS_PATH}/{json_filename}")

    # 3. Ensure VM asset dir exists, then upload everything in ONE scp call
    ssh(f"mkdir -p {VM_ASSETS_PATH}")
    scp_to_vm(f"{STAGING_DIR}/*", VM_ASSETS_PATH)

    # Also ensure per-scene output dirs exist on the VM
    scenes = sorted({scene for scene, *_ in image_map})
    ssh(f"mkdir -p " + " ".join(f"{VM_BASE_PATH}/scene_{s}" for s in scenes))

    # 4. Run inference ONCE with all JSON files
    input_list = " ".join(vm_json_paths)
    ssh(
        f"cd {VM_BASE_PATH} && source .venv/bin/activate && "
        f"python examples/inference.py -i {input_list} -o outputs/image2image"
    )

    # 5. Download all results in ONE scp call
    #    Grab the entire output dir and sort locally
    local_results = STAGING_DIR / "results"
    local_results.mkdir()
    scp_from_vm(VM_OUTPUT_PATH, str(local_results))

    # Move each result to the correct scene folder
    for scene, filename, image_name, inferred_name in image_map:
        result_file = local_results / "image2image" / f"{inferred_name}.jpg"
        dest = WINDOWS_BASE_PATH / f"scene_{scene}" / f"{inferred_name}.jpg"
        if result_file.exists():
            shutil.move(str(result_file), dest)
        else:
            print(f"[WARN] Expected result not found: {result_file}")

    # 6. Cleanup staging
    shutil.rmtree(STAGING_DIR)

    print("Batch inference completed successfully.")

if __name__ == "__main__":
    main()