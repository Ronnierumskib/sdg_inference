import re
import json
import subprocess
import shutil
import argparse
from pathlib import Path

# ==========================
# USER CONFIGURATION
# ==========================

VM_USER = "root"
VM_IP = "154.59.156.26"
VM_PORT = 20262

WINDOWS_BASE_PATH = Path("/mnt/c/isaac-sim/_out_basic_writer/2krun_low_res")
LOCAL_JSON_PATH = Path("image2image_road.json")
STAGING_DIR = Path("/tmp/cosmos_batch_upload")

VM_BASE_PATH = "/workspace/cosmos-transfer2.5"
VM_ASSETS_PATH = f"{VM_BASE_PATH}/assets/image_example"
VM_OUTPUT_PATH = f"{VM_BASE_PATH}/outputs/image2image"

SCENE_START = 0
SCENE_END = 9  # inclusive

MAX_BATCH_SIZE = 200  # Maximum images per inference call

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
# NEW HELPER FUNCTIONS
# ==========================

def batch_list(items, batch_size):
    """Split list into batches of specified size."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def check_staging_complete(image_map):
    """
    Check if staging directory exists and has all required files.
    Returns True if staging is complete, False otherwise.
    """
    if not STAGING_DIR.exists():
        print("[INFO] Staging directory does not exist")
        return False
    
    print("[INFO] Checking if staging directory is complete...")
    missing_files = []
    
    for scene, filename, image_name, inferred_name in image_map:
        # Check image
        scene_staging = STAGING_DIR / f"scene_{scene}"
        image_path = scene_staging / filename
        if not image_path.exists():
            missing_files.append(str(image_path))
        
        # Check JSON
        json_filename = f"batch_{scene}_{image_name}.json"
        json_path = STAGING_DIR / json_filename
        if not json_path.exists():
            missing_files.append(str(json_path))
    
    if missing_files:
        print(f"[INFO] Staging incomplete. Missing {len(missing_files)} files")
        return False
    
    print("[INFO] Staging directory is complete, reusing existing files")
    return True

def get_vm_asset_files():
    """
    Get list of all files in VM asset directory.
    Returns a set of absolute paths on the VM.
    """
    print("[INFO] Checking existing files on VM...")
    result = subprocess.run(
        f"ssh -p {VM_PORT} {VM_USER}@{VM_IP} 'find {VM_ASSETS_PATH} -type f 2>/dev/null || echo'",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("[WARN] Could not list VM files, assuming none exist")
        return set()
    
    files = set(line.strip() for line in result.stdout.strip().split('\n') if line.strip())
    print(f"[INFO] Found {len(files)} existing files on VM")
    return files

def get_files_to_upload(image_map):
    """
    Determine which files need to be uploaded to VM.
    Returns list of tuples: (local_path, relative_path)
    """
    vm_files = get_vm_asset_files()
    files_to_upload = []
    
    for scene, filename, image_name, inferred_name in image_map:
        # Check image
        rel_image_path = f"scene_{scene}/{filename}"
        vm_image_path = f"{VM_ASSETS_PATH}/{rel_image_path}"
        if vm_image_path not in vm_files:
            local_path = STAGING_DIR / rel_image_path
            files_to_upload.append((local_path, rel_image_path))
        
        # Check JSON
        json_filename = f"batch_{scene}_{image_name}.json"
        vm_json_path = f"{VM_ASSETS_PATH}/{json_filename}"
        if vm_json_path not in vm_files:
            local_path = STAGING_DIR / json_filename
            files_to_upload.append((local_path, json_filename))
    
    print(f"[INFO] {len(files_to_upload)} files need to be uploaded to VM")
    return files_to_upload

def upload_files_selectively(files_to_upload):
    """
    Upload only the files that don't exist on VM.
    Groups uploads by directory for efficiency.
    """
    if not files_to_upload:
        print("[INFO] All files already exist on VM, skipping upload")
        return
    
    # Group files by their parent directory
    by_dir = {}
    for local_path, rel_path in files_to_upload:
        parent = Path(rel_path).parent
        if parent not in by_dir:
            by_dir[parent] = []
        by_dir[parent].append((local_path, rel_path))
    
    # Upload each directory group
    for parent_dir, files in by_dir.items():
        if str(parent_dir) == '.':
            # Root level files (JSONs)
            for local_path, rel_path in files:
                print(f"[UPLOAD] {rel_path}")
                scp_to_vm(str(local_path), f"{VM_ASSETS_PATH}/{rel_path}")
        else:
            # Scene directory files
            ssh(f"mkdir -p {VM_ASSETS_PATH}/{parent_dir}")
            for local_path, rel_path in files:
                print(f"[UPLOAD] {rel_path}")
                scp_to_vm(str(local_path), f"{VM_ASSETS_PATH}/{rel_path}")

# ==========================
# MAIN LOGIC
# ==========================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Batch upload and inference script for Cosmos Transfer 2.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (upload and inference)
  python %(prog)s
  
  # Download results only (skip upload and inference)
  python %(prog)s --download-only
  python %(prog)s -d
        """
    )
    parser.add_argument(
        '-d', '--download-only',
        action='store_true',
        help='Skip upload and inference, only download and organize results from VM'
    )
    
    args = parser.parse_args()
    
    rgb_pattern = re.compile(r"^rgb_\d{4}\.png$")

    # (scene, filename, image_name, inferred_name)
    image_map = []

    # 1. Collect all images (needed even for download-only to know what to download)
    print("[STEP 1] Collecting images...")
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

    print(f"[INFO] Found {len(image_map)} images to process")

    # If download-only mode, skip to step 5
    if args.download_only:
        print("\n[MODE] Download-only mode enabled, skipping upload and inference")
        print("=" * 60)
        download_and_organize_results(image_map)
        print("\n=================================")
        print("Download completed successfully.")
        print(f"Retrieved results for {len(image_map)} images")
        print("=================================")
        return

    # 2. Build local staging directory (if needed)
    print("[STEP 2] Preparing staging directory...")
    
    staging_complete = check_staging_complete(image_map)
    
    if not staging_complete:
        if STAGING_DIR.exists():
            print("[INFO] Removing incomplete staging directory")
            shutil.rmtree(STAGING_DIR)
        
        print("[INFO] Creating staging directory")
        STAGING_DIR.mkdir(parents=True)

        with open(LOCAL_JSON_PATH, "r") as f:
            base_cfg = json.load(f)

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
        
        print(f"[INFO] Staging directory prepared with {len(image_map)} images")

    # 3. Upload files to VM (only missing ones)
    print("[STEP 3] Uploading files to VM...")
    ssh(f"mkdir -p {VM_ASSETS_PATH}")
    
    files_to_upload = get_files_to_upload(image_map)
    upload_files_selectively(files_to_upload)

    # Ensure per-scene output dirs exist on the VM
    scenes = sorted({scene for scene, *_ in image_map})
    ssh(f"mkdir -p " + " ".join(f"{VM_BASE_PATH}/scene_{s}" for s in scenes))

    # 4. Run inference in batches
    print(f"[STEP 4] Running inference in batches of {MAX_BATCH_SIZE}...")
    
    batches = list(batch_list(image_map, MAX_BATCH_SIZE))
    total_batches = len(batches)
    
    for batch_idx, batch in enumerate(batches, 1):
        print(f"[BATCH {batch_idx}/{total_batches}] Processing {len(batch)} images...")
        
        # Build JSON paths for this batch
        vm_json_paths = []
        for scene, filename, image_name, inferred_name in batch:
            json_filename = f"batch_{scene}_{image_name}.json"
            vm_json_paths.append(f"{VM_ASSETS_PATH}/{json_filename}")
        
        input_list = " ".join(vm_json_paths)
        ssh(
            f"cd {VM_BASE_PATH} && source .venv/bin/activate && "
            f"python examples/inference.py -i {input_list} -o outputs/image2image"
        )
        print(f"[BATCH {batch_idx}/{total_batches}] Completed")

    # 5-6. Download and organize results
    download_and_organize_results(image_map)

    # 7. Cleanup staging
    print("[STEP 7] Cleaning up...")
    shutil.rmtree(STAGING_DIR)

    print("\n=================================")
    print("Batch inference completed successfully.")
    print(f"Processed {len(image_map)} images in {total_batches} batches")
    print("=================================")

def download_and_organize_results(image_map):
    """Download results from VM and organize them into scene folders."""
    # 5. Download all results (only .jpg files)
    print("[STEP 5] Downloading results from VM...")
    local_results = STAGING_DIR / "results"
    local_results.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create the image2image subdirectory locally
        local_image2image_dir = local_results / "image2image"
        local_image2image_dir.mkdir(exist_ok=True)
        
        # Download only .jpg files from the VM output directory
        print("[INFO] Downloading only .jpg files...")
        vm_image2image_path = f"{VM_OUTPUT_PATH}/image2image"
        
        # Use scp with wildcard to download only .jpg files
        result = subprocess.run(
            f"scp -P {VM_PORT} {VM_USER}@{VM_IP}:{vm_image2image_path}/*.jpg {local_image2image_dir}/",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Failed to download results: {result.stderr}")
            print("[INFO] Make sure the VM output directory exists and contains .jpg results")
            return
            
        # Count downloaded files
        downloaded_files = list(local_image2image_dir.glob("*.jpg"))
        print(f"[INFO] Downloaded {len(downloaded_files)} .jpg files")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download results: {e}")
        print("[INFO] Make sure the VM output directory exists and contains results")
        return

    # 6. Move each result to the correct scene folder
    print("[STEP 6] Organizing results...")
    moved_count = 0
    missing_count = 0
    
    for scene, filename, image_name, inferred_name in image_map:
        result_file = local_results / "image2image" / f"{inferred_name}.jpg"
        dest = WINDOWS_BASE_PATH / f"scene_{scene}" / f"{inferred_name}.jpg"
        if result_file.exists():
            shutil.move(str(result_file), dest)
            print(f"[RESULT] {inferred_name}.jpg -> scene_{scene}/")
            moved_count += 1
        else:
            print(f"[WARN] Expected result not found: {result_file}")
            missing_count += 1
    
    print(f"[INFO] Successfully moved {moved_count} results")
    if missing_count > 0:
        print(f"[WARN] {missing_count} expected results were not found")
    
    # Cleanup results directory after organizing
    if local_results.exists():
        shutil.rmtree(local_results)
        print("[INFO] Cleaned up temporary results directory")

if __name__ == "__main__":
    main()