import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_url

REPO_ID = "llava-hf/llava-1.5-7b-hf"
OUTPUT_DIR = Path("llava_local")

def main():
    if shutil.which("aria2c") is None:
        raise SystemExit("aria2c is not installed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    
    print(f"Fetching file list for {REPO_ID}...")
    files = api.list_repo_files(REPO_ID)
    
    # Filter out pytorch *.bin files since we will use the .safetensors
    files = [f for f in files if not f.endswith(".bin") and not f.endswith(".pt")]
    
    lines = []
    for filename in files:
        url = hf_hub_url(repo_id=REPO_ID, filename=filename)
        lines.append(url)
        lines.append(f"  out={filename}")
        lines.append("")

    input_text = "\n".join(lines)
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", prefix="hf_aria2_model_", suffix=".txt", delete=False) as f:
            f.write(input_text)
            temp_path = f.name

        command = [
            "aria2c",
            f"--input-file={temp_path}",
            f"--dir={OUTPUT_DIR}",
            "--max-concurrent-downloads=8",
            "--split=8",
            "--max-connection-per-server=8",
            "--min-split-size=16M",
            "--continue=true",
            "--always-resume=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--lowest-speed-limit=16K",
            "--connect-timeout=10",
            "--timeout=30",
            "--max-tries=50",
            "--retry-wait=2",
            "--file-allocation=none",
            "--disk-cache=64M",
            "--enable-http-keep-alive=true",
            "--summary-interval=5",
            "--console-log-level=notice",
        ]

        print(f"Downloading {len(files)} model files to {OUTPUT_DIR.resolve()} using aria2c...")
        subprocess.run(command, check=True)
        print("\nAll model downloads completed!")

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

if __name__ == "__main__":
    main()
