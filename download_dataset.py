import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_url


load_dotenv()


REPO_ID = "Kevin-Pal/CUHK-X_Large_Model_Track"
REVISION = "main"

FILES = [
    "Large-Model-Track/Training/data/HAU.zip",
    "Large-Model-Track/Training/data/HARn.zip",
    "Large-Model-Track/Testing/data/large_model_track_test.zip",
    "Large-Model-Track/LMT_(IMU,Radar,Skeleton).zip",
]

OUTPUT_DIR = Path("hf_data_manual")


# All 4 files can download simultaneously.
CONCURRENT_FILES = 4

# Each individual ZIP may use up to 4 HTTP range connections.
#
# 4 files × 4 ranges = up to 16 active transfer connections.
#
# This is a good balance between speed and reliability.
CONNECTIONS_PER_FILE = 4


def build_url(filename: str) -> str:
    return hf_hub_url(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        revision=REVISION,
    )


def main():
    if shutil.which("aria2c") is None:
        raise SystemExit(
            "aria2c is not installed.\n"
            "Install it with:\n\n"
            "    brew install aria2"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN")

    #
    # aria2 input files let us give each URL its own output filename
    # and, if necessary, Authorization header.
    #
    # This also avoids putting HF_TOKEN directly into the command line.
    #
    lines = []

    for remote_path in FILES:
        url = build_url(remote_path)
        output_name = Path(remote_path).name

        lines.append(url)
        lines.append(f"  out={output_name}")

        if token:
            lines.append(f"  header=Authorization: Bearer {token}")

        lines.append("")

    input_text = "\n".join(lines)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="hf_aria2_",
            suffix=".txt",
            delete=False,
        ) as f:
            f.write(input_text)
            temp_path = f.name

        # Protect the temporary file if it contains HF_TOKEN.
        os.chmod(temp_path, 0o600)

        command = [
            "aria2c",

            # Use our generated URL list.
            f"--input-file={temp_path}",

            # Destination directory.
            f"--dir={OUTPUT_DIR}",

            # ---------------------------------------------------------
            # CONCURRENCY
            # ---------------------------------------------------------

            # Download all four independent ZIPs simultaneously.
            f"--max-concurrent-downloads={CONCURRENT_FILES}",

            # Split each huge ZIP into parallel HTTP ranges.
            f"--split={CONNECTIONS_PER_FILE}",

            # Allow that many connections to the same server per file.
            f"--max-connection-per-server={CONNECTIONS_PER_FILE}",

            # Don't bother splitting tiny pieces.
            "--min-split-size=16M",

            # ---------------------------------------------------------
            # RESUME / RELIABILITY
            # ---------------------------------------------------------

            # Resume partial downloads.
            "--continue=true",

            # Prefer preserving partial progress instead of starting over.
            "--always-resume=true",

            # Do not silently rename HAU.zip -> HAU.1.zip etc.
            "--auto-file-renaming=false",

            # Existing partial files may be continued.
            "--allow-overwrite=true",

            # If a connection effectively dies, don't leave it hanging
            # indefinitely at zero progress.
            "--lowest-speed-limit=16K",

            # Network timeouts.
            "--connect-timeout=10",
            "--timeout=30",

            # Retry failed HTTP requests.
            "--max-tries=50",
            "--retry-wait=2",

            # Fail after repeated genuine 404 responses.
            "--max-file-not-found=3",

            # ---------------------------------------------------------
            # PERFORMANCE
            # ---------------------------------------------------------

            # Avoid spending ages preallocating multi-GB files.
            "--file-allocation=none",

            # Buffer writes instead of constantly hitting disk.
            "--disk-cache=64M",

            # Keep HTTP connections alive where possible.
            "--enable-http-keep-alive=true",

            # No artificial bandwidth cap.
            "--max-overall-download-limit=0",
            "--max-download-limit=0",

            # ---------------------------------------------------------
            # OUTPUT
            # ---------------------------------------------------------

            "--summary-interval=2",
            "--console-log-level=notice",
            "--download-result=full",
            "--human-readable=true",
        ]

        print(
            f"Downloading {len(FILES)} files\n"
            f"Files in parallel:       {CONCURRENT_FILES}\n"
            f"Connections per file:    {CONNECTIONS_PER_FILE}\n"
            f"Maximum transfer streams: "
            f"{CONCURRENT_FILES * CONNECTIONS_PER_FILE}\n"
            f"Output: {OUTPUT_DIR.resolve()}\n"
        )

        subprocess.run(command, check=True)

        print("\nAll downloads completed.")

    except subprocess.CalledProcessError as exc:
        print(
            "\naria2 stopped before every download completed.\n"
            "Your partial downloads have NOT been thrown away.\n"
            "Run this script again and aria2 will resume them."
        )
        raise SystemExit(exc.returncode)

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()