import os
import requests
import time

urls = [
    "https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Large_Model_Track/resolve/main/Large-Model-Track/Training/data/HAU.zip",
    "https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Large_Model_Track/resolve/main/Large-Model-Track/Training/data/HARn.zip",
    "https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Large_Model_Track/resolve/main/Large-Model-Track/Testing/data/large_model_track_test.zip",
    "https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Large_Model_Track/resolve/main/Large-Model-Track/LMT_(IMU,Radar,Skeleton).zip"
]

token = os.environ.get("HF_TOKEN", "")
headers = {"Authorization": f"Bearer {token}"}

os.makedirs("hf_data_manual", exist_ok=True)

for url in urls:
    filename = url.split("/")[-1]
    filepath = os.path.join("hf_data_manual", filename)
    
    while True:
        try:
            resume_header = headers.copy()
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                resume_header["Range"] = f"bytes={file_size}-"
            else:
                file_size = 0
            
            with requests.get(url, headers=resume_header, stream=True, timeout=30) as r:
                if r.status_code == 416:
                    print(f"{filename} already fully downloaded.")
                    break
                if r.status_code not in [200, 206]:
                    print(f"Failed to fetch {url}, status code: {r.status_code}")
                    break
                    
                with open(filepath, "ab") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            print(f"{filename} download complete!")
            break 
        except Exception as e:
            print(f"Error downloading {filename}: {e}. Retrying in 5s...")
            time.sleep(5)
