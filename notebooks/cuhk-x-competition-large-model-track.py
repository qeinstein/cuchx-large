# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# %%
pip install -U bitsandbytes>=0.46.1

# %%
import os
import pandas as pd
import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm
import matplotlib.pyplot as plt

MODEL_ID = "llava-hf/llava-1.5-7b-hf" 

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print(f"Loading Model {MODEL_ID} in 4-bit...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

def extract_frames(video_path, num_frames=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return [Image.new('RGB', (224, 224), color='black')] * num_frames
        
    frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            frames.append(Image.new('RGB', (224, 224), color='black'))
            
    cap.release()
    return frames

def format_prompt(row):
    question = row['question']
    category = row['category']
    
    options_text = f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}"
    if pd.notna(row['D']) and str(row['D']).strip() != "":
        options_text += f"\nD) {row['D']}"
        
    if category in ['single', 'emotion', 'object_interaction', 'combination']:
        instruction = "Select the single correct option. Reply strictly with only one letter (e.g., A, B, C, or D)."
    elif category == 'multi':
        instruction = "Select all correct options. Reply strictly with the letters combined (e.g., AB, ACD)."
    elif category == 'sequence':
        instruction = "Order the actions chronologically. Reply strictly with the sequence of letters (e.g., DBCA)."
    else:
        instruction = "Reply strictly with the correct letter(s)."

    prompt = f"USER: <image>\n<image>\n<image>\n<image>\nBased on these sequential frames from a video, answer the following question.\nQuestion: {question}\n{options_text}\n{instruction}\nASSISTANT:"
    return prompt

def run_vlm_inference(csv_path, data_dir, output_csv="submission.csv"):
    df = pd.read_csv(csv_path)
    predictions = []
    
    if not data_dir.endswith('/'):
        data_dir += '/'

    print(f"Starting inference on {len(df)} questions...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_path = os.path.join(data_dir, row['path'])
        
        try:
            frames = extract_frames(video_path, num_frames=4)
            prompt = format_prompt(row)
            
            inputs = processor(text=prompt, images=frames, return_tensors="pt").to(model.device, torch.float16)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.2,
                    do_sample=False
                )
            
            generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
            answer = generated_text.split("ASSISTANT:")[-1].strip()
            clean_answer = "".join([c for c in answer if c in "ABCD"])
            
            if not clean_answer:
                clean_answer = "A"
                
            predictions.append(clean_answer)
            
        except Exception as e:
            print(f"Error processing {row['qa_id']}: {e}")
            predictions.append("A")
            
        torch.cuda.empty_cache()

    submission_df = pd.DataFrame({
        'qa_id': df['qa_id'],
        'prediction': predictions
    })
    
    submission_df.to_csv(output_csv, index=False)
    print(f"Finished! Saved to {output_csv}")

    plt.figure(figsize=(12, 6))
    submission_df['prediction'].value_counts().sort_index().plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Distribution of Predicted Answers')
    plt.xlabel('Predicted Option(s)')
    plt.ylabel('Number of Predictions')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('prediction_distribution.png')
    plt.show()

if __name__ == "__main__":
    test_csv = '/kaggle/input/competitions/cuhk-x-competition-large-model-track/test_qa.csv'
    dataset_base_dir = '/kaggle/input/competitions/cuhk-x-competition-large-model-track/' 
    run_vlm_inference(test_csv, dataset_base_dir)
