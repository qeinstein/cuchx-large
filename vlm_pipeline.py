import os
import pandas as pd
import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

def extract_frames(video_path, num_frames=4):
    """
    Extracts exactly `num_frames` from a multimodal sequence (treated as video).
    If missing, returns black frames.
    """
    if not os.path.exists(video_path):
        return [Image.new('RGB', (224, 224), color='black')] * num_frames
        
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

def evaluate_vlm(csv_path="splits/fold_0_val.csv", data_dir="hf_data_manual/"):
    df = pd.read_csv(csv_path)
    print(f"Loading {MODEL_ID} in 4-bit...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    predictions = []
    correct = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Construct path, assuming modalities are extracted inside data_dir
        # For baseline, we just test depth if available
        video_path = os.path.join(data_dir, row['path'] + "/depth/depth.mp4")
        frames = extract_frames(video_path, num_frames=4)
        prompt = format_prompt(row)
        
        inputs = processor(text=prompt, images=frames, return_tensors="pt").to(model.device, torch.float16)
        
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=10, temperature=0.2, do_sample=False)
            
        generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
        answer = generated_text.split("ASSISTANT:")[-1].strip()
        clean_answer = "".join([c for c in answer if c in "ABCD"])
        if not clean_answer: clean_answer = "A"
        
        predictions.append(clean_answer)
        if clean_answer == str(row['answer']):
            correct += 1
            
    acc = correct / len(df)
    print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate_vlm()
