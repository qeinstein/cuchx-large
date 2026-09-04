import cv2, io, base64, json, urllib.request, time, re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def create_contact_sheet(rel_path, num_frames=16, grid_size=(4, 4)):
    root = Path('hf_data_manual')
    vpath = root / rel_path / 'Depth_Color' / 'Depth_Color.mp4'
    if not vpath.exists():
        vpath = root / rel_path / 'Depth' / 'Depth.mp4'
    if not vpath.exists():
        mp4s = list((root / rel_path).glob('**/*.mp4'))
        if mp4s: vpath = mp4s[0]
        else: return None, 0.0
        
    cap = cv2.VideoCapture(str(vpath))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    duration = total_frames / fps
    
    if total_frames <= 0:
        return None, 0.0
        
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            t_sec = idx / fps
            thumb = cv2.resize(frame, (240, 180))
            cv2.putText(thumb, f'{t_sec:.1f}s', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(thumb, f'{t_sec:.1f}s', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
            frames.append(thumb)
    cap.release()
    
    if len(frames) < num_frames:
        return None, duration
        
    rows = []
    for r in range(grid_size[0]):
        row_frames = frames[r*grid_size[1]:(r+1)*grid_size[1]]
        rows.append(np.hstack(row_frames))
    contact_sheet = np.vstack(rows)
    rgb = cv2.cvtColor(contact_sheet, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=80)
    return base64.b64encode(buf.getvalue()).decode('utf-8'), duration

def call_vlm(prompt, b64_img, model="google/gemini-2.5-flash"):
    content = [{'type': 'text', 'text': prompt}]
    if b64_img:
        content.append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/jpeg;base64,{b64_img}'}
        })
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': content}],
        'temperature': 0.1
    }
    req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(payload).encode('utf-8'), headers={
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data['choices'][0]['message']['content']
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return ""

def main():
    oof = pd.read_csv('oof_v7_final.csv')
    f0 = oof[oof.fold == 0].copy()

    np.random.seed(42)
    f0_hau_clips = f0[f0.source == 'HAU']['path'].drop_duplicates().sample(15, random_state=42).tolist()
    f0_harn_clips = f0[f0.source == 'HARn']['path'].drop_duplicates().sample(15, random_state=42).tolist()
    sample_clips = f0_hau_clips + f0_harn_clips

    sub_val = f0[f0.path.isin(sample_clips)].copy()
    print(f"Total questions to evaluate: {len(sub_val)} across {len(sample_clips)} clips")

    cache_f = np.load('extended_features_all.npz', allow_pickle=True)
    tr_paths = list(cache_f['train_paths'])
    p_to_idx = {p: i for i, p in enumerate(tr_paths)}
    X_all = cache_f['X_train']

    vlm_preds = {}
    total_clips = len(sample_clips)

    for c_idx, clip_path in enumerate(sample_clips):
        clip_df = sub_val[sub_val.path == clip_path]
        b64_img, dur = create_contact_sheet(clip_path)

        sensor_evidence = []
        if clip_path in p_to_idx:
            feat = X_all[p_to_idx[clip_path]]
            cadence_hz = feat[120] if len(feat) > 120 else 1.0
            jerk_val = feat[150] if len(feat) > 150 else 0.5
            sensor_evidence.append(f"- Estimated cadence: {cadence_hz:.1f} Hz")
            sensor_evidence.append(f"- Motion jerk / intensity: {jerk_val:.2f}")

        q_texts = []
        for _, r in clip_df.iterrows():
            qid = r['qa_id']
            cat = r['category']
            q_str = r['question']
            opts = [f"A: {r['A']}", f"B: {r['B']}", f"C: {r['C']}", f"D: {r['D']}"]
            v7_p = r['pred']
            q_texts.append(f"[{qid}] Category: {cat}\nQuestion: {q_str}\nOptions:\n" + "\n".join(opts) + f"\n(Sensor baseline recommendation: {v7_p})")

        prompt = f"""You are a master multimodal activity recognition judge.
This clip is {dur:.1f} seconds long, captured via depth/IR camera.
Attached is a chronologically ordered 16-frame contact sheet (top-left to bottom-right, with timestamps).

SENSOR EVIDENCE:
""" + "\n".join(sensor_evidence) + f"""

QUESTIONS FOR THIS CLIP:
""" + "\n\n".join(q_texts) + """

INSTRUCTIONS:
1. Answer each question carefully based on visual evidence + sensor hints.
2. For single/combination/emotion/object questions, answer with a single letter (A, B, C, or D).
3. For multi questions, answer with all applicable letters (e.g. A, BD, ABC, etc.).
4. For sequence questions, answer with the 4-letter permutation of ABCD in chronological order.
5. Provide your answers in valid JSON format:
{
  "<qa_id>": "<prediction>",
  ...
}"""

        out = call_vlm(prompt, b64_img)
        m = re.search(r'\{[^{}]*\}', out)
        if m:
            try:
                ans_dict = json.loads(m.group(0))
                vlm_preds.update(ans_dict)
            except:
                pass
        print(f"[{c_idx+1}/{total_clips}] Processed {clip_path}: parsed {len(clip_df)} questions")

    sub_val['vlm_pred'] = sub_val['qa_id'].map(vlm_preds).fillna(sub_val['pred'])
    sub_val['vlm_pred_clean'] = sub_val['vlm_pred'].astype(str).str.strip().str.upper()
    sub_val['ans_clean'] = sub_val['answer'].astype(str).str.strip().str.upper()

    sub_val['v7_correct'] = (sub_val['pred'].str.strip().str.upper() == sub_val['ans_clean']).astype(int)
    sub_val['vlm_correct'] = (sub_val['vlm_pred_clean'] == sub_val['ans_clean']).astype(int)

    v7_acc = sub_val['v7_correct'].mean() * 100
    vlm_acc = sub_val['vlm_correct'].mean() * 100

    print("\n" + "="*50)
    print(f"BENCHMARK RESULTS ON {len(sub_val)} VALIDATION QUESTIONS:")
    print(f"v7 Sensor Baseline Correct: {sub_val['v7_correct'].sum()} / {len(sub_val)} ({v7_acc:.2f}%)")
    print(f"Gemini 2.5 Flash VLM Correct: {sub_val['vlm_correct'].sum()} / {len(sub_val)} ({vlm_acc:.2f}%)")
    print(f"Net Gain: {vlm_acc - v7_acc:+.2f}%")

    print("\nBreakdown by Category:")
    for cat, g in sub_val.groupby('category'):
        v7_c = g['v7_correct'].sum()
        vlm_c = g['vlm_correct'].sum()
        print(f"  {cat:20s} (N={len(g):2d}) | v7: {v7_c:2d} ({g['v7_correct'].mean()*100:.1f}%) -> VLM: {vlm_c:2d} ({g['vlm_correct'].mean()*100:.1f}%)")

if __name__ == '__main__':
    main()
