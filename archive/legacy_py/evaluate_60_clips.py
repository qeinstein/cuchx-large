import cv2, io, base64, json, urllib.request, time, re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import os as _os
API_KEY = _os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENROUTER_API_KEY env var (never hardcode keys).")
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
    np.random.seed(123)
    clips_f0 = oof[oof.fold == 0]['path'].drop_duplicates().sample(20, random_state=101).tolist()
    clips_f1 = oof[oof.fold == 1]['path'].drop_duplicates().sample(20, random_state=102).tolist()
    clips_f2 = oof[oof.fold == 2]['path'].drop_duplicates().sample(20, random_state=103).tolist()
    eval_clips = clips_f0 + clips_f1 + clips_f2

    eval_df = oof[oof.path.isin(eval_clips)].copy()
    print(f"Evaluating {len(eval_df)} validation questions across {len(eval_clips)} clips...")

    cache_f = np.load('extended_features_all.npz', allow_pickle=True)
    tr_paths = list(cache_f['train_paths'])
    p_to_idx = {p: i for i, p in enumerate(tr_paths)}
    X_all = cache_f['X_train']

    vlm_preds = {}
    total_clips = len(eval_clips)

    for c_idx, clip_path in enumerate(eval_clips):
        clip_df = eval_df[eval_df.path == clip_path]
        b64_img, dur = create_contact_sheet(clip_path)

        sensor_evidence = []
        if clip_path in p_to_idx:
            feat = X_all[p_to_idx[clip_path]]
            cadence_hz = feat[120] if len(feat) > 120 else 1.0
            jerk_val = feat[150] if len(feat) > 150 else 0.5
            sensor_evidence.append(f"- Estimated cadence: {cadence_hz:.1f} Hz")
            sensor_evidence.append(f"- Motion jerk: {jerk_val:.2f}")

        q_texts = []
        for _, r in clip_df.iterrows():
            qid = r['qa_id']
            cat = r['category']
            q_str = r['question']
            opts = [f"A: {r['A']}", f"B: {r['B']}", f"C: {r['C']}", f"D: {r['D']}"]
            v7_p = r['pred']
            q_texts.append(f"[{qid}] Category: {cat}\nQuestion: {q_str}\nOptions:\n" + "\n".join(opts) + f"\n(Sensor baseline recommendation: {v7_p})")

        prompt = f"""You are a master multimodal human activity recognition judge.
This video clip is {dur:.1f} seconds long, recorded with depth cameras and sensors.
Attached is a chronologically ordered 16-frame contact sheet (0.0s to {dur:.1f}s, top-left to bottom-right).

SENSOR EVIDENCE:
""" + "\n".join(sensor_evidence) + f"""

QUESTIONS FOR THIS CLIP:
""" + "\n\n".join(q_texts) + """

INSTRUCTIONS:
1. Examine the visual contact sheet carefully to see what activities occur and when.
2. For single/combination/emotion/object questions: answer with a single letter (A, B, C, or D).
3. For multi questions: answer with all applicable letters (e.g. A, BD, ABC).
4. For sequence questions: answer with the 4-letter permutation of ABCD in chronological order.
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
        if (c_idx + 1) % 10 == 0 or (c_idx + 1) == total_clips:
            print(f"[{c_idx+1}/{total_clips}] Processed {c_idx+1} clips, parsed {len(vlm_preds)} questions so far")

    # Save results
    eval_df['vlm_pred'] = eval_df['qa_id'].map(vlm_preds).fillna(eval_df['pred'])
    eval_df['vlm_pred_clean'] = eval_df['vlm_pred'].astype(str).str.strip().str.upper()
    eval_df['v7_pred_clean'] = eval_df['pred'].astype(str).str.strip().str.upper()
    eval_df['ans_clean'] = eval_df['answer'].astype(str).str.strip().str.upper()

    eval_df['v7_correct'] = (eval_df['v7_pred_clean'] == eval_df['ans_clean']).astype(int)
    eval_df['vlm_correct'] = (eval_df['vlm_pred_clean'] == eval_df['ans_clean']).astype(int)

    # Fused Category Specialist Ensemble:
    # Single -> VLM
    # Combination -> VLM
    # Emotion -> v7
    # Multi -> v7
    # Object -> v7
    # Sequence -> v7
    fused_preds = []
    for _, r in eval_df.iterrows():
        cat = r['category']
        if cat in ['single', 'combination']:
            fused_preds.append(r['vlm_pred_clean'])
        else:
            fused_preds.append(r['v7_pred_clean'])
    eval_df['fused_pred'] = fused_preds
    eval_df['fused_correct'] = (eval_df['fused_pred'] == eval_df['ans_clean']).astype(int)

    v7_acc = eval_df['v7_correct'].mean() * 100
    vlm_acc = eval_df['vlm_correct'].mean() * 100
    fused_acc = eval_df['fused_correct'].mean() * 100

    print("\n" + "="*50)
    print(f"MULTI-FOLD BENCHMARK RESULTS ON {len(eval_df)} VALIDATION QUESTIONS:")
    print(f"v7 Sensor Baseline:        {eval_df['v7_correct'].sum()} / {len(eval_df)} ({v7_acc:.2f}%)")
    print(f"Gemini 2.5 Flash VLM alone:{eval_df['vlm_correct'].sum()} / {len(eval_df)} ({vlm_acc:.2f}%)")
    print(f"Fused Category Specialist: {eval_df['fused_correct'].sum()} / {len(eval_df)} ({fused_acc:.2f}%)")
    print(f"NET GAIN OVER v7:          {fused_acc - v7_acc:+.2f}%")

    print("\nCategory Breakdown (Fused vs v7):")
    for cat, g in eval_df.groupby('category'):
        v7_c = g['v7_correct'].sum()
        fused_c = g['fused_correct'].sum()
        print(f"  {cat:20s} (N={len(g):2d}) | v7: {v7_c:2d} ({g['v7_correct'].mean()*100:.1f}%) -> Fused: {fused_c:2d} ({g['fused_correct'].mean()*100:.1f}%) [Delta: {g['fused_correct'].mean()*100 - g['v7_correct'].mean()*100:+.1f}%]")

    eval_df.to_csv('eval_60_clips_results.csv', index=False)
    print("Results saved to eval_60_clips_results.csv")

if __name__ == '__main__':
    main()
