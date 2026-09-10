"""Recover exact test clip nesting from depth frames; no label or submission access."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv,json,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/post332_20260908';OUT.mkdir(exist_ok=True)
CACHE=OUT/'pixel_fingerprints.npz'
def encode(path):
    clip=path.parent.parent.name;cap=cv2.VideoCapture(str(path));frames=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        frames.append(cv2.resize(frame,(20,15),interpolation=cv2.INTER_AREA).reshape(-1))
    cap.release()
    return clip,np.array(frames,dtype=np.uint8)
def main():
    cv2.setNumThreads(1)
    if CACHE.exists():
        z=np.load(CACHE);features={k:z[k] for k in z.files}
    else:
        paths=list((ROOT/'hf_data_manual/large_model_track_test').glob('*/Depth_Color/*.mp4'))
        features={}
        with ThreadPoolExecutor(max_workers=4) as ex:
            fs=[ex.submit(encode,p) for p in paths]
            for i,f in enumerate(as_completed(fs),1):
                k,x=f.result();features[k]=x
                if i%20==0:print('encoded',i,'/',len(paths),flush=True)
        np.savez_compressed(CACHE,**features)
    parents={k:x for k,x in features.items() if int(k[-4:])>=65 and len(x)}
    records=[]
    for k,child in sorted(features.items()):
        if int(k[-4:])>=65 or not len(child):continue
        best=[];nf=len(child);anchors=np.unique(np.linspace(0,nf-1,min(5,nf),dtype=int))
        for p,parent in parents.items():
            if len(parent)<nf:continue
            distances=np.zeros(len(parent)-nf+1)
            for i in anchors:
                distances+=np.mean(np.abs(parent[i:i+len(distances)].astype(float)-child[i].astype(float)),axis=1)/len(anchors)
            offset=int(np.argmin(distances));best.append((float(distances[offset]),p,offset))
        best.sort();top=best[:3]
        rec={'child':k,'nframes':nf,'best':top,'gap':top[1][0]-top[0][0] if len(top)>1 else None}
        records.append(rec)
    (OUT/'pixel_nesting.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
