"""Low-memory raw Depth_Color action probe for sensor-missing HARn test clips.

This is deliberately independent of the shipped solver.  It trains on HAU *single*
questions only (the answer text is the weak clip label), keeps users disjoint, and
reports held-out accuracy before making any test suggestions.  The test rows are
never used to choose a model or a threshold.
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/post332_20260908"
BASE = pd.read_csv(ROOT / "submission_097076_332of342_CHAMPION.csv")
TE = pd.read_csv(ROOT / "test_qa.csv").drop(columns=["prediction"], errors="ignore")
TR = pd.read_csv(ROOT / "training_qa.csv", encoding="utf-8-sig")

HARN2HAU = json.loads((ROOT / "champ/vocab.json").read_text())["HARN2HAU"]
CANON = set(HARN2HAU.values())

def norm_action(s: object) -> str:
    x = str(s).strip().lower()
    repl = {
        "doing ": "", "their ": "", "a ": "", "the ": "",
        "floor": "", "documents": "document", "pages": "page",
        "food": "", "water": "", "surfaces": "surface", "bowl": "bowls",
        "keyboard": "keyboard", "smartphone": "phone", "using tableware": "grabbing utensils",
        "taking off clothes": "undressing", "putting on clothes": "getting dressed",
        "opening the cabinet": "opening cabinet",
    }
    for a,b in repl.items(): x=x.replace(a,b)
    x=" ".join(x.split())
    aliases={
      "jumping jacks":"Jumping jacks", "mopping":"Mopping", "turning page":"Turning a page",
      "turning a page":"Turning a page", "wiping a bowls":"Washing dishes",
      "wiping bowls":"Washing dishes", "wiping a bowl":"Washing dishes",
      "wiping bowl":"Washing dishes", "wiping surface":"Wiping surface",
      "wiping surfaces":"Wiping surface", "standing up":"Standing up",
      "standing on one leg":"Standing up", "using a remote":"Using a remote",
      "opening cabinet":"Opening cabinet", "putting clothes":"Getting dressed",
      "getting dressed":"Getting dressed", "peeling with a knife":"Peeling fruit",
      "making a phone call":"Calling", "brushing teeth":"Brushing teeth",
      "stretching exercises":"Stretching", "doing stretching exercises":"Stretching",
      "doing lunges":"Lunges", "lunges":"Lunges", "drinking":"Drinking",
      "drinking water":"Drinking", "eating":"Eating", "eating ":"Eating",
      "walking":"Walking", "typing on the keyboard":"Typing on a keyboard",
      "using the keyboard":"Typing on a keyboard", "using a smartphone":"Using a phone",
      "taking a selfie":"Taking a selfie", "checking body temperature":"Checking body temperature",
      "taking their body temperature":"Checking body temperature", "taking medicine":"Taking medicine",
      "sweeping":"Sweeping", "sweeping the":"Sweeping", "wiping their hands":"Wiping hands",
      "wiping hands":"Wiping hands", "writing":"Writing", "combing their hair":"Combing hair",
      "combing hair":"Combing hair", "lying down":"Lying down", "sitting down":"Sitting down",
      "throwing away rubbish":"Throwing away rubbish", "taking off their clothes":"Undressing",
    }
    return aliases.get(x, str(s).strip())

def video_path(rel: str) -> Path:
    # HAU rows use their full relative path; test rows use LM_test_* under the public
    # large-model directory.  Prefer Depth_Color because all target HARn clips have it.
    if rel.startswith("large_model_track_test/"):
        rel = rel.split("large_model_track_test/",1)[1]
        base = ROOT / "hf_data_manual/large_model_track_test" / rel.split("/Depth",1)[0]
    else:
        base = ROOT / "hf_data_manual" / rel.split("/Depth",1)[0]
    for p in (base/"Depth_Color/Depth_Color.mp4", base/"Depth/Depth.mp4", base/"Thermal/Thermal.mp4"):
        if p.exists(): return p
    return base/"Depth_Color/Depth_Color.mp4"

def feat(path: Path, n: int=16) -> np.ndarray:
    cap=cv2.VideoCapture(str(path))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total<=0 or not cap.isOpened(): return np.full(16*16*3+80, np.nan, np.float32)
    ids=np.linspace(0,total-1,n).round().astype(int)
    frames=[]
    for i in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,fr=cap.read()
        if not ok: frames.append(np.zeros((16,16),np.float32)); continue
        gray=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        gray=cv2.resize(gray,(16,16),interpolation=cv2.INTER_AREA)
        frames.append(gray)
    cap.release(); x=np.stack(frames)
    # Remove clip-global brightness but retain silhouette/pose and temporal change.
    z=(x-x.mean((1,2),keepdims=True))/(x.std((1,2),keepdims=True)+1e-3)
    d=np.diff(z,axis=0)
    pooled=np.concatenate([z.mean(0).ravel(), z.std(0).ravel(), d.mean(0).ravel(), d.std(0).ravel()])
    # Small explicit temporal/shape summary is more stable than raw pixels.
    flat=np.array([x.mean((1,2)),x.std((1,2)),x.mean(1).mean(1),x.mean(2).mean(1)],np.float32).ravel()
    return np.nan_to_num(np.concatenate([pooled,flat]),nan=0.,posinf=0.,neginf=0.).astype(np.float32)

def action_from_row(r):
    ans=str(r.answer)
    vals=[str(r[L]).strip() for L in "ABCD"]
    if len(ans)!=1 or ans not in "ABCD": return None
    return norm_action(vals[ord(ans)-65])

def main():
    # HAU single rows are the only clip-level visual labels that do not require a
    # multi-action interpretation.  Exclude labels outside the shared vocabulary.
    tr=TR[(TR.source=="HAU") & (TR.category=="single")].copy()
    tr["label"]=tr.apply(action_from_row,axis=1)
    tr=tr[tr.label.isin(CANON)].copy()
    tr["user"]=tr.path.str.extract(r"/(user\d+)/")[0]
    # Cache one compact vector per path, so repeated question rows never reread video.
    cache={}; t0=time.time()
    def get(p):
        if p not in cache: cache[p]=feat(video_path(p))
        return cache[p]
    rows=[]
    for p,g in tr.groupby("path"):
        f=get(p)
        if np.isfinite(f).all(): rows.append((p,g.iloc[0].user,g.iloc[0].label,f))
    print("train clips",len(rows),"dim",rows[0][3].size if rows else 0,"seconds",round(time.time()-t0,1),flush=True)
    X=np.stack([x[3] for x in rows]); y=np.array([x[2] for x in rows]); users=np.array([x[1] for x in rows]); paths=np.array([x[0] for x in rows])
    le=LabelEncoder().fit(sorted(CANON)); yi=le.transform(y)
    # Fixed user-disjoint folds (same pseudotest convention as the championship).
    us=sorted(set(users),key=lambda s:int(s[4:])); folds=[us[i::5] for i in range(5)]
    oof=[]
    for fi,held in enumerate(folds):
        a=~np.isin(users,held); b=~a
        clf=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=.6,class_weight="balanced",random_state=20260908+fi,n_jobs=-1)
        clf.fit(X[a],yi[a]); pr=clf.predict(X[b]); prob=clf.predict_proba(X[b]);
        for p,u,t,g,pp in zip(paths[b],users[b],y[b],le.inverse_transform(pr),prob.max(1)):
            oof.append(dict(fold=fi,path=p,user=u,truth=t,prediction=g,confidence=float(pp),correct=int(t==g)))
    od=pd.DataFrame(oof); od.to_csv(OUT/"raw_video_action_oof.csv",index=False)
    print("OOF",int(od.correct.sum()),"/",len(od),"=",round(float(od.correct.mean()),4),"high",[(q,int((od[od.confidence>=q].correct).sum()),int((od.confidence>=q).sum())) for q in (.6,.7,.8,.9)])
    clf=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,max_features=.6,class_weight="balanced",random_state=20260908,n_jobs=-1).fit(X,yi)
    # All HARn singles in the test have a raw depth video.  Map model action to an
    # option letter only if one option exactly names the predicted action.
    test=TE[(TE.source=="HARn") & (TE.category=="single")].copy(); out=[]
    base=dict(zip(BASE.qa_id,BASE.prediction))
    for _,r in test.iterrows():
        f=get(r.path)
        if not np.isfinite(f).all(): continue
        pr=clf.predict(f[None])[0]; action=le.inverse_transform([pr])[0]
        oo=[norm_action(r[L]) for L in "ABCD"]; cand=["ABCD"[i] for i,a in enumerate(oo) if a==action]
        prob=float(clf.predict_proba(f[None]).max())
        if cand: out.append(dict(qa_id=r.qa_id,base=base[r.qa_id],visual=cand[0],action=action,confidence=prob,options="|".join(oo),changed=int(cand[0]!=base[r.qa_id])))
    td=pd.DataFrame(out); td.to_csv(OUT/"raw_video_action_test.csv",index=False)
    print(td[td.changed==1].to_string(index=False) if len(td) else "no mapped test suggestions")

if __name__=="__main__": main()
