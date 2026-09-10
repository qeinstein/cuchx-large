"""Extract evenly spaced frames for human/model inspection, without altering source video."""
from pathlib import Path
import cv2, numpy as np, json, sys
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/takeover_20260908/frames'
OUT.mkdir(exist_ok=True)
clips=['0007','0023','0046','0059','0062','0103','0107','0110','0135','0123','0124','0117','0145']
if len(sys.argv)>1:clips=sys.argv[1:]
stats=[]
for clip in clips:
    paths=list((ROOT/'hf_data_manual').glob('**/LM_test_'+clip+'/Depth_Color/*.mp4'))
    if not paths:
        stats.append({'clip':clip,'missing':True});continue
    p=paths[0];cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    if n<=0:continue
    indices=np.linspace(0,n-1,12,dtype=int)
    sheet=Image.new('RGB',(4*320,3*260),'white');draw=ImageDraw.Draw(sheet)
    index_to_j={int(i):j for j,i in enumerate(indices)}
    for i in range(n):
        ok,frame=cap.read()
        if not ok:break
        if i not in index_to_j:continue
        j=index_to_j[i]
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((320,240))
        x=(j%4)*320;y=(j//4)*260;sheet.paste(im,(x,y+20));draw.text((x+3,y+2),f'{clip}: frame {i} / {n}',fill='black')
    cap.release();sheet.save(OUT/(clip+'.jpg'))
    stats.append({'clip':clip,'frames':n,'fps':fps,'path':str(p.relative_to(ROOT))})
(OUT/('metadata_'+clips[0]+'.json')).write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2))
