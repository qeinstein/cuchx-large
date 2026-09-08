"""Extract labeled temporal windows around candidate action boundaries."""
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/post332_20260908'
def sheet(clip,indices,name):
    p=ROOT/f'hf_data_manual/large_model_track_test/LM_test_{clip}/Depth_Color/Depth_Color.mp4'
    c=cv2.VideoCapture(str(p));wanted={i:j for j,i in enumerate(indices)}
    im=Image.new('RGB',(1280,260*((len(indices)+3)//4)),'white');d=ImageDraw.Draw(im)
    for i in range(max(indices)+1):
        ok,f=c.read()
        if not ok:break
        if i not in wanted:continue
        j=wanted[i];x=j%4*320;y=j//4*260
        frame=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));frame.thumbnail((320,240))
        im.paste(frame,(x,y+20));d.text((x+3,y+2),f'{clip} frame {i}',fill='black')
    c.release();im.save(OUT/(name+'.jpg'))
def crops(clip,indices,name):
    p=ROOT/f'hf_data_manual/large_model_track_test/LM_test_{clip}/Depth_Color/Depth_Color.mp4'
    c=cv2.VideoCapture(str(p)); ims=[]
    for i in range(max(indices)+1):
        ok,f=c.read()
        if not ok: break
        if i in indices:
            ims.append(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).crop((180,0,640,480)))
    c.release(); out=Image.new('RGB',(460*len(ims),480),'white')
    for j,x in enumerate(ims): out.paste(x,(460*j,0))
    out.save(OUT/(name+'.jpg'))
sheet('0198',list(range(46,111,4)),'0483_parent_window')
sheet('0017',list(range(16)),'0483_all_frames')
sheet('0197',list(range(136,201,4)),'0526_parent_window')
sheet('0016',list(range(17)),'0526_all_frames')
sheet('0008',list(range(0,80,4)),'0478_all_frames')
sheet('0047',list(range(0,80,4)),'0507_all_frames')
sheet('0027',list(range(0,80,4)),'0530_all_frames')
sheet('0054',list(range(0,80,4)),'0538_all_frames')
sheet('0056',list(range(0,80,4)),'0540_all_frames')
for _clip,_name in [('0050','0510'),('0051','0511'),('0052','0512'),('0053','0513'),('0055','0514'),('0059','0517'),('0062','0519')]:
    sheet(_clip,list(range(0,80,4)),_name+'_all_frames')
sheet('0023',list(range(0,80,4)),'0488_all_frames')
sheet('0007',list(range(0,80,4)),'0477_all_frames')
sheet('0046',list(range(0,80,4)),'0506_all_frames')
sheet('0032',list(range(0,80,4)),'0533_all_frames')
sheet('0019',list(range(0,80,4)),'0527_all_frames')
sheet('0041',list(range(0,80,4)),'0501_all_frames')
crops('0008',[0,5,10,15,19],'0478_crops')
crops('0047',[0,5,10,15,20,22],'0507_crops')
