"""Verify local depth videos against official ZIP directory CRCs without bulk download."""
from pathlib import Path
import json,binascii
from huggingface_hub import get_token
import requests
from remotezip import RemoteZip
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/post332_20260908'
URL='https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Large_Model_Track/resolve/main/Large-Model-Track/Testing/data/large_model_track_test.zip'
def main():
    session=requests.Session();token=get_token()
    if token:session.headers['Authorization']='Bearer '+token
    with RemoteZip(URL,session=session,timeout=60) as z:
        members=z.infolist();lookup={x.filename:x for x in members}
        result=[]
        for p in sorted((ROOT/'hf_data_manual/large_model_track_test').glob('*/Depth_Color/*.mp4')):
            suffix=str(p.relative_to(ROOT/'hf_data_manual'))
            matches=[v for k,v in lookup.items() if k.endswith(suffix)]
            if len(matches)!=1:
                result.append({'file':suffix,'status':'member_not_unique','matches':len(matches)});continue
            info=matches[0];data=p.read_bytes();crc=binascii.crc32(data)&0xffffffff
            result.append({'file':suffix,'status':'match' if len(data)==info.file_size and crc==info.CRC else 'MISMATCH','local_size':len(data),'archive_size':info.file_size,'local_crc':crc,'archive_crc':info.CRC})
        (OUT/'source_crc_audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print('ZIP members',len(members),'local depth videos',len(result),'matches',sum(r['status']=='match' for r in result),flush=True)
        print(json.dumps([r for r in result if r['status']!='match'],indent=2))
if __name__=='__main__':main()
