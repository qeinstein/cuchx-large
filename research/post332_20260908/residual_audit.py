"""Compact post-332 decision audit.

This report intentionally uses only saved subject-disjoint route audits.  It does not
pretend that a route's standalone accuracy is a flip precision unless both the old and
new decisions were scored on the same OOF rows.
"""
import os, json
import pandas as pd

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=os.path.join(ROOT,'research/post332_20260908')

def wr_rw(d, old='fallback_correct', new='template_correct'):
    c=d[d[old].notna() & d[new].notna() & (d[old]!=d[new])]
    wr=int(((c[old]==0)&(c[new]==1)).sum()); rw=int(((c[old]==1)&(c[new]==0)).sum())
    return dict(n=int(len(c)),wr=wr,rw=rw,net=wr-rw,precision=wr/max(1,wr+rw))

def main():
    out={}
    f=os.path.join(ROOT,'research/exact_visible_object_fallback_oof_20260905.csv')
    d=pd.read_csv(f); out['object_exact_template']={}
    for s in [1,2,3,4,5]:
        g=d[d.support>=s]; z=wr_rw(g); z.update(rows=int(len(g)),fallback_correct=int(g.fallback_correct.sum()),template_correct=int(g.template_correct.sum())); out['object_exact_template'][str(s)]=z
    f=os.path.join(ROOT,'research/harn_template_sensor_gate_oof_20260905.csv')
    d=pd.read_csv(f); q=d[d.disagree==1].copy(); q['old']=q.sensor_correct; q['new']=q.template_correct
    out['harn_template_vs_sensor']=wr_rw(q,'old','new'); out['harn_template_vs_sensor']['rows']=int(len(q))
    # Existing independent routes with explicit old/new decisions.  The top-k table is
    # sorted by the route's own margin/support where available, and is a diagnostic, not
    # a claim that these routes should be submitted.
    routes=[]
    for f,old,new,score,desc in [
      ('research/recovered_pool_sequence_transfer_oof_20260905.csv','base_correct','new_correct','margin','recovered_pool_sequence'),
      ('research/exact_child_sequence_oof_20260906.csv','base_correct','new_correct','coverage','exact_child_sequence'),
      ('research/frame_forest_sequence_oof_20260906.csv','base_correct','forest_correct','separation','frame_forest_sequence'),
      ('research/pool_full_subblocks_paired_oof_20260905.csv','correct_base','correct_corrected',None,'paired_pool')]:
        d=pd.read_csv(os.path.join(ROOT,f)); d=d[d.changed.astype(bool)].copy();
        if not len(d): continue
        if score is not None: d=d.sort_values(score,ascending=False)
        for k in [1,3,5,10,20]:
            z=d.head(k); wr=int(((z[old]==0)&(z[new]==1)).sum()); rw=int(((z[old]==1)&(z[new]==0)).sum())
            routes.append(dict(route=desc,k=k,rows=len(z),wr=wr,rw=rw,net=wr-rw,precision=wr/max(1,wr+rw)))
    out['negative_or_mixed_routes']=routes
    json.dump(out,open(os.path.join(OUT,'residual_audit.json'),'w'),indent=2)
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
