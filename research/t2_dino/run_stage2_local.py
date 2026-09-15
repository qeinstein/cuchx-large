"""Local (CPU) stage-2 for T2-DINO: identical protocol to kernel_dino_specialist.py
stage 2, with the concat bug fixed (v3/v4 crashed: per-row lists concatenated
along the wrong axis -> (2,nD) instead of (n,2D) -> sklearn sample mismatch).

Consumes downloaded kernel embeddings; writes kernel-format OOF npzs so
oof_eval_cpu.py runs unchanged. Memory-frugal: one modality resident at a time.

Usage:
  python3 research/t2_dino/run_stage2_local.py --dir /tmp/dino_out --out research/t2_dino/local_oof
"""
import argparse
import gc
import json
import os
import sys
import types

import numpy as np
import pandas as pd

ROOT = '/home/fluxx/Workspace/cuchx-large'


def load_kernel_lib():
    """Import kernel stage-2 functions without /kaggle mount resolution."""
    src = open(os.path.join(ROOT, 'research/t2_dino/kernel_dino_specialist.py')).read()
    src = src.replace("_resolve_vid()", "'/tmp/novid'")
    src = src.replace("_resolve_root('meta.csv')", "'/tmp/noinp'")
    mod = types.ModuleType('kspec')
    mod.__name__ = 'kspec'
    exec(compile(src, os.path.join(ROOT, 'research/t2_dino/kernel_dino_specialist.py'), 'exec'),
         mod.__dict__)
    return mod


def load_memmap(d, f):
    z = np.load(os.path.join(d, f), mmap_mode='r', allow_pickle=True)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--attn', action='store_true', help='also run torch attn head (slow on CPU)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    K = load_kernel_lib()

    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    tr = pd.read_csv(os.path.join(ROOT, 'training_qa.csv'))
    nest = K.build_nest(meta)
    classes = sorted(meta[meta.kind == 'train_harn'].action.unique())
    oof_store, report = {}, {}

    for mod, fn, zname in (('depth', 'dino_s_depth.npz', 'depth'),
                           ('thermal', 'dino_s_thermal.npz', 'thermal')):
        p = os.path.join(a.dir, fn)
        if not os.path.exists(p) or os.path.getsize(p) < 1000:
            print('%s: MISSING/EMPTY, skipped' % fn, flush=True)
            continue
        Z = load_memmap(a.dir, fn)
        print('%s: %d keys' % (fn, len(Z)), flush=True)
        descs = {}
        for stream, sname in (('full', zname + '_full'), ('crop1.6', zname + '_crop')):
            X, y, u, paths = K.harn_descriptors(Z, meta, nest, stream)
            if X is None:
                print('  %s: no descriptors' % sname, flush=True)
                continue
            descs[sname] = (X, y, u, paths)
            store, acc, (c, n), table = K.oof_logreg(X, y, u, paths, classes)
            for k, v in store.items():
                oof_store['%s|%s' % (sname, k)] = v
            report[sname] = {'top1': round(acc, 4), 'correct': c, 'n': n,
                             'per_action': table}
            print('  LogReg %-12s top-1 %d/%d = %.4f' % (sname, c, n, acc), flush=True)
        # FIXED concat: stack per-clip feature-concats -> (n, 2D)
        fa, fb = zname + '_full', zname + '_crop'
        if fa in descs and fb in descs:
            Xa, ya, ua, pa = descs[fa]
            Xb, yb, ub, pb = descs[fb]
            common = sorted(set(pa) & set(pb))
            ia = {p_: i for i, p_ in enumerate(pa)}
            ib = {p_: i for i, p_ in enumerate(pb)}
            Xc = np.stack([np.concatenate([Xa[ia[p_]], Xb[ib[p_]]]) for p_ in common])
            assert Xc.shape == (len(common), Xa.shape[1] + Xb.shape[1]), Xc.shape
            yc = np.array([ya[ia[p_]] for p_ in common])
            uc = np.array([ua[ia[p_]] for p_ in common])
            store, acc, (c, n), table = K.oof_logreg(Xc, yc, uc, np.array(common), classes)
            for k, v in store.items():
                oof_store['%s_concat|%s' % (zname, k)] = v
            report[zname + '_concat'] = {'top1': round(acc, 4), 'correct': c, 'n': n,
                                        'per_action': table}
            print('  LogReg %-12s top-1 %d/%d = %.4f' % (zname + '_concat', c, n, acc), flush=True)
        if a.attn and descs:
            best = max(descs, key=lambda k: report[k]['top1'])
            stream = 'full' if best.endswith('_full') else 'crop1.6'
            mi = meta.set_index('qa_path')
            seqs, y, u, paths = [], [], [], []
            for r in meta[meta.kind == 'train_harn'].itertuples():
                par = nest.get(r.qa_path)
                key = (par + '|' + stream) if par else None
                if not key or key not in Z:
                    continue
                A = np.asarray(Z[key], np.float32)
                cf0, cf1, pf0 = (mi.loc[r.qa_path, 'f0'], mi.loc[r.qa_path, 'f1'],
                                 mi.loc[par, 'f0'])
                if not (np.isfinite(cf0) and np.isfinite(pf0)):
                    continue
                i0 = int(max(0, cf0 - pf0))
                i1 = int(min(len(A), cf1 - pf0 + 1))
                if i1 - i0 < 2:
                    continue
                seqs.append(A[i0:i1])
                y.append(r.action)
                u.append(r.user)
                paths.append(r.qa_path)
            if seqs:
                store, acc, (c, n) = K.attn_head_oof(seqs, np.array(y), np.array(u),
                                                    np.array(paths), classes, 'cpu')
                for k, v in store.items():
                    oof_store['attn_%s|%s' % (best, k)] = v
                report['attn_' + best] = {'top1': round(acc, 4), 'correct': c, 'n': n}
                print('  Attn   %-12s top-1 %d/%d = %.4f' % (best, c, n, acc), flush=True)
        del Z
        gc.collect()

    np.savez_compressed(os.path.join(a.out, 'oof_harn_action.npz'),
                        classes=np.array(classes), **oof_store)
    json.dump(report, open(os.path.join(a.out, 'harn_action_report.json'), 'w'), indent=1)
    print('saved oof_harn_action.npz (%d keys) + report' % len(oof_store), flush=True)

    # HAU pool heads (reload one modality at a time)
    voc = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))
    hau2idx = {v: i for i, v in enumerate(sorted(set(voc['HARN2HAU'].values())))}
    actions = sorted(hau2idx)
    clipsets = K.hau_clip_action_sets(tr)
    pool_store, pool_report = {}, {}
    for mod, fn in (('depth', 'dino_s_depth.npz'), ('thermal', 'dino_s_thermal.npz')):
        p = os.path.join(a.dir, fn)
        if not os.path.exists(p) or os.path.getsize(p) < 1000:
            continue
        Z = load_memmap(a.dir, fn)
        rows, Y, u, paths = [], [], [], []
        for r in meta[meta.kind == 'train_hau'].itertuples():
            kf, kc = r.qa_path + '|full', r.qa_path + '|crop1.6'
            if kf not in Z or kc not in Z:
                continue
            d = np.concatenate([K.pool(np.asarray(Z[kf], np.float32)),
                                K.pool(np.asarray(Z[kc], np.float32))])
            s = clipsets.get(r.qa_path, set())
            rows.append(d)
            Y.append([1 if x in s else 0 for x in actions])
            u.append(r.user)
            paths.append(r.qa_path)
        if rows:
            P, rep = K.oof_multilabel(np.stack(rows), np.array(Y), np.array(u),
                                      np.array(paths), actions)
            pool_store[mod + '|P'] = P
            pool_store[mod + '|Y'] = np.array(Y)
            pool_store[mod + '|paths'] = np.array(paths)
            pool_report[mod] = {k: v for k, v in rep.items() if k != 'per_action_AP'}
            pool_report[mod]['per_action_AP'] = rep['per_action_AP']
            print('pool %-7s macro-mAP %.4f micro-AP %.4f'
                  % (mod, rep['macro_mAP'], rep['micro_AP']), flush=True)
        del Z
        gc.collect()
    if pool_store:
        np.savez_compressed(os.path.join(a.out, 'oof_hau_pool.npz'),
                            actions=np.array(actions), **pool_store)
    print('STAGE2-LOCAL done', flush=True)


if __name__ == '__main__':
    main()
