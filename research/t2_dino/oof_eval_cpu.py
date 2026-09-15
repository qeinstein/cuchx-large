"""CPU-runnable OOF evaluation for downloaded T2-DINO kernel outputs.

Runs locally (numpy/pandas/sklearn only). Reproduces, from the kernel's saved
OOF log-probs (no re-training):
  1. HARn action-identity top-1 + per-action table per DINO stream.
  2. Late-fusion sweep DINO + skeleton/IMU (harn_clf.npz OOF) on same clips,
     including the end-to-end HARn-single-question accuracy (the 409/429 ->
     417/429 comparison) with a split-half nested weight check.
  3. HAU clip action-set presence mAP tables.
  4. Embedding coverage/dim sanity + manifest alignment audit.

Usage:
  python3 research/t2_dino/oof_eval_cpu.py --dir /tmp/dino_out
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = '/home/fluxx/Workspace/cuchx-large'


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def report_harn(oof, meta):
    classes = [str(x) for x in oof.pop('classes')]
    truth = dict(zip(meta.qa_path, meta.action))
    streams = sorted({k.split('|oof|')[0] for k in oof if '|oof|' in k})
    print('== 1. HARn action identity (subject-disjoint OOF) ==')
    for s in streams:
        ks = [k for k in oof if k.startswith(s + '|oof|')]
        c = n = 0
        per = {cl: [0, 0] for cl in classes}
        for k in ks:
            p = k.split('|oof|', 1)[1]
            t = truth.get(p)
            if t is None:
                continue
            pred = classes[int(np.argmax(np.asarray(oof[k], np.float32)))]
            c += int(pred == t)
            n += 1
            per[t][0] += int(pred == t)
            per[t][1] += 1
        print('  %-14s top-1 %d/%d = %.4f' % (s, c, n, c / max(n, 1)))
        worst = sorted(per.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))
        print('    worst-5: ' + ', '.join(
            '%s %d/%d' % (a, v[0], v[1]) for a, v in worst[:5]))
        best = sorted(per.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1))
        print('    best-5:  ' + ', '.join(
            '%s %d/%d' % (a, v[0], v[1]) for a, v in best[:5]))
    return streams, classes


def report_fusion(oof, streams, classes, meta, tr):
    hz = load_npz(os.path.join(ROOT, 'champ', 'harn_clf.npz'))
    hcls = [str(x) for x in hz.pop('classes')]
    hlog = {k[4:]: np.asarray(v, np.float32) for k, v in hz.items()
            if k.startswith('oof|')}
    di = [classes.index(c) for c in hcls]
    voc = json.load(open(os.path.join(ROOT, 'champ', 'vocab.json')))
    s2a = {v: k for k, v in voc['HARN2SELF'].items()}
    single = tr[(tr.source == 'HARn') & (tr.category == 'single')]
    print('== 2. Fusion DINO + skeleton/IMU (same clips) ==')
    for s in streams:
        pref = s + '|oof|'
        dlog = {k[len(pref):]: np.asarray(oof[k], np.float32)
                for k in oof if k.startswith(pref)}
        common = [p for p in dlog if p in hlog]
        if not common:
            continue
        users = sorted({p.split('/')[2] for p in common})
        half = {u: i % 2 for i, u in enumerate(users)}

        def acc_at(paths, w):
            c = sum(hcls[int(np.argmax((1 - w) * hlog[p] + w * dlog[p][di]))]
                        == p.split('/')[1] for p in paths)
            return c, len(paths)

        def single_at(paths, w):
            pset = set(paths)
            rows = [r for r in single.itertuples() if r.path in pset]
            c = 0
            for r in rows:
                lp = (1 - w) * hlog[r.path] + w * dlog[r.path][di]
                pred_a = hcls[int(np.argmax(lp))]
                oo = [str(getattr(r, L)).strip() for L in 'ABCD']
                acts = [s2a.get(o) for o in oo]
                try:
                    guess = 'ABCD'[acts.index(pred_a)]
                except ValueError:
                    guess = None
                c += int(guess == str(r.answer))
            return c, len(rows)

        print('  stream %s (%d clips, %d single-Q)' % (s, len(common),
              sum(1 for r in single.itertuples() if r.path in set(common))))
        for w in [0, .1, .2, .3, .4, .5, .6, .8, 1.0]:
            c, n = acc_at(common, w)
            cs, ns = single_at(common, w)
            print('    w=%.1f clip-top1 %d/%d=%.4f  single-Q %d/%d=%.4f'
                  % (w, c, n, c / n, cs, ns, cs / max(ns, 1)))
        # split-half nested weight check (pick w on even users, eval on odd)
        grid = [round(x * .1, 1) for x in range(11)]
        for pick_h, eval_h in ((0, 1), (1, 0)):
            pk = [p for p in common if half[p.split('/')[2]] == pick_h]
            ev = [p for p in common if half[p.split('/')[2]] == eval_h]
            bw = max(grid, key=lambda w: acc_at(pk, w)[0])
            c, n = acc_at(ev, bw)
            c0, _ = acc_at(ev, 0.0)
            print('    nested pick-half%d w=%.1f -> eval-half%d %d/%d=%.4f '
                  '(w=0: %d/%d)' % (pick_h, bw, eval_h, c, n, c / n, c0, n))


def report_pool(d, out):
    z = load_npz(os.path.join(d, out))
    actions = [str(x) for x in z.pop('actions')]
    from sklearn.metrics import average_precision_score
    print('== 3. HAU clip action-set presence (subject-disjoint OOF) ==')
    for mod in ('depth', 'thermal'):
        if mod + '|P' not in z:
            print('  %s: missing' % mod)
            continue
        P, Y = z[mod + '|P'], z[mod + '|Y']
        aps = [average_precision_score(Y[:, j], P[:, j])
               for j in range(len(actions))]
        print('  %-7s macro-mAP %.4f micro-AP %.4f over %d clips x %d actions'
              % (mod, float(np.mean(aps)),
                 float(average_precision_score(Y.ravel(), P.ravel())),
                 Y.shape[0], Y.shape[1]))
        order = np.argsort(aps)
        print('    worst-5 AP: ' + ', '.join(
            '%s %.3f' % (actions[j], aps[j]) for j in order[:5]))
        print('    best-5 AP:  ' + ', '.join(
            '%s %.3f' % (actions[j], aps[j]) for j in order[-5:][::-1]))


def report_coverage(d, meta):
    print('== 4. Embedding coverage / sanity ==')
    for f in ('dino_s_depth.npz', 'dino_s_thermal.npz', 'dino_b_depth.npz'):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            print('  %s: MISSING' % f)
            continue
        z = load_npz(p)
        keys = [k for k in z if not k.endswith('|mod')
                and not k.endswith('|boxsrc')]
        full = [k for k in keys if k.endswith('|full')]
        crop = [k for k in keys if k.endswith('|crop1.6')]
        dims = {np.asarray(z[k]).shape[1] for k in keys} if keys else set()
        nfr = sum(len(np.asarray(z[k])) for k in keys)
        print('  %s: %d full + %d crop keys, dim=%s, %.1fk frames, %.0f MB'
              % (f, len(full), len(crop), dims, nfr / 1e3,
                 os.path.getsize(p) / 1e6))
    mp = os.path.join(d, 'manifest.json')
    if os.path.exists(mp):
        m = json.load(open(mp))
        print('  manifest: device=%s total=%.1fmin backbones=%s' % (
            m.get('device'), m.get('timings', {}).get('total_min'),
            {k: v.get('ok', v) for k, v in m.get('backbones', {}).items()}))
        print('  alignment audit: %s' % m.get('alignment'))
        if m.get('fallbacks'):
            print('  fallbacks: %s' % m['fallbacks'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='downloaded kernel outputs')
    a = ap.parse_args()
    meta = pd.read_csv(os.path.join(ROOT, 'champ', 'meta.csv'))
    tr = pd.read_csv(os.path.join(ROOT, 'training_qa.csv'))
    report_coverage(a.dir, meta)
    oof = load_npz(os.path.join(a.dir, 'oof_harn_action.npz'))
    streams, classes = report_harn(oof, meta)
    report_fusion(oof, streams, classes, meta, tr)
    report_pool(a.dir, 'oof_hau_pool.npz')


if __name__ == '__main__':
    main()
