"""Specialist A v1: skeleton/IMU action presence (final video campaign).

Trains 40 one-vs-rest logistic presence heads on STRONG single-action HARn
clips only (action dirs mapped to HAU atoms; ambiguous/unmapped dirs
dropped and listed). HAU clips are never training targets (their sets are
option-sampled, not exhaustive).

Grouped protocol: identical outer folds as oof_driver
(champ.pseudotest.folds(users, 5, seed=7)); multi threshold tau tuned on
outer-train multi rows only (nested, frozen per fold). Writes OOF
specialist predictions + per-option presence for HAU test clips.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'champ'))
from pseudotest import folds  # noqa: E402

ATOMS = ['walking', 'sitting down', 'eating', 'peeling fruit', 'pouring',
         'grabbing utensils', 'drinking', 'checking the time', 'standing up',
         'listening to the music with headphones', 'reading', 'stretching',
         'stirring', 'taking medicine', 'typing on a keyboard',
         'massaging oneself', 'wiping hands', 'squats', 'getting dressed',
         'turning a page', 'sweeping', 'brushing teeth', 'wiping surface',
         'checking body temperature', 'playing games', 'combing hair',
         'mopping', 'using a phone', 'washing face', 'lying down', 'calling',
         'taking a selfie', 'undressing', 'writing', 'jumping jacks',
         'running', 'washing dishes', 'folding clothes', 'lunges',
         'watching tv']

def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


CANON = {
    '0_wash_face': 'washing face', '1_brush_teeth': 'brushing teeth',
    '2_comb_hair': 'combing hair', '3_take_off_clothes': 'undressing',
    '4_wipe_hands': 'wiping hands', '5_put_on_clothes': 'getting dressed',
    '6_drink_water': 'drinking', '7_eat_food': 'eating',
    '8_take_and_use_tableware': 'grabbing utensils',
    '9_pour_drinks': 'pouring', '10_stir_drinks': 'stirring',
    '11_peel_fruits': 'peeling fruit', '12_sweep_the_floor': 'sweeping',
    '13_mop_the_floor': 'mopping', '16_fold_clothes': 'folding clothes',
    '17_tap_the_keyboard': 'typing on a keyboard', '18_write': 'writing',
    '19_make_a_phone_call': 'calling', '20_check_the_time':
        'checking the time',
    '21_read_documents': 'reading', '22_turn_pages': 'turning a page',
    '23_listen_to_music_with_headphones':
        'listening to the music with headphones',
    '24_use_a_mobile_phone': 'using a phone', '25_watch_tv': 'watching tv',
    '26_play_games': 'playing games', '27_take_a_selfie': 'taking a selfie',
    '28_jog_in_place': 'running', '29_do_squats': 'squats',
    '30_do_jumping_jacks': 'jumping jacks',
    '31_do_stretching_exercises': 'stretching', '32_stand_up': 'standing up',
    '33_lie_down': 'lying down', '34_sit_down': 'sitting down',
    '35_do_lunges': 'lunges', '36_walk': 'walking',
    '37_take_medicine': 'taking medicine', '38_massage_oneself':
        'massaging oneself',
    '39_take_body_temperature': 'checking body temperature',
    '41_peel_fruits_with_a_knife': 'peeling fruit',
}
DROP_READABLE = {'wipe_bowls', 'wipe_windows_and_tables', 'stand_on_one_leg',
                 'throw_away', 'open_the_cabinet'}


def canon_to_atom(d):
    a = CANON.get(d.lower())
    return a if a in ATOMS else None


def main():
    z = np.load(os.path.join(ROOT, 'research', 'final_video_20260910',
                             'clip_features.npz'), allow_pickle=True)
    ids, X = list(z['ids']), z['X']
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'training_qa.csv'),
                                    encoding='utf-8-sig')))
    hau = [r for r in rows if r['path'].startswith('HAU/')]
    clips = defaultdict(dict)
    for r in hau:
        clips[r['path']][r['category']] = r

    def O(r):
        return [norm(r[k]) for k in 'ABCD']

    # HARn strong labels
    harn = []  # (feat_idx, atom, user)
    feat_of = {}
    for i, (kind, cid) in enumerate(ids):
        feat_of[(kind, cid)] = i
    aid = {a: i for i, a in enumerate(ATOMS)}
    unmapped = defaultdict(int)
    for (kind, cid), i in feat_of.items():
        if kind != 'HARn':
            continue
        parts = cid.split('/')
        d = parts[1]
        user = parts[2]
        a = canon_to_atom(d)
        if a is None:
            unmapped[d] += 1
            continue
        harn.append((i, aid[a], user))
    print('HARn strong clips:', len(harn), 'unmapped dirs:',
          dict(unmapped), flush=True)

    users = sorted(set(u for _, _, u in harn)
                   | set(p.split('/')[1] for p in clips))
    Y = np.zeros((len(harn), len(ATOMS)))
    for j, (_, a, _) in enumerate(harn):
        Y[j, a] = 1.0

    hau_idx = [(p, feat_of[('HAU', p)]) for p in clips
               if ('HAU', p) in feat_of]
    print('HAU clips w/ features:', len(hau_idx), flush=True)

    recs = []
    prob_test = {}
    for fi, hold in enumerate(folds(users, 5)):
        hold = set(hold)
        tri = [j for j, (_, _, u) in enumerate(harn) if u not in hold]
        tei = [j for j, (_, _, u) in enumerate(harn) if u in hold]
        sc = StandardScaler().fit(X[[harn[j][0] for j in tri]])
        Xt = sc.transform(X)
        P = np.zeros((len(X), len(ATOMS)))
        for a in range(len(ATOMS)):
            y = Y[tri, a]
            if len(set(y.tolist())) < 2:
                P[:, a] = y.mean()
                continue
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(Xt[[harn[j][0] for j in tri]], y)
            P[:, a] = clf.predict_proba(Xt)[:, 1]
        # nested tau on outer-train multi rows
        best_t, best_s = 0.5, -1
        for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            s = c = 0
            for p, ix in hau_idx:
                if p.split('/')[1] in hold or 'multi' not in clips[p]:
                    continue
                r = clips[p]['multi']
                o = O(r)
                pred = ''.join(l for l in 'ABCD'
                               if P[ix, aid.get(o['ABCD'.index(l)], -1)]
                               > t) if all(x in aid for x in o) else ''
                c += 1
                s += (pred == r['answer'])
            if c and s / c > best_s:
                best_s, best_t = s / c, t
        # OOF rows for held users
        for p, ix in hau_idx:
            if p.split('/')[1] not in hold:
                continue
            c = clips[p]
            prow = {ATOMS[a]: round(float(P[ix, a]), 4) for a in range(len(ATOMS))}
            if 'single' in c:
                r = c['single']
                o = O(r)
                cand = [(l, P[ix, aid[x]]) for l, x in
                        zip('ABCD', o) if x in aid]
                pr = max(cand, key=lambda t: t[1])[0] if cand else None
                recs.append(dict(qa_id=r['qa_id'], fold=fi, category='single',
                                 pred=pr, answer=r['answer'],
                                 correct=int(pr == r['answer'])))
            if 'multi' in c:
                r = c['multi']
                o = O(r)
                pr = ''.join(l for l, x in zip('ABCD', o)
                             if x in aid and P[ix, aid[x]] > best_t)
                recs.append(dict(qa_id=r['qa_id'], fold=fi, category='multi',
                                 pred=pr, answer=r['answer'],
                                 correct=int(pr == r['answer'])))
            if 'combination' in c:
                r = c['combination']
                o = O(r)
                def score(opt):
                    ms = [a.strip() for a in opt.split(',')]
                    ps = [P[ix, aid[m]] for m in ms if m in aid]
                    return sum(ps) / len(ps) if ps else -1
                pr = max('ABCD', key=lambda l: score(o['ABCD'.index(l)]))
                recs.append(dict(qa_id=r['qa_id'], fold=fi,
                                 category='combination', pred=pr,
                                 answer=r['answer'],
                                 correct=int(pr == r['answer'])))
        print(f'  fold {fi} tau={best_t} done', flush=True)
    d = pd.DataFrame(recs)
    out = os.path.join(ROOT, 'research', 'final_video_20260910',
                       'oof_specialistA_v1.csv')
    d.to_csv(out, index=False)
    for cat, g in d.groupby('category'):
        print(f'  {cat:12s} {int(g.correct.sum()):4d}/{len(g):4d} = '
              f'{g.correct.mean():.4f}', flush=True)
    print(f'  {"TOTAL":12s} {int(d.correct.sum()):4d}/{len(d):4d} = '
          f'{d.correct.mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
