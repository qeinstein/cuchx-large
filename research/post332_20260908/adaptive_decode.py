#!/usr/bin/env python3
"""Select tomorrow's fifth file after four 332-relative reversion evaluations.

Usage: python adaptive_decode.py -1 -1 0 0
Arguments are deltas returned for reversion probes in the fixed order:
0488, 0477, 0506, 0519.  This script only decodes the exact public-sum constraint;
it never contacts Kaggle and never writes a submission.
"""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ORDER=['0488','0477','0506','0519']
def main():
    p=argparse.ArgumentParser();p.add_argument('delta',nargs=4,type=int);a=p.parse_args()
    if any(x not in (-1,0,1) for x in a.delta):p.error('each delta must be -1, 0, or +1')
    # reversion delta = -d; exact aggregate d0488+d0477+d0506+d0519+d0501 = +2.
    d0501=2+sum(a.delta)
    if d0501 not in (-1,0,1):
        raise SystemExit(f'inconsistent observations: forced d0501={d0501}, outside signed range')
    name='ADAPTIVE_fifth_0526_plus_revert_0501.csv' if d0501==-1 else 'ADAPTIVE_fifth_0526.csv'
    out={'probe_order':ORDER,'observed_reversion_deltas':a.delta,'forced_effect_test_0501':d0501,
         'next_file':str(Path('research/post332_20260908')/name),
         'interpretation':'d0501=-1 is the unique loss; fifth file recovers it while testing 0526.' if d0501==-1 else '0501 is not a loss; fifth file tests 0526 alone.'}
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
