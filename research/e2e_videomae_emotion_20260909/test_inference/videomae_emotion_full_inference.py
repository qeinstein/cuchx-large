"""Full-data training and test candidate inference for validated VideoMAE Emotion model.

Trained on all 270 valid HAU training sessions for 5 epochs with task-aligned structured
assignment loss, manner group loss, and ordinal speed ranking.
Inference is executed on the 29 complete 3-clip test sessions under the exact frozen gate:
- Complete session triples only
- Assignment margin >= 0.50
- Prediction != champion prediction
Only gated flips are applied to the 332 champion.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

gpu_name = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True,
    text=True,
    check=False,
).stdout.strip()
if "P100" in gpu_name:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "torch==2.5.1", "torchvision==0.20.1",
    ])

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.48.0,<5", "accelerate>=1.2.0", "opencv-python-headless",
])

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

SEED = 20260909
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
N_FRAMES = 16
EPOCHS = 5
LR = 2.0e-5
HEAD_LR = 2.0e-4
GRAD_ACCUM = 4
PAIR_AUGMENT_PROB = 0.45
UNFREEZE_BLOCKS = 2
GATE_MARGIN = 0.50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working")

FAST = {
    "Quickly", "quickly", "Rapidly", "Hastily", "Hasitly", "Hurriedly", "Swiftly",
    "Briskly", "Urgently", "Frantically", "Impatiently", "Eagerly", "Forcefully",
}
SLOW = {
    "Slowly", "Leisurely", "Unhurriedly", "Calmly", "Peacefully", "Relaxedly", "Lazily",
    "Gently", "Softly", "Quietly", "Comfortably", "Soothingly", "Patiently", "Lightly",
    "Contently", "Absentmindedly",
}
CARE = {
    "Carefully", "Cautiously", "Meticulously", "Precisely", "Thoroughly", "Deliberately",
    "Methodically", "Attentively", "Intently", "Diligently", "Earnestly", "Neatly",
    "Orderly", "Seriously", "Serioiusly",
}
NERV = {"Nervously", "Anxiously", "Tensely", "Tensly", "Restlessly"}
GROUPS = ["SLOW", "CARE", "NEUT", "NERV", "FAST"]
G2I = {value: index for index, value in enumerate(GROUPS)}

TEST_TRIPLES = [[["LM_test_0065", "test_0650"], ["LM_test_0066", "test_0360"], ["LM_test_0067", "test_0361"]], [["LM_test_0068", "test_0651"], ["LM_test_0069", "test_0362"], ["LM_test_0070", "test_0363"]], [["LM_test_0071", "test_0652"], ["LM_test_0072", "test_0364"], ["LM_test_0073", "test_0365"]], [["LM_test_0074", "test_0653"], ["LM_test_0075", "test_0366"], ["LM_test_0076", "test_0367"]], [["LM_test_0077", "test_0654"], ["LM_test_0078", "test_0368"], ["LM_test_0079", "test_0369"]], [["LM_test_0080", "test_0655"], ["LM_test_0081", "test_0370"], ["LM_test_0082", "test_0371"]], [["LM_test_0083", "test_0656"], ["LM_test_0084", "test_0372"], ["LM_test_0085", "test_0373"]], [["LM_test_0086", "test_0657"], ["LM_test_0087", "test_0374"], ["LM_test_0088", "test_0375"]], [["LM_test_0089", "test_0658"], ["LM_test_0090", "test_0376"], ["LM_test_0091", "test_0377"]], [["LM_test_0092", "test_0659"], ["LM_test_0093", "test_0378"], ["LM_test_0094", "test_0379"]], [["LM_test_0095", "test_0660"], ["LM_test_0096", "test_0380"], ["LM_test_0097", "test_0381"]], [["LM_test_0098", "test_0661"], ["LM_test_0099", "test_0382"], ["LM_test_0100", "test_0383"]], [["LM_test_0107", "test_0664"], ["LM_test_0108", "test_0388"], ["LM_test_0109", "test_0389"]], [["LM_test_0110", "test_0665"], ["LM_test_0111", "test_0390"], ["LM_test_0112", "test_0391"]], [["LM_test_0116", "test_0667"], ["LM_test_0117", "test_0394"], ["LM_test_0118", "test_0395"]], [["LM_test_0122", "test_0669"], ["LM_test_0123", "test_0398"], ["LM_test_0124", "test_0399"]], [["LM_test_0125", "test_0670"], ["LM_test_0126", "test_0400"], ["LM_test_0127", "test_0401"]], [["LM_test_0128", "test_0671"], ["LM_test_0129", "test_0402"], ["LM_test_0130", "test_0403"]], [["LM_test_0131", "test_0672"], ["LM_test_0132", "test_0404"], ["LM_test_0133", "test_0405"]], [["LM_test_0134", "test_0673"], ["LM_test_0135", "test_0406"], ["LM_test_0136", "test_0407"]], [["LM_test_0137", "test_0674"], ["LM_test_0138", "test_0408"], ["LM_test_0139", "test_0409"]], [["LM_test_0140", "test_0675"], ["LM_test_0141", "test_0410"], ["LM_test_0142", "test_0411"]], [["LM_test_0143", "test_0676"], ["LM_test_0144", "test_0412"], ["LM_test_0145", "test_0413"]], [["LM_test_0146", "test_0677"], ["LM_test_0147", "test_0414"], ["LM_test_0148", "test_0415"]], [["LM_test_0149", "test_0678"], ["LM_test_0150", "test_0416"], ["LM_test_0151", "test_0417"]], [["LM_test_0152", "test_0679"], ["LM_test_0153", "test_0418"], ["LM_test_0154", "test_0419"]], [["LM_test_0155", "test_0680"], ["LM_test_0156", "test_0420"], ["LM_test_0157", "test_0421"]], [["LM_test_0158", "test_0681"], ["LM_test_0159", "test_0422"], ["LM_test_0160", "test_0423"]], [["LM_test_0161", "test_0682"], ["LM_test_0162", "test_0424"], ["LM_test_0163", "test_0425"]]]
CHAMPION_DICT = {"test_0001": "B", "test_0002": "B", "test_0003": "D", "test_0004": "B", "test_0005": "C", "test_0006": "D", "test_0007": "B", "test_0008": "D", "test_0009": "C", "test_0010": "D", "test_0011": "C", "test_0012": "D", "test_0013": "A", "test_0014": "A", "test_0015": "C", "test_0016": "B", "test_0017": "A", "test_0018": "C", "test_0019": "A", "test_0020": "A", "test_0021": "A", "test_0022": "A", "test_0023": "D", "test_0024": "D", "test_0025": "A", "test_0026": "B", "test_0027": "A", "test_0028": "C", "test_0029": "C", "test_0030": "A", "test_0031": "B", "test_0032": "D", "test_0033": "C", "test_0034": "D", "test_0035": "A", "test_0036": "B", "test_0037": "C", "test_0038": "C", "test_0039": "C", "test_0040": "C", "test_0041": "A", "test_0042": "D", "test_0043": "B", "test_0044": "C", "test_0045": "B", "test_0046": "D", "test_0047": "C", "test_0048": "D", "test_0049": "B", "test_0050": "B", "test_0051": "D", "test_0052": "B", "test_0053": "C", "test_0054": "D", "test_0055": "B", "test_0056": "D", "test_0057": "C", "test_0058": "B", "test_0059": "A", "test_0060": "B", "test_0061": "C", "test_0062": "A", "test_0063": "A", "test_0064": "D", "test_0065": "C", "test_0066": "C", "test_0067": "B", "test_0068": "C", "test_0069": "C", "test_0070": "A", "test_0071": "B", "test_0072": "A", "test_0073": "D", "test_0074": "C", "test_0075": "D", "test_0076": "B", "test_0077": "C", "test_0078": "C", "test_0079": "C", "test_0080": "B", "test_0081": "C", "test_0082": "D", "test_0083": "A", "test_0084": "A", "test_0085": "B", "test_0086": "B", "test_0087": "B", "test_0088": "D", "test_0089": "D", "test_0090": "A", "test_0091": "C", "test_0092": "A", "test_0093": "B", "test_0094": "D", "test_0095": "A", "test_0096": "A", "test_0097": "A", "test_0098": "C", "test_0099": "C", "test_0100": "C", "test_0101": "D", "test_0102": "B", "test_0103": "B", "test_0104": "D", "test_0105": "D", "test_0106": "C", "test_0107": "A", "test_0108": "A", "test_0109": "C", "test_0110": "B", "test_0111": "C", "test_0112": "A", "test_0113": "A", "test_0114": "AB", "test_0115": "AB", "test_0116": "BD", "test_0117": "BD", "test_0118": "CD", "test_0119": "BD", "test_0120": "B", "test_0121": "A", "test_0122": "D", "test_0123": "C", "test_0124": "BC", "test_0125": "CD", "test_0126": "ABC", "test_0127": "BD", "test_0128": "AD", "test_0129": "BC", "test_0130": "BD", "test_0131": "A", "test_0132": "A", "test_0133": "BC", "test_0134": "BD", "test_0135": "ACD", "test_0136": "B", "test_0137": "AC", "test_0138": "A", "test_0139": "CD", "test_0140": "BCD", "test_0141": "ACD", "test_0142": "ABC", "test_0143": "ABC", "test_0144": "AB", "test_0145": "ABC", "test_0146": "BCD", "test_0147": "BCD", "test_0148": "CD", "test_0149": "D", "test_0150": "ABD", "test_0151": "ABC", "test_0152": "AC", "test_0153": "A", "test_0154": "AC", "test_0155": "AD", "test_0156": "AC", "test_0157": "D", "test_0158": "BC", "test_0159": "ABC", "test_0160": "D", "test_0161": "B", "test_0162": "BCD", "test_0163": "ACD", "test_0164": "BC", "test_0165": "CD", "test_0166": "AB", "test_0167": "AB", "test_0168": "D", "test_0169": "B", "test_0170": "ABC", "test_0171": "ABC", "test_0172": "ACD", "test_0173": "AD", "test_0174": "AD", "test_0175": "A", "test_0176": "D", "test_0177": "AD", "test_0178": "D", "test_0179": "C", "test_0180": "ABC", "test_0181": "AC", "test_0182": "B", "test_0183": "ABD", "test_0184": "D", "test_0185": "ACD", "test_0186": "ABC", "test_0187": "BCD", "test_0188": "BC", "test_0189": "A", "test_0190": "A", "test_0191": "C", "test_0192": "AD", "test_0193": "B", "test_0194": "AC", "test_0195": "B", "test_0196": "B", "test_0197": "ABD", "test_0198": "C", "test_0199": "D", "test_0200": "C", "test_0201": "CD", "test_0202": "ACD", "test_0203": "AD", "test_0204": "AC", "test_0205": "C", "test_0206": "CD", "test_0207": "C", "test_0208": "AC", "test_0209": "AC", "test_0210": "D", "test_0211": "BCD", "test_0212": "AC", "test_0213": "C", "test_0214": "ACD", "test_0215": "D", "test_0216": "A", "test_0217": "BCD", "test_0218": "B", "test_0219": "A", "test_0220": "CD", "test_0221": "A", "test_0222": "AC", "test_0223": "B", "test_0224": "C", "test_0225": "C", "test_0226": "A", "test_0227": "B", "test_0228": "C", "test_0229": "D", "test_0230": "C", "test_0231": "B", "test_0232": "C", "test_0233": "A", "test_0234": "D", "test_0235": "B", "test_0236": "B", "test_0237": "C", "test_0238": "C", "test_0239": "B", "test_0240": "D", "test_0241": "B", "test_0242": "A", "test_0243": "B", "test_0244": "D", "test_0245": "A", "test_0246": "D", "test_0247": "C", "test_0248": "D", "test_0249": "C", "test_0250": "D", "test_0251": "D", "test_0252": "D", "test_0253": "C", "test_0254": "B", "test_0255": "C", "test_0256": "B", "test_0257": "A", "test_0258": "A", "test_0259": "D", "test_0260": "C", "test_0261": "A", "test_0262": "D", "test_0263": "D", "test_0264": "D", "test_0265": "D", "test_0266": "B", "test_0267": "A", "test_0268": "B", "test_0269": "D", "test_0270": "C", "test_0271": "D", "test_0272": "D", "test_0273": "A", "test_0274": "B", "test_0275": "B", "test_0276": "B", "test_0277": "C", "test_0278": "D", "test_0279": "C", "test_0280": "A", "test_0281": "B", "test_0282": "D", "test_0283": "A", "test_0284": "B", "test_0285": "A", "test_0286": "D", "test_0287": "A", "test_0288": "B", "test_0289": "A", "test_0290": "A", "test_0291": "D", "test_0292": "A", "test_0293": "B", "test_0294": "D", "test_0295": "C", "test_0296": "D", "test_0297": "D", "test_0298": "B", "test_0299": "C", "test_0300": "B", "test_0301": "A", "test_0302": "B", "test_0303": "A", "test_0304": "A", "test_0305": "B", "test_0306": "B", "test_0307": "A", "test_0308": "A", "test_0309": "B", "test_0310": "D", "test_0311": "D", "test_0312": "B", "test_0313": "B", "test_0314": "D", "test_0315": "B", "test_0316": "A", "test_0317": "A", "test_0318": "D", "test_0319": "A", "test_0320": "B", "test_0321": "C", "test_0322": "C", "test_0323": "C", "test_0324": "B", "test_0325": "D", "test_0326": "D", "test_0327": "B", "test_0328": "D", "test_0329": "C", "test_0330": "CDAB", "test_0331": "DBCA", "test_0332": "ABDC", "test_0333": "CADB", "test_0334": "BACD", "test_0335": "DBCA", "test_0336": "BADC", "test_0337": "ADCB", "test_0338": "ACBD", "test_0339": "CDAB", "test_0340": "BADC", "test_0341": "DCBA", "test_0342": "ACDB", "test_0343": "CDAB", "test_0344": "CBDA", "test_0345": "DCAB", "test_0346": "DBCA", "test_0347": "BADC", "test_0348": "BACD", "test_0349": "ADBC", "test_0350": "DCBA", "test_0351": "DABC", "test_0352": "ABDC", "test_0353": "CDBA", "test_0354": "ABDC", "test_0355": "CADB", "test_0356": "DACB", "test_0357": "BACD", "test_0358": "ADBC", "test_0359": "DBAC", "test_0360": "A", "test_0361": "B", "test_0362": "D", "test_0363": "C", "test_0364": "A", "test_0365": "A", "test_0366": "C", "test_0367": "A", "test_0368": "C", "test_0369": "B", "test_0370": "C", "test_0371": "B", "test_0372": "A", "test_0373": "A", "test_0374": "C", "test_0375": "A", "test_0376": "D", "test_0377": "C", "test_0378": "D", "test_0379": "A", "test_0380": "A", "test_0381": "B", "test_0382": "B", "test_0383": "C", "test_0384": "B", "test_0385": "A", "test_0386": "B", "test_0387": "C", "test_0388": "A", "test_0389": "A", "test_0390": "D", "test_0391": "C", "test_0392": "C", "test_0393": "C", "test_0394": "C", "test_0395": "A", "test_0396": "B", "test_0397": "D", "test_0398": "D", "test_0399": "C", "test_0400": "B", "test_0401": "D", "test_0402": "C", "test_0403": "D", "test_0404": "B", "test_0405": "D", "test_0406": "D", "test_0407": "C", "test_0408": "D", "test_0409": "A", "test_0410": "B", "test_0411": "C", "test_0412": "C", "test_0413": "D", "test_0414": "C", "test_0415": "D", "test_0416": "A", "test_0417": "B", "test_0418": "B", "test_0419": "B", "test_0420": "D", "test_0421": "A", "test_0422": "C", "test_0423": "C", "test_0424": "B", "test_0425": "C", "test_0426": "B", "test_0427": "B", "test_0428": "D", "test_0429": "C", "test_0430": "D", "test_0431": "A", "test_0432": "C", "test_0433": "A", "test_0434": "A", "test_0435": "D", "test_0436": "C", "test_0437": "A", "test_0438": "B", "test_0439": "D", "test_0440": "C", "test_0441": "B", "test_0442": "C", "test_0443": "B", "test_0444": "B", "test_0445": "C", "test_0446": "C", "test_0447": "A", "test_0448": "C", "test_0449": "D", "test_0450": "A", "test_0451": "A", "test_0452": "A", "test_0453": "D", "test_0454": "D", "test_0455": "C", "test_0456": "C", "test_0457": "B", "test_0458": "B", "test_0459": "B", "test_0460": "A", "test_0461": "C", "test_0462": "B", "test_0463": "C", "test_0464": "D", "test_0465": "A", "test_0466": "A", "test_0467": "A", "test_0468": "B", "test_0469": "A", "test_0470": "B", "test_0471": "B", "test_0472": "C", "test_0473": "B", "test_0474": "B", "test_0475": "C", "test_0476": "A", "test_0477": "B", "test_0478": "C", "test_0479": "C", "test_0480": "C", "test_0481": "A", "test_0482": "C", "test_0483": "A", "test_0484": "C", "test_0485": "C", "test_0486": "A", "test_0487": "A", "test_0488": "C", "test_0489": "C", "test_0490": "A", "test_0491": "B", "test_0492": "B", "test_0493": "B", "test_0494": "C", "test_0495": "C", "test_0496": "B", "test_0497": "C", "test_0498": "A", "test_0499": "A", "test_0500": "A", "test_0501": "B", "test_0502": "B", "test_0503": "A", "test_0504": "C", "test_0505": "B", "test_0506": "C", "test_0507": "A", "test_0508": "A", "test_0509": "A", "test_0510": "B", "test_0511": "B", "test_0512": "A", "test_0513": "B", "test_0514": "A", "test_0515": "C", "test_0516": "C", "test_0517": "A", "test_0518": "C", "test_0519": "C", "test_0520": "A", "test_0521": "C", "test_0522": "C", "test_0523": "A", "test_0524": "B", "test_0525": "C", "test_0526": "D", "test_0527": "C", "test_0528": "D", "test_0529": "C", "test_0530": "D", "test_0531": "A", "test_0532": "B", "test_0533": "A", "test_0534": "B", "test_0535": "A", "test_0536": "C", "test_0537": "A", "test_0538": "C", "test_0539": "B", "test_0540": "D", "test_0541": "C", "test_0542": "B", "test_0543": "D", "test_0544": "C", "test_0545": "C", "test_0546": "D", "test_0547": "D", "test_0548": "A", "test_0549": "B", "test_0550": "B", "test_0551": "C", "test_0552": "D", "test_0553": "D", "test_0554": "D", "test_0555": "C", "test_0556": "C", "test_0557": "B", "test_0558": "C", "test_0559": "B", "test_0560": "C", "test_0561": "D", "test_0562": "A", "test_0563": "B", "test_0564": "B", "test_0565": "D", "test_0566": "B", "test_0567": "C", "test_0568": "B", "test_0569": "C", "test_0570": "C", "test_0571": "B", "test_0572": "B", "test_0573": "D", "test_0574": "D", "test_0575": "C", "test_0576": "BC", "test_0577": "BC", "test_0578": "CD", "test_0579": "C", "test_0580": "ABC", "test_0581": "ABD", "test_0582": "BD", "test_0583": "BD", "test_0584": "ACD", "test_0585": "BC", "test_0586": "C", "test_0587": "D", "test_0588": "AB", "test_0589": "C", "test_0590": "AC", "test_0591": "AD", "test_0592": "A", "test_0593": "BC", "test_0594": "D", "test_0595": "AB", "test_0596": "BCD", "test_0597": "D", "test_0598": "AB", "test_0599": "B", "test_0600": "BD", "test_0601": "BD", "test_0602": "C", "test_0603": "D", "test_0604": "B", "test_0605": "C", "test_0606": "ACD", "test_0607": "A", "test_0608": "AD", "test_0609": "C", "test_0610": "A", "test_0611": "A", "test_0612": "B", "test_0613": "D", "test_0614": "C", "test_0615": "B", "test_0616": "B", "test_0617": "A", "test_0618": "D", "test_0619": "A", "test_0620": "C", "test_0621": "A", "test_0622": "A", "test_0623": "B", "test_0624": "D", "test_0625": "A", "test_0626": "A", "test_0627": "B", "test_0628": "B", "test_0629": "D", "test_0630": "A", "test_0631": "C", "test_0632": "C", "test_0633": "C", "test_0634": "A", "test_0635": "D", "test_0636": "A", "test_0637": "A", "test_0638": "D", "test_0639": "A", "test_0640": "D", "test_0641": "BDAC", "test_0642": "DACB", "test_0643": "DBCA", "test_0644": "BCAD", "test_0645": "ACBD", "test_0646": "BADC", "test_0647": "DCBA", "test_0648": "ABCD", "test_0649": "ADBC", "test_0650": "C", "test_0651": "C", "test_0652": "B", "test_0653": "B", "test_0654": "A", "test_0655": "B", "test_0656": "D", "test_0657": "B", "test_0658": "B", "test_0659": "B", "test_0660": "D", "test_0661": "C", "test_0662": "C", "test_0663": "C", "test_0664": "C", "test_0665": "D", "test_0666": "B", "test_0667": "D", "test_0668": "D", "test_0669": "B", "test_0670": "C", "test_0671": "C", "test_0672": "C", "test_0673": "B", "test_0674": "C", "test_0675": "C", "test_0676": "A", "test_0677": "C", "test_0678": "B", "test_0679": "B", "test_0680": "A", "test_0681": "D", "test_0682": "C"}

def manner_group(value: str) -> str:
    if value in FAST:
        return "FAST"
    if value in SLOW:
        return "SLOW"
    if value in CARE:
        return "CARE"
    if value in NERV:
        return "NERV"
    return "NEUT"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_all_inputs() -> tuple[Path, Path, Path, Path]:
    input_dir = Path("/kaggle/input")
    print("Scanning /kaggle/input:", [str(p.name) for p in sorted(input_dir.iterdir())], flush=True)
    
    train_videos = list(input_dir.rglob("hau__user1__1-1-1.mp4"))
    if not train_videos:
        train_videos = list(input_dir.rglob("hau__*.mp4"))
    if not train_videos:
        raise FileNotFoundError("cannot locate HAU training videos (hau__*.mp4)")
    video_root = train_videos[0].parent

    test_videos = list(input_dir.rglob("test__LM_test_0065.mp4"))
    if not test_videos:
        test_videos = list(input_dir.rglob("test__*.mp4"))
    if not test_videos:
        sample_files = [str(p) for p in list(input_dir.rglob("*"))[:50]]
        raise FileNotFoundError(f"cannot locate HAU test videos (test__*.mp4). Sample files under input: {sample_files}")
    test_root = test_videos[0].parent

    qa_paths = list(input_dir.rglob("training_qa.csv"))
    if not qa_paths:
        raise FileNotFoundError("cannot locate training_qa.csv")
    training_qa_path = qa_paths[0]

    te_paths = list(input_dir.rglob("test_qa.csv"))
    if not te_paths:
        raise FileNotFoundError("cannot locate test_qa.csv")
    test_qa_path = te_paths[0]

    print(f"Paths located: video_root={video_root}, test_root={test_root}, "
          f"training_qa={training_qa_path}, test_qa={test_qa_path}", flush=True)
    return video_root, test_root, training_qa_path, test_qa_path


VIDEO_ROOT, TEST_ROOT, TRAINING_QA_PATH, TEST_QA_PATH = find_all_inputs()



def path_parts(path: str) -> tuple[str, str, int]:
    match = re.fullmatch(r"HAU/(user\d+)/(\d+-\d+)-(\d+)", path)
    if not match:
        raise ValueError(path)
    return match.group(1), match.group(2), int(match.group(3))


def session_of(path: str) -> str:
    user, session, _ = path_parts(path)
    return f"{user}/{session}"


def video_path(path: str) -> Path:
    user, session, trial = path_parts(path)
    result = VIDEO_ROOT / f"hau__{user}__{session}-{trial}.mp4"
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def test_video_path(clip: str) -> Path:
    result = TEST_ROOT / f"test__{clip}.mp4"
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def ordered_rows(group: pd.DataFrame) -> list[pd.Series]:
    rows = [part.iloc[0] for _, part in group.groupby("path")]
    return sorted(rows, key=lambda row: path_parts(row.path)[2])


def candidate_manners(rows: list[pd.Series]) -> list[str]:
    sets = [{str(row[letter]).strip() for letter in "ABCD"} for row in rows]
    return sorted(set.intersection(*sets)) if sets else []


def truth_manner(row: pd.Series) -> str:
    return str(row[str(row.answer)]).strip()


def valid_session(group: pd.DataFrame) -> bool:
    rows = ordered_rows(group)
    candidates = set(candidate_manners(rows))
    return len(rows) >= 2 and len(candidates) >= len(rows) and {
        truth_manner(row) for row in rows
    } <= candidates


def decode_frames(path: Path, rng: random.Random | None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"zero frames: {path}")
    edges = np.linspace(0, total, N_FRAMES + 1)
    indices = []
    for left, right in zip(edges[:-1], edges[1:]):
        lo = int(math.floor(left))
        hi = max(lo, int(math.ceil(right)) - 1)
        index = (rng.randint(lo, hi) if rng is not None else (lo + hi) // 2)
        indices.append(min(index, total - 1))
    result = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            frame = result[-1].copy() if result else np.zeros((224, 224, 3), np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result.append(frame)
    cap.release()
    if rng is not None and rng.random() < 0.5:
        result = [np.ascontiguousarray(frame[:, ::-1]) for frame in result]
    return result


qa = pd.read_csv(TRAINING_QA_PATH)
qa = qa[(qa.source == "HAU") & (qa.category == "emotion")].copy()
qa["user"] = qa.path.map(lambda value: path_parts(value)[0])
qa["session"] = qa.path.map(session_of)

# Full training: all valid sessions
train_sessions = [
    (name, group.copy()) for name, group in qa.groupby("session")
    if valid_session(group)
]


def position_prior(sessions) -> dict[tuple[str, int, int], float]:
    exact = Counter()
    totals = Counter()
    group = Counter()
    group_totals = Counter()
    for _, frame in sessions:
        rows = ordered_rows(frame)
        k = len(rows)
        for pos, row in enumerate(rows):
            manner = truth_manner(row)
            exact[(manner, pos, k)] += 1
            totals[(manner, k)] += 1
            grp = manner_group(manner)
            group[(grp, pos, k)] += 1
            group_totals[(grp, k)] += 1
    result = {}
    manners = {truth_manner(row) for _, frame in sessions for row in ordered_rows(frame)}
    for manner in manners:
        grp = manner_group(manner)
        for k in (2, 3):
            for pos in range(k):
                back = (group[(grp, pos, k)] + 1.0) / (group_totals[(grp, k)] + k)
                result[(manner, pos, k)] = (
                    exact[(manner, pos, k)] + 3.0 * back
                ) / (totals[(manner, k)] + 3.0)
    return result


POS_PRIOR = position_prior(train_sessions)
print(json.dumps({
    "device": str(DEVICE),
    "train_sessions": len(train_sessions),
    "test_triples": len(TEST_TRIPLES),
}, indent=2), flush=True)

seed_all(SEED)
processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_ID, num_labels=len(GROUPS), ignore_mismatched_sizes=True,
).to(DEVICE)
for parameter in model.videomae.parameters():
    parameter.requires_grad = False
for block in model.videomae.encoder.layer[-UNFREEZE_BLOCKS:]:
    for parameter in block.parameters():
        parameter.requires_grad = True
final_norm = model.fc_norm if model.fc_norm is not None else model.videomae.layernorm
if final_norm is not None:
    for parameter in final_norm.parameters():
        parameter.requires_grad = True
for parameter in model.classifier.parameters():
    parameter.requires_grad = True
rank_head = torch.nn.Linear(model.config.hidden_size, 1).to(DEVICE)

backbone_params = [p for p in model.videomae.parameters() if p.requires_grad]
head_params = list(model.classifier.parameters()) + list(rank_head.parameters())
if final_norm is not None:
    head_params += list(final_norm.parameters())
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LR},
    {"params": head_params, "lr": HEAD_LR},
], weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")


def pixels_for_train(rows: list[pd.Series], rng: random.Random | None) -> torch.Tensor:
    values = []
    for row in rows:
        frames = decode_frames(video_path(row.path), rng)
        values.append(processor(frames, return_tensors="pt").pixel_values[0])
    return torch.stack(values).to(DEVICE)


def session_loss(rows: list[pd.Series], rng: random.Random):
    pixel_values = pixels_for_train(rows, rng)
    targets = torch.tensor(
        [G2I[manner_group(truth_manner(row))] for row in rows],
        dtype=torch.long, device=DEVICE,
    )
    output = model.videomae(pixel_values)
    pooled = output.last_hidden_state.mean(1)
    if model.fc_norm is not None:
        pooled = model.fc_norm(pooled)
    logits = model.classifier(pooled)
    ce = F.cross_entropy(logits, targets, label_smoothing=0.05)
    speed = rank_head(pooled).flatten()
    rank_losses = [F.softplus(0.25 - (speed[j] - speed[i]))
                   for i in range(len(rows)) for j in range(i + 1, len(rows))]
    rank = torch.stack(rank_losses).mean() if rank_losses else ce.new_zeros(())
    candidates = candidate_manners(rows)
    truth = [candidates.index(truth_manner(row)) for row in rows]
    assignments = list(itertools.permutations(range(len(candidates)), len(rows)))
    scores = []
    for assignment in assignments:
        value = ce.new_zeros(())
        for pos, candidate_index in enumerate(assignment):
            manner = candidates[candidate_index]
            value = value + F.log_softmax(logits[pos], -1)[G2I[manner_group(manner)]]
            value = value + 0.30 * math.log(max(POS_PRIOR.get((manner, pos, len(rows)), 1 / len(rows)), 1e-6))
        scores.append(value)
    structured = F.cross_entropy(
        torch.stack(scores)[None],
        torch.tensor([assignments.index(tuple(truth))], device=DEVICE),
    )
    return ce + 0.35 * rank + 0.50 * structured, ce.detach(), rank.detach(), structured.detach()


print("--- Starting Full Training ---", flush=True)
optimizer.zero_grad(set_to_none=True)
global_step = 0
for epoch in range(EPOCHS):
    shuffled = list(train_sessions)
    random.Random(SEED + epoch).shuffle(shuffled)
    model.train()
    rank_head.train()
    metrics = []
    for item, (_, frame) in enumerate(shuffled, 1):
        rows = ordered_rows(frame)
        rng = random.Random(SEED + epoch * 10000 + item)
        if len(rows) == 3 and rng.random() < PAIR_AUGMENT_PROB:
            del rows[rng.randrange(3)]
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda", dtype=torch.float16):
            loss, ce, rank, structured = session_loss(rows, rng)
            scaled_loss = loss / GRAD_ACCUM
        scaler.scale(scaled_loss).backward()
        metrics.append((float(loss.detach()), float(ce), float(rank), float(structured)))
        if item % GRAD_ACCUM == 0 or item == len(shuffled):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(backbone_params + head_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        if item % 10 == 0:
            mean = np.mean(metrics[-10:], axis=0)
            print(f"epoch={epoch} session={item}/{len(shuffled)} step={global_step} "
                  f"loss={mean[0]:.4f} ce={mean[1]:.4f} rank={mean[2]:.4f} struct={mean[3]:.4f}", flush=True)

print("--- Full Training Complete. Starting Test Inference ---", flush=True)

te_df = pd.read_csv(TEST_QA_PATH).set_index("qa_id")

@torch.inference_mode()
def test_clip_logits(clip: str) -> np.ndarray:
    model.eval()
    frames = decode_frames(test_video_path(clip), None)
    pixel_values = processor(frames, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda", dtype=torch.float16):
        result = model(pixel_values=pixel_values).logits
    return result[0].float().cpu().numpy()


test_predictions = []
gated_flips = []

for session_idx, triple in enumerate(TEST_TRIPLES, 1):
    clips = [item[0] for item in triple]
    qa_ids = [item[1] for item in triple]
    qrows = [te_df.loc[qid] for qid in qa_ids]
    
    # Candidate manners
    sets = [{str(r[letter]).strip() for letter in "ABCD"} for r in qrows]
    common = sorted(set.intersection(*sets))
    k = len(clips)
    assert k == 3 and len(common) == 3, f"Unexpected size for session {clips}: k={k}, cand={common}"
    
    logits = np.stack([test_clip_logits(c) for c in clips])
    logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    
    scored = []
    for assignment in itertools.permutations(common, k):
        score = sum(logp[i, G2I[manner_group(assignment[i])]] for i in range(k))
        score += 0.30 * sum(math.log(max(POS_PRIOR.get((assignment[i], i, k), 1 / k), 1e-6))
                            for i in range(k))
        scored.append((float(score), assignment))
    scored.sort(reverse=True)
    best_score, best_assignment = scored[0]
    second_score, _ = scored[1]
    margin = best_score - second_score
    
    for i in range(k):
        clip = clips[i]
        qid = qa_ids[i]
        qrow = qrows[i]
        manner = best_assignment[i]
        pred_letter = next(letter for letter in "ABCD" if str(qrow[letter]).strip() == manner)
        champ_pred = CHAMPION_DICT[qid]
        
        is_flip = (margin >= GATE_MARGIN) and (pred_letter != champ_pred)
        item = {
            "session_idx": session_idx,
            "clip": clip,
            "qa_id": qid,
            "assigned_manner": manner,
            "model_prediction": pred_letter,
            "champion_prediction": champ_pred,
            "margin": float(margin),
            "is_gated_flip": bool(is_flip),
        }
        test_predictions.append(item)
        if is_flip:
            gated_flips.append({
                "qa_id": qid,
                "clip": clip,
                "old_champ": champ_pred,
                "new_pred": pred_letter,
                "margin": float(margin),
                "manner": manner,
            })

test_pred_df = pd.DataFrame(test_predictions)
test_pred_df.to_csv(OUT / "videomae_emotion_test_all_predictions.csv", index=False)

flips_df = pd.DataFrame(gated_flips)
flips_df.to_csv(OUT / "videomae_emotion_gated_flips.csv", index=False)

print(f"Total complete test triple questions: {len(test_pred_df)}")
print(f"Gated overrides (margin >= {GATE_MARGIN} and pred != champ): {len(flips_df)}")
print(flips_df.to_string())

# Build candidate submission
candidate_dict = dict(CHAMPION_DICT)
for _, flip in flips_df.iterrows():
    candidate_dict[flip["qa_id"]] = flip["new_pred"]

sub_rows = [{"qa_id": qid, "prediction": candidate_dict[qid]} for qid in sorted(candidate_dict)]
sub_df = pd.DataFrame(sub_rows)
sub_path = OUT / "submission_candidate_videomae_emotion.csv"
sub_df.to_csv(sub_path, index=False)

# Build exact diff
diff_rows = []
for qid in sorted(CHAMPION_DICT):
    if candidate_dict[qid] != CHAMPION_DICT[qid]:
        diff_rows.append({
            "qa_id": qid,
            "champion": CHAMPION_DICT[qid],
            "candidate": candidate_dict[qid],
        })
diff_df = pd.DataFrame(diff_rows)
diff_path = OUT / "submission_candidate_videomae_emotion.diff.csv"
diff_df.to_csv(diff_path, index=False)

# Compute SHA-256
sha256 = hashlib.sha256(sub_path.read_bytes()).hexdigest()

summary = {
    "model": MODEL_ID,
    "protocol": "full_training_5_epochs_test_inference",
    "test_sessions": len(TEST_TRIPLES),
    "test_questions": len(test_pred_df),
    "gate_margin": GATE_MARGIN,
    "flips_count": len(flips_df),
    "flips": gated_flips,
    "candidate_submission_sha256": sha256,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("\n--- Summary ---")
print(json.dumps(summary, indent=2))

