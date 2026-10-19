# Copyright (c) 2023 embodied-generalist. MIT License; see LICENSE-LEO.txt.
# Retained from the supplied Scene-R1 QA code, which attributes these functions
# to LEO commit 477dc44b8b18dbfbe6823c307436d896ec8b062e.
# Modification: ignore empty normalized references during refined matching.
import re


def answer_match(pred, gts):
    # Empty normalized references must not match every prediction.
    gts = [gt for gt in gts if gt.strip()]
    if len(pred) == 0:
        return 0, 0
    if pred in gts:
        return 1, 1
    for gt in gts:
        if ''.join(pred.split()) in ''.join(gt.split()) or ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0


# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/data/data_utils.py#L322-L379
def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$', '', data)
    data = re.sub('^[ ]+', '', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub(r'\.[ ]{2,}', '. ', data)
    data = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", '', data)
    data = re.sub('ç', 'c', data)
    data = re.sub('’', '\'', data)
    data = re.sub(r'\bletf\b', 'left', data)
    data = re.sub(r'\blet\b', 'left', data)
    data = re.sub(r'\btehre\b', 'there', data)
    data = re.sub(r'\brigth\b', 'right', data)
    data = re.sub(r'\brght\b', 'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b', 'TV', data)
    data = re.sub(r'\bchai\b', 'chair', data)
    data = re.sub(r'\bwasing\b', 'washing', data)
    data = re.sub(r'\bwaslked\b', 'walked', data)
    data = re.sub(r'\boclock\b', 'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b', 'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b', r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)', r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data
