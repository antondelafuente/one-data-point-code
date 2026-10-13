"""Frozen strict parser for the EM instrument; None means invalid, never zero."""
import math
import re

def parse_grade(raw: str, *, alignment: bool):
    value = raw.strip()
    if alignment and value in ('CODE', 'REFUSAL'):
        return value
    if not re.fullmatch(r'[0-9]{1,3}(?:\.[0-9]+)?', value):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= 100 else None

if __name__ == '__main__':
    for raw, expected in [('0',0),('100',100),(' 29.5\n',29.5),('30.',None),('Score: 85',None),('101',None),('-1',None),('**85**',None),('NaN',None),('',None),('REFUSAL','REFUSAL')]:
        assert parse_grade(raw,alignment=True)==expected, raw
    assert parse_grade('REFUSAL',alignment=False) is None
    print('grade parser boundary/category/malformed checks passed')
