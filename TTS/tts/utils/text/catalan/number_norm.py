""" skeleton from https://github.com/keithito/tacotron """

import re
from typing import Dict


_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_currency_re = re.compile(r"(€|£|\$|¥)([0-9\,\.]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(r|n|t|è)") # 1r, 2n, 3r, 4t, 5è, 6è, ...
_number_re = re.compile(r"-?[0-9]+")


def _remove_commas(m):
    pass

def _expand_decimal_point(m):
    pass

def __expand_currency(value: str, inflection: Dict[float, str]) -> str:
    pass

def _expand_currency(m: "re.Match") -> str:
    pass

def _expand_ordinal(m):
    pass

def _expand_number(m):
    pass

def normalize_numbers(text):
    # Look at https://pypi.org/project/inflect/ for number to string conversion
    return text
