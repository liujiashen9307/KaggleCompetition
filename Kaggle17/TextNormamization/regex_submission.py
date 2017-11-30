# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:37:22 2017

@author: Jiashen Liu
"""


import os
import operator
from num2words import num2words
import gc
from regex_kernel import *


INPUT_PATH = 'input/'
DATA_INPUT_PATH = 'input/en_with_types'
SUBM_PATH = INPUT_PATH

SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
OTH = str.maketrans("፬", "4")

def reg_trans(clas,x):
    if clas=='ADDRESS':
        return address(x)
    elif clas == 'CARDINAL':
        return cardinal(x)
    elif clas=='DIGIT':
        return digit(x)
    elif clas =='ELECTRONIC':
        return electronic(x)
    elif clas =='DIGIT':
        return fraction(x)
    elif clas == 'MONEY':
        return money(x)
    elif clas == 'TELEPHONE':
        return telephone(x)
    elif clas =='LETTERS':
        return letters(x)
    elif clas == ''

