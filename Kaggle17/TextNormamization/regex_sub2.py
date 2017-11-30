
import pandas as pd
import pickle
import operator
import os
import os
import operator
from num2words import num2words
import gc
from regex_kernel import *

INPUT_PATH = 'input/'
DATA_INPUT_PATH = 'input/en_with_types'
SUBM_PATH = INPUT_PATH


train = pd.read_csv('input/en_train.csv')
test = pd.read_csv('input/test2_with_class.csv',encoding='latin1')

out = []
classes = list(test['class'])
before = list(test['before'])

SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
OTH = str.maketrans("፬", "4")


sdict = {}
sdict['km2'] = 'square kilometers'
sdict['km'] = 'kilometers'
sdict['kg'] = 'kilograms'
sdict['lb'] = 'pounds'
sdict['dr'] = 'doctor'
sdict['m²'] = 'square meters'

res = pickle.load(open('dict.pkl','rb'))

changed = []
changes=0
total=0
out = open(os.path.join(SUBM_PATH, 'sub_with_class.csv'), "w", encoding='UTF8')
out.write('"id","after"\n')
test = open(os.path.join(INPUT_PATH, "en_test_2.csv"), encoding='UTF8')
line = test.readline().strip()
while 1:
    line = test.readline().strip()
    if line == '':
        break

    pos = line.find(',')
    i1 = line[:pos]
    line = line[pos + 1:]

    pos = line.find(',')
    i2 = line[:pos]
    line = line[pos + 1:]

    line = line[1:-1]
    out.write('"' + i1 + '_' + i2 + '",')
    if line in res:
        srtd = sorted(res[line].items(), key=operator.itemgetter(1), reverse=True)
        out.write('"' + srtd[0][0] + '"')
        changes += 1
        changed.append(1)
    else:
        changed.append(0)
        if len(line) > 1:
            val = line.split(',')
            if len(val) == 2 and val[0].isdigit and val[1].isdigit:
                line = ''.join(val)

        if line.isdigit():
            srtd = line.translate(SUB)
            srtd = srtd.translate(SUP)
            srtd = srtd.translate(OTH)
            out.write('"' + num2words(float(srtd)) + '"')
            changes += 1
        elif len(line.split(' ')) > 1:
            val = line.split(' ')
            for i, v in enumerate(val):
                if v.isdigit():
                    srtd = v.translate(SUB)
                    srtd = srtd.translate(SUP)
                    srtd = srtd.translate(OTH)
                    val[i] = num2words(float(srtd))
                elif v in sdict:
                    val[i] = sdict[v]

            out.write('"' + ' '.join(val) + '"')
            changes += 1
        else:
            out.write('"' + line + '"')

    out.write('\n')
    total += 1

print('Total: {} Changed: {}'.format(total, changes))
test.close()
out.close()


test = pd.read_csv('input/sub_with_class.csv')

test['changed'] = changed

withclass = pd.read_csv('input/test2_with_class.csv',encoding = 'latin1')

test['class'] = list(withclass['class'])

def futher_change(changed,cls,x):
    if changed==1:
        return x
    else:
        if cls=='CARDINAL':
            return cardinal(x)
        elif cls=='DIGIT':
            return digit(x)
        elif cls=='LETTERS':
            return letters(x)
        elif cls=='TELEPHONE':
            return telephone(x)
        elif cls=='ELECTRONIC':
            return electronic(x)
        elif cls=='MONEY':
            return money(x)
        elif cls=='FRACTION':
            return fraction(x)
        elif cls =='ORDINAL':
            return ordinal(x)
        elif cls =='ADDRESS':
            return address(x)
        else:
            return x

test.columns
test['after'] = test.apply(lambda x:futher_change(x['changed'],x['class'],x['after']),axis=1)

test3 = test[['id','after']]

test3.to_csv('input/sub_after_tune.csv',index=False)