# Importamos librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import true

# 1)
# a)
# Leemos el archivo
dfcrossmatches_small = pd.read_pickle("dfcrossmatches_small.pickle")

print(dfcrossmatches_small)

classALeRCE = dfcrossmatches_small['classALeRCE']

# graficamos
pd.Index.value_counts(classALeRCE).plot.bar()
# plt.show()

# b)
# Leemos el archivo
features_small = pd.read_pickle("features_small.pickle")

print(features_small)

# pivoteamos
fs_2 = features_small.pivot(columns = ['name', 'fid'], values = 'value')

print(fs_2)

# c)
# vemos la media por columna
fs_3 = fs_2.median()

# reemplazamos los NaN
fs_4 = fs_2.fillna(value=fs_3)

print(fs_4)

# d)
# concatenamos
conca = pd.concat([classALeRCE, fs_4], axis=1)

print(conca)

# eliminamos NaNs
drop = conca.dropna()

print(drop)

# e)
# graficamos
pd.Index.value_counts(drop['classALeRCE']).plot.bar()
# plt.show()

# f)
# agrupamos clases
agru = drop.replace({'EA':'EB', 'EB/EW':'EB'})

# se eliminan las clases ...
elim = agru.loc[~agru['classALeRCE'].isin(['NLAGN', 'NLQSO', 'TDE', 'ZZ'])]
print(elim)

# g)
# Matriz de diseno
X = elim.drop(columns='classALeRCE')

# Vector de nombres
y = elim['classALeRCE']

# 2
