# Importamos librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1)
# a)
# Leemos el archivo
dfcrossmatches_small = pd.read_pickle("dfcrossmatches_small.pickle")

print(dfcrossmatches_small)

classALeRCE = dfcrossmatches_small['classALeRCE']

# print(classALeRCE)

pd.Index.value_counts(classALeRCE).plot.bar()
plt.show()

# b)
# Leemos el archivo
features_small = pd.read_pickle("features_small.pickle")

print(features_small)

