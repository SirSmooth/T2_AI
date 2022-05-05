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

# classALeRCE.plot.bar(np.array(pd.Index.value_counts(classALeRCE).index), pd.Index.value_counts(classALeRCE)['classALeRCE'].astype(float), rot=0)

pd.Index.value_counts(classALeRCE).plot.bar()
plt.show()

# b)