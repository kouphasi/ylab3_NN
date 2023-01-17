# %%
import pandas as pd
import numpy as np

# %%
from sklearn import preprocessing
import scipy.stats

# %%

class MakeData:
    def __init__(self):
        lifex = pd.read_csv("../Life expectancy.csv")
        suicide_rate = pd.read_csv("../Suicide Rate.csv")


        df = pd.merge(lifex,suicide_rate)


        input_array = df.iloc[:,1:6].values
        output_array = df.iloc[:,7].values.reshape((-1, 1))

        # input_array = preprocessing.MinMaxScaler().fit_transform(input_array)*0.999999999
        input_array = scipy.stats.zscore(input_array)

        # output_array = scipy.stats.zscore(df.iloc[:,6].values.reshape((-1, 1)))
        # output_array = preprocessing.MinMaxScaler().fit_transform(output_array)*0.999999999

        #シャッフルする
        np.random.shuffle(input_array)
        np.random.shuffle(output_array)
        # print(input_array)

        self.input_learn = input_array[:80]
        self.output_learn = output_array[:80]
        self.input_test = input_array[80:]
        self.output_test = output_array[80:]



# %%
