# %%
import numpy as np
import scipy.special
import csv
import math
import pandas as pd

# %%
from neuralNetwork import neuralNetwork



def nn(hidden_nodes):

    input_nodes = 5
    # hidden_nodes = 64
    output_nodes = 3 
    learning_rate = 0.3


    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


    import makedata


    Data = makedata.MakeData()

    inputs = Data.input_learn
    outputs = Data.output_learn
    input_test = Data.input_test
    output_test = Data.output_test
    mses = []
    num_list = []
    es = []#期待値の評価

    import scipy
    import scipy.special


    for i in range(len(inputs)):
        targets = np.zeros(output_nodes)
        targets[int(outputs[i][0])-1] = 0.99
        n.train(inputs[i], targets)
        num_list.append(i)
        mse = 0
        e_true = 0.99*outputs[i][0]#本来の期待値
        for k in range(len(input_test)):
            query = n.query(input_test[k])
            # print(query)
            test_targets = np.zeros(output_nodes)
            test_targets[int(output_test[k][0])-1] = 0.99
        e = 0
        for j in range(len(query)):
            e += query[j][0]*(j+1)
            mse += (query[j][0] - test_targets[j])**2
            # print(mse)
        es.append(abs(e_true-e))
        mse = mse/5/len(input_test)

        mses.append(mse)
    return mses, es



import matplotlib.pyplot as plt


# plt.plot(mses, ".")

# %%
# n.query(input_test[1])

# # %%
# x = np.arange(-4,4,0.025)
# y = np.array([ i*(i>0.0) for i in x])

# # %%
# plt.plot(x,y)
# plt.title("ReLU Function")
# plt.show()

# # %%




# %%
