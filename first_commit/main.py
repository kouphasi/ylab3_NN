import numpy
import scipy.special
#from scipy.special import expit
import csv
from first_commit.neuralNetwork import neuralNetwork

input_nodes = 2#入力ノード数
hidden_nodes = 16#隠れ層のノード数
output_nodes = 3#出力ノード数
learning_rate = 0.3#学習率

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)#ニューラルネットワークのインスタンス生成

with open("learn_data.csv",encoding='utf-8-sig',newline='') as f:#CSVファイルから学習用のデータを取り出す
    reader = csv.reader(f)
    training_data_list = [row for row in reader]
    f.close()

data = numpy.array(training_data_list, dtype="float")

inputs, outputs = numpy.hsplit(data,[2])#学習用データの入力と出力に分割
outputs = outputs.astype(numpy.int32)

# 0 ~ 0.99の間にデータが収まるように調節する
sub_array = numpy.array([20,55],dtype="float")
div_array = numpy.array([17,45],dtype="float")
mult_array = numpy.array([0.99,0.98],dtype="float")
add_array = numpy.array([0.01,0.01],dtype="float")
inputs=(inputs-sub_array)/div_array*mult_array+add_array

for i in range(len(inputs)):#学習させる
    targets = numpy.zeros(output_nodes)
    targets[outputs[i][0]] = 0.99
    n.train(inputs[i], targets)

#テスト用データを同様に0 ~ 0.99 の間に収まるようにする
test_data = numpy.array([30,75], dtype="float")
sub_array = numpy.array([20,55],dtype="float")
div_array = numpy.array([17,45],dtype="float")
mult_array = numpy.array([0.99,0.98],dtype="float")
add_array = numpy.array([0.01,0.01],dtype="float")
test_data = (test_data-sub_array)/div_array*mult_array+add_array

#予測演算
n.query(test_data)

#あとは評価するだけ