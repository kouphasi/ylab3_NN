# はじめに

これは、2022 年度秋学期ゼミの分析発表ように用いたコードです

データは[Kaggle](https://www.kaggle.com/datasets/marshuu/suicide-rate-and-life-expectancy?resource=download&select=Suicide+Rate.csv)から引用させていただいています

# データのありか

基本的にデータの分析は、first_commit 直下の`main.ipynb`内に記述しています

隠れ層のノード数をいじったりしてグラフ出力しているものは first_commit 内の`many_hiddens.ipynb`内に記述しています

`first_commit/execute_function.py`の中には様々な隠れ層のノード数で学習させられるように関数を定義しています

`first_commit/neuralNetwork.py`にニューラルネットワークのクラスを記述しています

`fires_commit/makedata.py`でデータを習得、加工済みのインスタンス変数を定義して簡単に入力、出力データを取得できるようなクラスを定義しています
