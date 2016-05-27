# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# y_data = W_data * x_data + b_data というx_dataとy_dataの関係になるようにあらかじめ
# W_dataとb_dataの値を定める。機械学習が適切に行われるとWはこのW_dataに、bはこのb_dataに近づく
# なお、W_dataは２行２列のテンソル、b_dataは２行１列のテンソル
W_data = np.array([[0.1,0],[0,0.1]])
b_data = np.array([0.3,0.3])

# 乱数生成を利用して0から1の間の数値を持つ２行１列の行列「X_data」を浮動小数として100個生成
x_data = np.random.rand(100,2,1).astype("float64")

# その後で、生成した100個のxに対して、行列版でのy=0.1x+0.3となるようなyを100個生成
y_data = W_data * x_data + b_data

# 上記で生成したxとy（共に２行１列のテンソル）の組を学習データとして用いる

# 機械学習で最適化するWとbを設定する。Wは２行２列のテンソル。bは２行１列のテンソル。
W = tf.Variable(tf.random_uniform([2,2],-1.0,1.0))
b = tf.Variable(tf.zeros([2]))
y = W * x_data + b

# TensorBoardへ表示するための変数を用意する（ヒストグラム用）
W_hist = tf.histogram_summary("weights",W)
b_hist = tf.histogram_summary("biases",b)
y_hist = tf.histogram_summary("y",y)

# 学習において、その時点での学習のダメ程度を表すlossを、学習データのyとその時点でのyの差の２乗と定義
# Wとbの最適化のアルゴリズムを最急降下法（勾配法）とし、その１回の最適化処理にoptimizerと名前を付ける
# 上記の最適化処理の繰り返しによりlossを最小化する処理をtrainと呼ぶことにする
loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# TensorBoardへloss（学習のダメ具合の指標として設定したスカラー値）を表示するための変数を用意する（イベント用）
loss_sum = tf.scalar_summary("loss",loss)

# Launch the graph.（おきまりの文句）
sess = tf.Session()

# 上記で用意した合計４つのTensorBoard描画用の変数を、TensorBoardが利用するSummaryデータとしてmerge（合体）する
# また、そのSummaryデータを書き込むSummaryWriterを用意し、書き込み先を'/tmp/tf_logs'ディレクトリに指定する
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("./tmp/tf_logs",sess.graph_def)

# 学習を始める前にこのプログラムで使っている変数を全てリセットして空っぽにする
init = tf.initialize_all_variables()
sess.run(init)

# 学習を1000回行い、100回目ごとに画面に学習回数とWとbのその時点の値を表示する
for step in xrange(1001):
    #sess.run(train)
    if step % 10 == 0:
        result = sess.run([merged, loss])
        summary_str = result[0]
        writer.add_summary(summary_str,step)
        print step, sess.run(W), sess.run(b)
    else:
        sess.run(train)
