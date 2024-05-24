PSO(粒子群最適化)
=============
# 概要
このリポジトリは、多変数最適化問題を解くための粒子群最適化アルゴリズムを実装したものです。

## Step 1
コンパイルします
```
g++ -std=c++11 pso.cpp
```

## Step 2
実行します
```
./a.out
```

時間を計測する場合
```
time ./a.out
```

# 課題
10次元のRosenbrock関数の最適化を行った。
Rosenbrock関数は次のように定義される。
```
f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
```
この関数は最小値0を持つが、その最小値を求めることは容易ではない。

```
T = 100;
NUM_PARTICLES = 100;
vector<double> w_values = {0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
vector<double> c1_values = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
vector<double> c2_values = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
```
上記の条件でグリッドサーチを行い、最適なパラメータを求めた。
結果は、w:0.6, c1:0, c2:2.5であった。

そこで、以下のようにさらに詳細なグリッドサーチを行った。
```
vector<double> w_values = {0.55,0.6,0.65};
vector<double> c1_values = {0.0,0.1,0.2,0.3,0.4,0.5};
vector<double> c2_values = {2.5,2.6,2.7,2.8,2.9,3.0};
```
結果は、w:0.55, C1:0.1, C2:2.5であった。


