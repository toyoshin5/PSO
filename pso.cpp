#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

const int NUM_PARTICLES = 30;
const int NUM_ITERATIONS = 100;
const int DIM = 2; // 問題の次元
const double W = 0.5;
const double C1 = 2.0;
const double C2 = 2.0;

// 乱数生成器
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0.0, 1.0);
uniform_real_distribution<> pos_dis(-5.0, 5.0);

// Rosenbrock関数の定義
double rosenbrock(const vector<double>& x) {
    double result = 0.0;
    for (int i = 0; i < DIM - 1; ++i) {
        result += 100.0 * pow(x[i + 1] - x[i] * x[i], 2) + pow(1.0 - x[i], 2);
    }
    return result;
}

/*
変数の定義
DIM:次元
x: 粒子の位置 x[k][t]はt回目の反復におけるk番目の粒子の位置ベクトル
v: 粒子の速度 v[k][t]はt回目の反復におけるk番目の粒子の速度ベクトル
g: 全粒子がこれまでに発見した暫定解。初期値は空
fg: gの評価値 (初期値は十分大きい値とする)
p: 各粒子がこれまでに発見した暫定解。p[k]はk番目の粒子の暫定解
fp: 各粒子の暫定解の評価値　fp[k]はk番目の粒子の暫定解の評価値 (初期値は十分大きい値とする)
*/

int main() {
    vector<vector<double>> x(NUM_PARTICLES, vector<double>(DIM));
    vector<vector<double>> v(NUM_PARTICLES, vector<double>(DIM, 0.0));
    vector<vector<double>> p = x;
    vector<double> fp(NUM_PARTICLES, numeric_limits<double>::max());
    vector<double> g(DIM);
    double fg = numeric_limits<double>::max();

    // 粒子の初期化
    for (int k = 0; k < NUM_PARTICLES; ++k) {
        for (int d = 0; d < DIM; ++d) {
            x[k][d] = pos_dis(gen);
        }
    }

    // PSOのメインループ
    for (int t = 0; t < NUM_ITERATIONS; ++t) {
        for (int i = 0; i < NUM_PARTICLES; ++i) {
            for (int d = 0; d < DIM; ++d) {
                x[i][d] += v[i][d];
            }

            double score = rosenbrock(x[i]);
            if (score < fp[i]) {
                p[i] = x[i];
                fp[i] = score;
            }

            if (score < fg) {
                fg = score;
                g = x[i];
            }
        }

        for (int i = 0; i < NUM_PARTICLES; ++i) {
            for (int d = 0; d < DIM; ++d) {
                double r1 = dis(gen);
                double r2 = dis(gen);
                v[i][d] = W * v[i][d] +
                                   C1 * r1 * (p[i][d] - x[i][d]) +
                                   C2 * r2 * (g[d] - x[i][d]);
            }
        }
    }

    cout << "最良位置: ";
    for (double val : g) {
        cout << val << " ";
    }
    cout << endl;
    cout << "最小値: " << fg << endl;

    return 0;
}