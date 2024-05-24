#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

// PSOクラスの定義
class PSO {
public:
    PSO(int num_particles, int dim, int iterations, double w, double c1, double c2);
    void initialize();
    void optimize();
    double get_best_solution();
    void print_best_solution();

private:
    double rosenbrock(const vector<double>& x);

    int num_particles;  // 粒子の数
    int dim;            // 次元
    int iterations;     // 実行回数
    double w;           // 慣性係数
    double c1;          // 認知係数
    double c2;          // 社会係数

    // 乱数生成器
    random_device rd;
    mt19937 gen;
    uniform_real_distribution<> dis;
    uniform_real_distribution<> pos_dis;
    uniform_real_distribution<> vec_dis;

    // 粒子の位置、速度、暫定解、および評価値
    vector<vector<double>> x;
    vector<vector<double>> v;
    vector<vector<double>> p;
    vector<double> fp;
    vector<double> g;
    double fg;
};

// コンストラクタ(授業スライドの(1)(2))
PSO::PSO(int num_particles, int dim, int iterations, double w, double c1, double c2)
    : num_particles(num_particles), dim(dim), iterations(iterations), w(w), c1(c1), c2(c2), 
      gen(rd()), dis(0.0, 1.0), pos_dis(-2.048, 2.048), vec_dis(-1.0, 1.0),
      x(num_particles, vector<double>(dim)), v(num_particles, vector<double>(dim, 0.0)), 
      p(num_particles, vector<double>(dim)), fp(num_particles, numeric_limits<double>::max()), 
      g(dim), fg(numeric_limits<double>::max()) {}

// Rosenbrock関数の定義
double PSO::rosenbrock(const vector<double>& x) {
    double result = 0.0;
    for (int i = 0; i < dim - 1; ++i) {
        result += 100.0 * pow(x[i + 1] - x[i] * x[i], 2) + pow(1.0 - x[i], 2);
    }
    return result;
}

// 変数の初期化(授業スライドの(3))
void PSO::initialize() {
    // 粒子の初期化
    for (int k = 0; k < num_particles; ++k) {
        for (int d = 0; d < dim; ++d) {
            x[k][d] = pos_dis(gen);   // 位置の初期化
            v[k][d] = vec_dis(gen);   // 速度の初期化
        }
    }
}

// PSOアルゴリズムの実行(授業スライドの(4)(5))
void PSO::optimize() {
    for (int t = 0; t < iterations; ++t) {
        // 粒子の位置の更新
        for (int i = 0; i < num_particles; ++i) {
            for (int d = 0; d < dim; ++d) {
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

        // 粒子の速度の更新
        for (int i = 0; i < num_particles; ++i) {
            for (int d = 0; d < dim; ++d) {
                double r1 = dis(gen);
                double r2 = dis(gen);
                v[i][d] = w * v[i][d] +
                          c1 * r1 * (p[i][d] - x[i][d]) +
                          c2 * r2 * (g[d] - x[i][d]);
            }
        }
    }
}

// 最良の解を出力
void PSO::print_best_solution() {
    cout << "最適解: ";
    for (double val : g) {
        cout << val << " ";
    }
    cout << endl;
    cout << "評価値: " << fg << endl;
}

double PSO::get_best_solution() {
    return fg;
}

void grid_search(int num_particles, int dim, int iterations, const vector<double>& w_values, const vector<double>& c1_values, const vector<double>& c2_values) {
    double best_fg = numeric_limits<double>::max();
    vector<double> best_params(3);

    for (double w : w_values) {
        for (double c1 : c1_values) {
            for (double c2 : c2_values) {
                cout << "W: " << w << ", C1: " << c1 << ", C2: " << c2 << endl;
                double avg_fg = 0.0;
                for (int i = 0; i < 10; ++i) {
                    PSO pso(num_particles, dim, iterations, w, c1, c2);
                    pso.initialize();
                    pso.optimize();
                    avg_fg += pso.get_best_solution();
                }
                avg_fg /= 10.00000;
                if (avg_fg < best_fg) {
                    best_fg = avg_fg;
                    best_params = {w, c1, c2};
                }
            }
        }
    }

    cout << "最適なパラメータ:" << endl;
    cout << "W: " << best_params[0] << ", C1: " << best_params[1] << ", C2: " << best_params[2] << endl;
    cout << endl;
    cout << "最小値: " << best_fg << endl;
}


int main() {
    int NUM_PARTICLES = 100;  // 粒子の数
    int T = 5000;             // 実行回数
    const int DIM = 10;              // 次元
    const double W = 0.55;           // 慣性係数
    const double C1 = 0.1;          // 認知係数
    const double C2 = 2.5;          // 社会係数

    // // PSOクラスのインスタンスを生成
    PSO pso(NUM_PARTICLES, DIM, T, W, C1, C2);
    pso.initialize();  // 変数の初期化
    pso.optimize();    // PSOアルゴリズムの実行
    pso.print_best_solution();  // 最良の解を出力

     // グリッドサーチの範囲を定義
    // vector<double> w_values = {0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    // vector<double> c1_values = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
    // vector<double> c2_values = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
    vector<double> w_values = {0.55,0.6,0.65};
    vector<double> c1_values = {0.0,0.1,0.2,0.3,0.4,0.5};
    vector<double> c2_values = {2.5,2.6,2.7,2.8,2.9,3.0};

    // グリッドサーチを実行
    T = 100;
    NUM_PARTICLES = 100;
    //grid_search(NUM_PARTICLES, DIM, T, w_values, c1_values, c2_values);


    return 0;
}
