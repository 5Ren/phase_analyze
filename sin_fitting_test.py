import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# CSVファイルのパス
csv_file_path = './vibration_samples/1-9_n1_analysed_height.csv'

fs = 2000

# CSVファイルの読み込み
data = np.genfromtxt(csv_file_path, delimiter=',')

# 指定範囲の行を設定（例: 3行目から10行目まで）
start_row = 18
end_row = 48

# 指定範囲のデータを取得
specified_range_data = data[start_row:end_row + 1, 1]

# 負の値を持つ要素を見つけ、それより前の要素を削除
neg_index = np.where(specified_range_data < 0)[0]
if len(neg_index) > 0:
    specified_range_data = specified_range_data[neg_index[-1] + 1:]

specified_range_data -= specified_range_data.mean()

# 時系列データの振幅を取得
amplitude_data = specified_range_data

# 時間軸の生成
time = np.arange(0, len(amplitude_data)) / fs


# 正弦波関数の定義
def sine_function(t, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * t + phi) + offset


# フィッティングパラメータの初期値
initial_params = [1.0, 180, 0.0, 0.0]

# 正弦波フィッティング
fit_params, _ = curve_fit(sine_function, time, amplitude_data, p0=initial_params)

# フィッティングした正弦波の生成
fit_amplitude = sine_function(time, *fit_params)

# フィッティングした正弦波の式を表示
equation_text = f'Fit Equation: A * sin(2π * {fit_params[1]:.2f} * t + {fit_params[2]:.2f}) + {fit_params[3]:.2f}'

# プロット
plt.plot(time, amplitude_data, label='Raw Data')
plt.plot(time, fit_amplitude, label=equation_text, linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()