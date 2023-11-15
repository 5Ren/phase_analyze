import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# サンプリング周波数
fs = 2000.0

# 振幅時系列データ（適当なデータで置き換えてください）
amplitude_data = np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0])

# 時間軸の生成
time = np.arange(0, len(amplitude_data)) / fs

# 正弦波関数の定義
def sine_function(t, *params):
    A, f, phi, offset = params[:4]
    return A * np.sin(2 * np.pi * f * t + phi) + offset

# フィッティングパラメータの初期値
initial_params = [1.0, 180, 0.0, 0.0]

# 初期値の設定（これを変更して試してみてください）
bounds = ([0, 0, -np.pi, -np.inf], [np.inf, 2 * fs, np.pi, np.inf])

# 正弦波フィッティング
fit_params, _ = curve_fit(sine_function, time, amplitude_data, p0=initial_params, bounds=bounds)

# フィッティングした正弦波の生成
fit_amplitude = sine_function(time, *fit_params)

# プロット
plt.plot(time, amplitude_data, label='Raw Data')
plt.plot(time, fit_amplitude, label='Approximation', linestyle='--')

# フィッティングした正弦波の式を表示
equation_text = f'Fit Equation: A * sin(2π * {fit_params[1]:.2f} * t + {fit_params[2]:.2f}) + {fit_params[3]:.2f}'
plt.text(0.5, 1.2, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
