import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# サンプリング周波数
fs = 2000.0

# 振幅時系列データ（適当なデータで置き換えてください）
amplitude_data = np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 1.0, 0.5])

# 時間軸の生成
time = np.arange(0, len(amplitude_data)) / fs

# 正弦波関数の定義
def sine_function(t, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * t + phi) + offset

# フィッティングパラメータの初期値
initial_params = [1.0, 1.0, 0.0, 0.0]

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
