import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

subplot_row, subplot_column = 4, 2
x_axis_limit = 0, 300


def fetch_amplitude(start_index, end_index, stretch_data, subplot_num):
    # 読み込んだそのままのデータ
    plt.subplot(subplot_row, subplot_column, subplot_num)
    ymin, ymax = -2, 4
    plt.vlines(start_index, ymin, ymax, colors='orange', label='FFT Start', linestyle='dashed', linewidth=2)
    plt.vlines(end_index, ymin, ymax, colors='green', label='FFT End', linestyle='dashed', linewidth=2)
    plt.plot(stretch_data[:, 1])
    plt.title('Row Data')
    plt.xlabel('Index')
    plt.ylabel('Droplet stretch (mm)')
    plt.xlim(x_axis_limit)
    plt.legend()

    # 指定範囲のデータを取得
    data_range_changed = stretch_data[start_index:end_index + 1, 1]

    # データをゼロ基準に (振幅にする)
    amplitude_data = data_range_changed - data_range_changed.mean()

    # 時系列データのプロット
    plt.subplot(subplot_row, subplot_column, subplot_num + 1)
    plt.plot(amplitude_data)
    plt.title('Amplitude Data')
    plt.xlabel('Index')
    plt.ylabel('Amplitude (mm)')

    return amplitude_data


def excuse_fft(amplitude_data, dt_sec, subplot_num):
    # FFTを実行
    fft_result = np.fft.fft(amplitude_data)
    fft_freq = np.fft.fftfreq(len(amplitude_data), d=dt_sec)

    # ピークのインデックスを取得
    peak_index = np.argmax(np.abs(fft_result))
    peak_frequency = np.abs(fft_freq[peak_index])

    # ピーク周波数を強調したFFT結果（振幅）のプロット
    plt.subplot(subplot_row, subplot_column, subplot_num)
    plt.plot(np.abs(fft_freq), np.abs(fft_result))
    plt.scatter(np.abs(fft_freq[peak_index]), np.abs(fft_result[peak_index]), color='red', marker='o',
                label=f'Peak Frequency: {peak_frequency} Hz')
    plt.title('FFT Result with Peak Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (mm)')
    plt.legend()
    return peak_frequency


def sine_function(t, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * t + phi) + offset


def fitting_sin_curve(amplitude_data, fps, frequency, subplot_num):
    # 時間軸の生成
    time = np.arange(0, len(amplitude_data)) / fps
    # 衝突前の時間
    end_time_s = time.max()
    print(f'{end_time_s=}')

    # 振幅
    amplitude_mm = (amplitude_data.max() + abs(amplitude_data.min())) / 2

    if amplitude_mm < 1:
        amplitude_mm = 1.0

        # フィッティングパラメータの初期値
    initial_params = [amplitude_mm, frequency, 0.0, 0.0]

    # 正弦波フィッティング
    fit_params, _ = curve_fit(sine_function, time, amplitude_data, p0=initial_params)

    # フィッティングした正弦波の生成
    fit_amplitude = sine_function(time, *fit_params)

    # フィッティングした正弦波の式を表示
    equation_text = f'Fit Equation: A * sin(2π * {fit_params[1]:.2f} * t + {fit_params[2]:.2f}) + {fit_params[3]:.2f}'

    # 終わりの位相
    end_phase = 2 * np.pi * fit_params[1] * end_time_s + fit_params[2]

    # プロット
    plt.subplot(subplot_row, subplot_column, subplot_num)
    plt.title('Sin curve fitting')
    plt.plot(time, amplitude_data, label='Raw Data')
    plt.plot(time, fit_amplitude, color='red', label=equation_text, linestyle='--')
    ymin, ymax = -0.15, 0.15
    plt.vlines(end_time_s, ymin, ymax, colors='green', linewidth=2,
               label=f'End phase: {(end_phase / np.pi):.2f} π ')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    return end_phase


if __name__ == '__main__':
    # CSVファイルのパス
    csv_file_path = './vibration_samples/1-9_n1_analysed_height.csv'

    # サンプリング周波数 (フレームレート 2000 fps)
    fps = 2000

    # グラフのプロット
    plt.figure(figsize=(12, 12))

    # CSVファイルの読み込み
    row_data = np.genfromtxt(csv_file_path, delimiter=',')

    # FFTする範囲を ndarray にする
    specified_range_data = fetch_amplitude(start_index=70, end_index=200, stretch_data=row_data, subplot_num=1)

    # FFT を実行し、周波数をとってくる
    frequency = excuse_fft(amplitude_data=specified_range_data, dt_sec=fps ** -1, subplot_num=3)

    # 正弦波にフィッティングするデータをndarrayにする
    sin_curve_fitting_data = fetch_amplitude(start_index=20, end_index=48, stretch_data=row_data, subplot_num=5)

    # sin curve フィッティングして，位相を取ってくる
    fitting_sin_curve(amplitude_data=sin_curve_fitting_data, fps=fps, frequency=frequency, subplot_num=7)

    # グラフ表示
    plt.tight_layout()
    plt.show()
