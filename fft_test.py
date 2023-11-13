import numpy as np
import matplotlib.pyplot as plt

# CSVファイルのパス
csv_file_path = './vibration_samples/1-9_n1_analysed_height.csv'

# CSVファイルの読み込み
data = np.genfromtxt(csv_file_path, delimiter=',')
# 2列目のデータを取得
col_2_data = data[:, 1]

# 負の値を持つ要素を見つけ、それより前の要素を削除
neg_index = np.where(col_2_data < 0)[0]
if len(neg_index) > 0:
    col_2_data = col_2_data[neg_index[-1] + 1:]

# 時系列データの振幅を取得
amplitude_data = col_2_data

# サンプリング間隔とデータ数の設定
dt = 1/2000
N = len(amplitude_data)

# フーリエ変換を実行
y_fft = np.fft.fft(amplitude_data)
freq = np.fft.fftfreq(N, dt)
amp = np.abs(y_fft / N)

# 周波数スペクトルのプロット
plt.plot(freq[:N//2], amp[:N//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.ylim(0, 0.04)
plt.show()
