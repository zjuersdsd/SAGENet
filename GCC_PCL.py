import librosa
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import chirp
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from utils.Opt_ import Opt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in arcsin")

def cfar(signal, guard_cells, training_cells, rate_fa):
    """
    CFAR 算法实现
    :param signal: 输入信号
    :param guard_cells: 保护单元数
    :param training_cells: 训练单元数
    :param rate_fa: 虚警率
    :return: 检测到的峰值索引
    """
    num_cells = len(signal)
    peak_indices = []

    for i in range(training_cells + guard_cells, num_cells - training_cells - guard_cells):
        # 训练单元
        leading_training_cells = signal[i - training_cells - guard_cells:i - guard_cells]
        trailing_training_cells = signal[i + guard_cells + 1:i + guard_cells + 1 + training_cells]

        # 计算噪声水平
        noise_level = np.mean(np.concatenate((leading_training_cells, trailing_training_cells)))

        # 计算检测阈值
        threshold = noise_level * rate_fa

        # 检测峰值
        if signal[i] > threshold:
            peak_indices.append(i)

    return peak_indices

def gcc_phat(sig, refsig, fs=1, thresh=0.3, plot_peaks=True):
    """
    Compute the Generalized Cross Correlation - Phase Transform (GCC-PHAT) between two signals.
    
    Args:
        sig: Signal to be aligned
        refsig: Reference signal
        fs: Sampling frequency of the signals
    
    Returns:
        tau: Time delay between the signals
        peak: Peak value of the cross-correlation
    """
    # Make sure the length of the signals are the same
    n = sig.shape[0] + refsig.shape[0]
    
    # FFT of the signals
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    
    # Cross-spectral density
    R = SIG * np.conj(REFSIG)
    
    # Phase transform
    R /= np.abs(R)
    
    # Inverse FFT to get cross-correlation
    cc = np.fft.irfft(R, n=n)
    # max_shift = int(n / 2)
    # cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    
    # Find all peaks above the threshold
    if isinstance(thresh, (list, tuple)) and len(thresh) > 1:
        peaks = cfar(np.abs(cc), guard_cells=thresh[0], training_cells=thresh[1], rate_fa=thresh[2])
        peaks = np.array(peaks, dtype=np.int64)
        peaks = peaks[np.abs(cc)[peaks] >= 0.04 * np.max(np.abs(cc))]
    else:
        peaks, _ = find_peaks(np.abs(cc), height=thresh * max(np.abs(cc)))
    
    # Get the peak values
    peak_values = np.abs(cc)[peaks.astype(int)]
    if plot_peaks:
        import matplotlib.pyplot as plt

        # Plot the cross-correlation and peaks
        plt.figure(figsize=(10, 6))
        plt.plot(np.abs(cc), label='Cross-correlation')
        plt.plot(peaks, peak_values, 'x', label='Peaks')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title('Cross-correlation and Peaks')
        plt.legend()
        plt.show()
    
    return peaks/fs, peak_values

def generate_pointcloud(d, audio, chirp_param, thresh=0.1, fov = 180, plot_peaks=True):
    Fs = chirp_param['fs']
    T = chirp_param['T_chirp']
    t = np.linspace(0, T, int(Fs * T))
    chirp_template = chirp(t, f0=chirp_param['f_start'], f1=chirp_param['f_end'], t1=T, method='linear')
    
    # 使用gcc_phat函数寻找峰值
    tau_0, peak_0 = gcc_phat(audio[0, :], chirp_template, fs=Fs, thresh=thresh, plot_peaks=plot_peaks)
    tau_1, peak_1 = gcc_phat(audio[1, :], chirp_template, fs=Fs, thresh=thresh, plot_peaks=plot_peaks)

    cost_matrix = np.abs(tau_0[:, np.newaxis] - tau_1)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    angle = np.rad2deg(np.arcsin((tau_0[row_ind] - tau_1[col_ind]) * 343 / d))

    power_L = peak_0[row_ind]
    power_R = peak_1[col_ind]
    power   = (power_L + power_R) / 2

    distance_L = (tau_0[row_ind] - tau_0[0]) * 343 / 2
    distance_R = (tau_1[col_ind] - tau_1[0]) * 343 / 2
    distance = (distance_L + distance_R) / 2
    pointcloud_rth = np.array([distance, angle, power]).T
    
    min_distance = 0.005 * 343 / 2
    pointcloud_rth = pointcloud_rth[~np.isnan(pointcloud_rth[:, 1])]
    pointcloud_rth = pointcloud_rth[pointcloud_rth[:, 0] > min_distance]
    
    pointcloud_rth = pointcloud_rth[(pointcloud_rth[:, 1] >= -0.5 * fov) & (pointcloud_rth[:, 1] <= 0.5 * fov)]

    
    # Convert polar coordinates to Cartesian coordinates
    x = pointcloud_rth[:, 0] * np.cos(np.deg2rad(pointcloud_rth[:, 1]))
    y = pointcloud_rth[:, 0] * np.sin(np.deg2rad(pointcloud_rth[:, 1]))
    p = pointcloud_rth[:, 2]
    pointcloud = np.vstack((x, y, p)).T
    
    return pointcloud

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples

if __name__ == '__main__':
    # Example usage
    chirp_param = {'fs': 44100, 'T_chirp': 0.003, 'f_start': 20, 'f_end': 20000}
    d = 0.1

    opt_file = 'config/config.yaml'
    opt = Opt(opt_file)
    vis_index = 234
    save_res = False

    dataloader_val = CustomDatasetDataLoader()
    dataloader_val.initialize(opt)
    dataset_val = dataloader_val.load_data()
    dataset_size_val = len(dataloader_val)
    print('#validation clips = %d' % dataset_size_val)

    while True:
        try:
            vis_index = int(input("Enter vis_index (negative number to exit): "))
            if vis_index < 0 or vis_index >= len(dataloader_val.dataset):
                print("Invalid index. Exiting.")
                break

            val_data = dataset_val.dataset.__getitem__(vis_index)
            depth_gt = val_data['depth']
            img = val_data['img'].squeeze().cpu().numpy().transpose(1, 2, 0)
            audio = val_data['audio'].squeeze().cpu().numpy()
            
            # Perform inverse STFT on the audio
            audio_istft = librosa.istft(audio, n_fft=512, win_length=64)

            # Generate point cloud
            audio_istft = normalize(audio_istft)
            pointcloud = generate_pointcloud(d, audio_istft, chirp_param, thresh=0.15, fov=80, plot_peaks=True)
            print(pointcloud.shape)

            # Visualize the results
            fig, axs = plt.subplots(2, 2, figsize=(9, 6))

            # Visualize ground truth depth
            axs[0, 0].imshow(depth_gt.squeeze().cpu().numpy(), cmap='viridis')
            axs[0, 0].set_title('Ground Truth Depth')
            axs[0, 0].axis('off')

            # Visualize the input image
            img = val_data['img'].squeeze().cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            axs[1, 0].imshow(img)
            axs[1, 0].set_title('Input Image')
            axs[1, 0].axis('off')

            audio = audio.mean(axis=0)  # Average the channels to create a 2D array
            axs[0, 1].imshow(audio, aspect='auto', origin='lower', cmap='viridis')
            axs[0, 1].set_title('Audio Spectrogram')
            axs[0, 1].axis('off')

            axs[1, 1].scatter(pointcloud[:, 0], pointcloud[:, 1], c='b', marker='o')
            axs[1, 1].set_xlim([0, 11])
            axs[1, 1].set_ylim([-5, 5])
            axs[1, 1].set_xlabel('x (meters)')
            axs[1, 1].set_ylabel('y (meters)')
            axs[1, 1].set_title('Point Cloud')

            plt.show()

        except ValueError:
            print("Invalid input. Please enter a valid integer index.")