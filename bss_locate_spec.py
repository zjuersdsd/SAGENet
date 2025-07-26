import numpy as np
from scipy.signal import find_peaks
from scipy.fftpack import fft
from scipy.linalg import eigh

def bss_locate_spec(x, fs, d, nsrc, local='GCC-NONLIN', pooling='max', tau_grid=None):
    if x.shape[1] != 2:
        raise ValueError('The input signal must be stereo.')
    if tau_grid is None:
        tau_grid = np.linspace(-d/343, d/343, 181)

    wlen = 1024
    if 'GCC' in local:
        X = stft_multi(x.T, wlen)
        X = X[1:, :, :]
    else:
        hatRxx = qstft(x, fs, wlen, 8, 2)
        hatRxx = np.transpose(hatRxx[:, :, 1:, :], (2, 3, 0, 1))
    f = fs / wlen * np.arange(1, wlen // 2 + 1)

    if local == 'GCC-PHAT':
        spec = phat_spec(X, f, tau_grid)
    elif local == 'GCC-NONLIN':
        c = 343
        alpha = 10 * c / (d * fs)
        spec = nonlin_spec(X, f, alpha, tau_grid)
    elif local == 'MUSIC':
        spec = music_spec(hatRxx, f, tau_grid)
    elif local == 'DS':
        spec = ds_spec(hatRxx, f, tau_grid)
    elif local == 'MVDR':
        spec = mvdr_spec(hatRxx, f, tau_grid)
    elif local == 'DNM':
        spec = dnm_spec(hatRxx, f, d, tau_grid)
    elif local == 'DSW':
        spec = dsw_spec(hatRxx, f, d, tau_grid)
    elif local == 'MVDRW':
        spec = mvdrw_spec(hatRxx, f, d, tau_grid)
    else:
        raise ValueError('Unknown local angular spectrum.')

    if pooling == 'max':
        spec = np.max(np.sum(spec, axis=0), axis=0)
    elif pooling == 'sum':
        spec = np.sum(np.sum(spec, axis=0), axis=0)
    else:
        raise ValueError('Unknown pooling function.')

    peaks, _ = find_peaks(spec, distance=2)
    tau = tau_grid[np.argsort(spec[peaks])[-nsrc:]]

    return tau, spec

def phat_spec(X, f, tau_grid):
    nbin, nfram = X.shape[0], X.shape[1]
    ngrid = len(tau_grid)
    X1, X2 = X[:, :, 0], X[:, :, 1]

    spec = np.zeros((nbin, nfram, ngrid))
    P = X1 * np.conj(X2)
    P /= (np.abs(P) + 1e-10)
    for ind in range(ngrid):
        EXP = np.exp(-2j * np.pi * tau_grid[ind] * f)[:, None]
        spec[:, :, ind] = np.real(P * EXP)

    return spec

def nonlin_spec(X, f, alpha, tau_grid):
    nbin, nfram = X.shape[0], X.shape[1]
    ngrid = len(tau_grid)
    X1, X2 = X[:, :, 0], X[:, :, 1]

    spec = np.zeros((nbin, nfram, ngrid))
    P = X1 * np.conj(X2)
    P /= (np.abs(P) + 1e-10)
    for ind in range(ngrid):
        EXP = np.exp(-2j * np.pi * tau_grid[ind] * f)[:, None]
        with np.errstate(invalid='ignore'):
            spec[:, :, ind] = 1 - np.tanh(alpha * np.real(np.sqrt(2 - 2 * np.real(P * EXP))))

    return spec

def music_spec(hatRxx, f, tau_grid):
    nbin, nfram = hatRxx.shape[0], hatRxx.shape[1]
    ngrid = len(tau_grid)
    R11, R12, R22 = np.real(hatRxx[:, :, 0, 0]), hatRxx[:, :, 0, 1], np.real(hatRxx[:, :, 1, 1])
    TR = R11 + R22
    DET = R11 * R22 - np.abs(R12) ** 2
    lambda_ = 0.5 * (TR + np.sqrt(TR ** 2 - 4 * DET))
    V2 = (lambda_ - R11) / R12

    spec = np.zeros((nbin, nfram, ngrid))
    for ind in range(ngrid):
        EXP = np.exp(-2j * np.pi * tau_grid[ind] * f)[:, None]
        spec[:, :, ind] = 1 / (1 - 0.5 * np.abs(1 + V2 * np.conj(EXP)) ** 2 / (1 + np.abs(V2) ** 2))

    return spec

def ds_spec(hatRxx, f, tau_grid):
    nbin, nfram = hatRxx.shape[0], hatRxx.shape[1]
    ngrid = len(tau_grid)
    R11, R12, R22 = hatRxx[:, :, 0, 0], hatRxx[:, :, 0, 1], hatRxx[:, :, 1, 1]
    TR = np.real(R11 + R22)

    SNR = np.zeros((nbin, nfram, ngrid))
    for ind in range(ngrid):
        EXP = np.exp(-2j * np.pi * tau_grid[ind] * f)[:, None]
        SNR[:, :, ind] = (TR + 2 * np.real(R12 * EXP)) / (TR - 2 * np.real(R12 * EXP))
    spec = SNR

    return spec

def mvdr_spec(hatRxx, f, tau_grid):
    nbin, nfram = hatRxx.shape[0], hatRxx.shape[1]
    ngrid = len(tau_grid)
    R11, R12, R21, R22 = hatRxx[:, :, 0, 0], hatRxx[:, :, 0, 1], hatRxx[:, :, 1, 0], hatRxx[:, :, 1, 1]
    TR = np.real(R11 + R22)

    SNR = np.zeros((nbin, nfram, ngrid))
    for ind in range(ngrid):
        EXP = np.exp(-2j * np.pi * tau_grid[ind] * f)[:, None]
        NUM = np.real(R11 * R22 - R12 * R21) / (TR - 2 * np.real(R12 * EXP))
        SNR[:, :, ind] = NUM / (0.5 * TR - NUM)
    spec = SNR

    return spec

def dnm_spec(hatRxx, f, d, tau_grid):
    nbin, nfram = hatRxx.shape[0], hatRxx.shape[1]
    ngrid = len(tau_grid)
    R11, R12, R21, R22 = np.real(hatRxx[:, :, 0, 0]), hatRxx[:, :, 0, 1], hatRxx[:, :, 1, 0], np.real(hatRxx[:, :, 1, 1])
    c = 343
    SINC = np.sinc(2 * f * d / c)
    SINC2 = SINC ** 2

    vs = np.zeros((nbin, nfram, ngrid))
    vb = np.zeros((nbin, nfram, ngrid))
    for ind in range(ngrid):
        EXP = np.exp(-2j * np.pi * tau_grid[ind] * f)
        P = SINC * EXP
        invA11 = np.sqrt(0.5) / (1 - np.real(P)) * (1 - np.conj(P))
        invA12 = -(1 - P) / (SINC - EXP) * invA11

        DEN = 0.5 / (1 - 2 * np.real(P) + SINC2)
        invL12 = (SINC2 - 1) * DEN
        invL22 = 2 * (1 - np.real(P)) * DEN

        ARA1 = np.abs(invA11) ** 2 * R11 + np.abs(invA12) ** 2 * R22
        ARA2 = ARA1 - 2 * np.real(invA11 * invA12 * R21)
        ARA1 += 2 * np.real(invA11 * np.conj(invA12) * R12)
        vsind = 0.5 * ARA1 + invL12 * ARA2
        vbind = invL22 * ARA2

        neg = (vsind < 0) | (vbind < 0)
        vsind[neg] = 0
        vs[:, :, ind] = vsind
        vb[:, :, ind] = vbind
    spec = vs / vb

    return spec

def dsw_spec(hatRxx, f, d, tau_grid):
    nbin, nfram = hatRxx.shape[0], hatRxx.shape[1]
    ngrid = len(tau_grid)
    R11, R12, R22 = hatRxx[:, :, 0, 0], hatRxx[:, :, 0, 1], hatRxx[:, :, 1, 1]
    TR = np.real(R11 + R22)
    c = 343
    SINC = np.sinc(2 * f * d / c)

    SNR = np.zeros((nbin, nfram, ngrid))
    for ind in range(ngrid):
        EXP = np.exp(-2j * np.pi * tau_grid[ind] * f)[:, None]
        SNR[:, :, ind] = -(1 + SINC) / 2 + (1 - SINC) / 2 * (TR + 2 * np.real(R12 * EXP)) / (TR - 2 * np.real(R12 * EXP))
    spec = SNR

    return spec

def mvdrw_spec(hatRxx, f, d, tau_grid):
    nbin, nfram = hatRxx.shape[0], hatRxx.shape[1]
    ngrid = len(tau_grid)
    R11, R12, R21, R22 = hatRxx[:, :, 0, 0], hatRxx[:, :, 0, 1], hatRxx[:, :, 1, 0], hatRxx[:, :, 1, 1]
    TR = np.real(R11 + R22)
    c = 343
    SINC = np.sinc(2 * f * d / c)

    SNR = np.zeros((nbin, nfram, ngrid))
    for ind in range(ngrid):
        EXP = np.exp(-2j * np.pi * tau_grid[ind] * f)[:, None]
        NUM = np.real(R11 * R22 - R12 * R21) / (TR - 2 * np.real(R12 * EXP))
        SNR[:, :, ind] = -(1 + SINC) / 2 + (1 - SINC) / 2 * NUM / (0.5 * TR - NUM)
    spec = SNR

    return spec

def stft_multi(x, wlen):
    nchan, nsampl = x.shape
    if nchan > nsampl:
        raise ValueError('The signals must be within rows.')
    if wlen % 4 != 0:
        raise ValueError('The window length must be a multiple of 4.')

    win = np.sin((np.arange(0.5, wlen) / wlen) * np.pi)
    nfram = int(np.ceil(nsampl / wlen * 2))
    x = np.pad(x, ((0, 0), (0, nfram * wlen // 2 - nsampl)), mode='constant')
    x = np.pad(x, ((0, 0), (wlen // 4, wlen // 4)), mode='constant')
    swin = np.zeros((nfram + 1) * wlen // 2)
    for t in range(nfram):
        swin[t * wlen // 2:t * wlen // 2 + wlen] += win ** 2
    swin = np.sqrt(wlen * swin)
    nbin = wlen // 2 + 1
    X = np.zeros((nbin, nfram, nchan), dtype=complex)
    for i in range(nchan):
        for t in range(nfram):
            frame = x[i, t * wlen // 2:t * wlen // 2 + wlen] * win / swin[t * wlen // 2:t * wlen // 2 + wlen]
            fframe = fft(frame)
            X[:, t, i] = fframe[:nbin]
    return X

def qstft(x, fs, wlen, lf, lt):
    nsampl, nchan = x.shape
    if nchan > nsampl:
        raise ValueError('The input signal must be in columns.')
    if lf is None:
        lf = 2
    if lt is None:
        lt = 2

    X = stft_multi(x.T, wlen)
    nbin, nfram, nchan = X.shape

    winf = np.hanning(2 * lf - 1)
    wint = np.hanning(2 * lt - 1)
    Cx = np.zeros((nchan, nchan, nbin, nfram), dtype=complex)
    for f in range(nbin):
        for t in range(nfram):
            indf = slice(max(0, f - lf + 1), min(nbin, f + lf))
            indt = slice(max(0, t - lt + 1), min(nfram, t + lt))
            nind = (indf.stop - indf.start) * (indt.stop - indt.start)
            XX = X[indf, indt, :].reshape(nind, nchan).T
            wei = np.outer(winf[indf.start - f + lf - 1:indf.stop - f + lf - 1], wint[indt.start - t + lt - 1:indt.stop - t + lt - 1]).flatten()
            Cx[:, :, f, t] = (XX * wei) @ XX.T / np.sum(wei)
    return Cx, fs / wlen * np.arange(nbin)

def test_bss_locate_spec():
    import soundfile as sf
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt
    # Load the audio file
    x, fs = sf.read('dataset/echoes_navigable/apartment_0/3ms_sweep/0/4.wav')
    
    # Set parameters
    d = 0.1
    nsrc = 2
    local = 'GCC-NONLIN'
    pooling = 'max'
    tau_grid = np.linspace(-d/343, d/343, 181)
    echo_time = [round(0.00 * fs), round(0.06 * fs)]
    raw_signal = x[0:echo_time[0], :]
    x = x[echo_time[0]:echo_time[1], :]

    # Call the function
    # _, spec_raw = bss_locate_spec(raw_signal, fs, d, nsrc, local, pooling, tau_grid)
    tau, spec = bss_locate_spec(x, fs, d, nsrc, local, pooling, tau_grid)

    # Plot the angular spectrum
    plt.figure()
    angles = np.arcsin(tau_grid * 343 / d) * 180 / np.pi
    plt.plot(angles, spec , 'b', label='Angle Spectrum Difference')
    plt.xlabel('degrees')
    plt.ylabel('Angular Spectrum')
    plt.title('Angular Spectrum using bss_locate_spec')
    plt.grid(True)
    plt.show()

    # # Find peaks in the angular spectrum
    # peaks, _ = find_peaks(spec[:len(tau_grid)], height=max(spec)*0.3)

    # # Convert delay times to original signal time points
    # time_points = echo_time[0] / fs + tau_grid[peaks]

    # # Calculate distances
    # ranges = 0.5 * 343 * time_points
    # print('Peak distances (meters):')
    # print(ranges)

    # # Calculate angles
    # angles = np.arcsin(tau_grid[peaks] * 343 / d) * 180 / np.pi

    # # Combine angles and distances into an array
    # points = np.column_stack((angles, ranges))

    # # Convert to Cartesian coordinates
    # x_cart = points[:, 1] * np.cos(np.deg2rad(points[:, 0]))
    # y_cart = points[:, 1] * np.sin(np.deg2rad(points[:, 0]))

    # # Plot the 2D point cloud in Cartesian coordinates
    # plt.figure()
    # plt.scatter(x_cart, y_cart, c='blue', marker='o')
    # plt.xlabel('X (meters)')
    # plt.ylabel('Y (meters)')
    # plt.title('2D Point Cloud in Cartesian Coordinates')
    # plt.grid(True)
    # plt.show()

if __name__ == '__main__':
    test_bss_locate_spec()