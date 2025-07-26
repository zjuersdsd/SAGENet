import os
import shutil
import librosa
import torch
import pandas as pd
import numpy as np
import torchaudio
import cv2
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

from GCC_PCL import generate_pointcloud
from bss_locate_spec import bss_locate_spec
from scipy.signal import find_peaks

# Custom transforms 
class MinMaxNorm(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        assert isinstance(min, (float, tuple)) and isinstance(max, (float, tuple))
        self.min = torch.tensor(min)
        self.max = torch.tensor(max)
        
    def forward(self, tensor):
        if tensor.shape[0] == 2:
            norm_tensor_c0 = (tensor[0,...] - self.min[0]) / (self.max[0] - self.min[0])
            norm_tensor_c1 = (tensor[1,...] - self.min[1]) / (self.max[1] - self.min[1])
            norm_tensor = torch.concatenate([norm_tensor_c0.unsqueeze(0), norm_tensor_c1.unsqueeze(0)], dim = 0)

        else:
            norm_tensor = (tensor - self.min) / (self.max - self.min)

        return norm_tensor 
    
def get_transform(cfg, convert =  False, depth_norm = False):
    # Create list of transform to apply to data
    transform_list = []

    if convert:
        # Convert data to Tensor type
        transform_list += [transforms.ToTensor()]

    if 'resize' in cfg.preprocess:
        # Resize
        transform_list.append(transforms.Resize((cfg.image_resolution,cfg.image_resolution)))

    if depth_norm:
        # MinMax depth normalization
        max_depth_dataset = cfg.max_depth 
        min_depth_dataset = 0.0
        transform_list += [MinMaxNorm(min = min_depth_dataset, max = max_depth_dataset)]

    return transforms.Compose(transform_list)

class BatvisionV1Dataset(Dataset):
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.root_dir = 'dataset/Batvision-Dataset/BatVision'
        self.audio_format = 'spectrogram'
        annotation_file = 'filtered_' + cfg.mode + '.csv'
        self.instances = pd.read_csv(os.path.join(self.root_dir, annotation_file))
        self.initialize_flag = False

        self.d = 0.1
        self.tau_grid = np.linspace(-self.d/343, self.d/343, 181)
        self.local = 'GCC-NONLIN'
        self.pooling = 'max'

        self.angles = np.arcsin(self.tau_grid * 343 / self.d) * 180 / np.pi
        self.num_queries = 15
            
    def __len__(self):
        return len(self.instances)
    
    def initialize(self):
        self.initialize_flag = True
        self.audio_cache = {}
        terminal_width = shutil.get_terminal_size().columns
        with tqdm(total=len(self.instances), ncols=int(0.7*terminal_width)) as pbar:
            for idx in range(len(self.instances)):
                # Access instance 
                instance = self.instances.iloc[idx]
                
                # Load path
                depth_path = os.path.join(self.root_dir,instance['depth path'])
                audio_path_left = os.path.join(self.root_dir,instance['audio path left'])
                audio_path_right = os.path.join(self.root_dir,instance['audio path right'])
                cam_img_path = os.path.join(self.root_dir, instance['camera path left'])
                
                # Load depth map
                depth = np.load(depth_path)
                # Set nan value to 0
                depth = np.nan_to_num(depth)
                depth[depth == -np.inf] = 0
                depth[depth == np.inf] = 0
                
                depth = depth / 1000 # to go from mm to m
                depth[depth > self.cfg.max_depth] = self.cfg.max_depth 
                depth[depth < 0.0] = 0.0

                # Transform 
                depth_transform = get_transform(self.cfg, convert =  True, depth_norm = self.cfg.depth_norm)
                gt_depth = depth_transform(depth)
                
                ## Audio
                # Load audio binaural waveform
                waveform_left = np.load(audio_path_left).astype(np.float32)
                waveform_right = np.load(audio_path_right).astype(np.float32)
                waveform = np.stack((waveform_left,waveform_right))

                # Cut audio to fit max depth
                if self.cfg.max_depth:
                    cut = int((2*self.cfg.max_depth / 343) * self.cfg.audio_sampling_rate)
                    waveform = waveform[:, 970:2646+970]

                # Spectrogram
                audio2return = self.generate_spectrogram(waveform[0, :], waveform[1, :], winl=64)
                audio2return = audio2return[:, :, :self.cfg.audio_shape[2]]


                # Generate point cloud
                pointcloud_1 = torch.FloatTensor(generate_pointcloud(self.d, waveform, self.cfg.chirp_params, thresh=0.15, fov=80, plot_peaks=False)) # 0.06 for vis, 0.15 for train
                pointcloud_1 = pointcloud_1.permute(1, 0)
                pointcloud_2 = torch.FloatTensor(generate_pointcloud(self.d, waveform, self.cfg.chirp_params, thresh=0.2, fov=80, plot_peaks=False)) # 0.04 for vis, 0.2 for train
                pointcloud_2 = pointcloud_2.permute(1, 0)
                pointcloud_3 = torch.FloatTensor(generate_pointcloud(self.d, waveform, self.cfg.chirp_params, thresh=0.25, fov=80, plot_peaks=False)) # 0.05 for vis, 0.25 for train
                pointcloud_3 = pointcloud_3.permute(1, 0)

                _, spec = bss_locate_spec(waveform.T, self.cfg.audio_sampling_rate, self.d, 1, self.local, self.pooling, self.tau_grid)
                peaks, _ = find_peaks(spec - min(spec))
                if len(peaks) > 15:
                    top_peaks = np.argsort(spec[peaks])[-15:]
                else:
                    top_peaks = np.arange(len(peaks))

                queries = np.zeros((self.num_queries, 2))
                queries[:len(top_peaks), 0] = self.angles[peaks][top_peaks]
                queries[:len(top_peaks), 1] = spec[peaks][top_peaks]
                queries = np.sort(queries)
                spec = torch.FloatTensor(spec)
                queries = torch.FloatTensor(queries)

                ## Camera
                # Load camera image
                cam_img = cv2.imread(cam_img_path)
                cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
                cam_img = np.transpose(cam_img, (2,0,1))
                cam_img = torch.from_numpy(cam_img)

                # return audio2return, gt_depth
                # cam_img = torch.FloatTensor(cam_img)
                gt_depth = torch.FloatTensor(gt_depth)
                audio2return = torch.FloatTensor(audio2return)
                pointcloud_1 = torch.FloatTensor(pointcloud_1)
                pointcloud_2 = torch.FloatTensor(pointcloud_2)
                pointcloud_3 = torch.FloatTensor(pointcloud_3)
                queries = torch.FloatTensor(queries)
                spec = torch.FloatTensor(spec)

                if self.cfg.model_baseline == 'batvision':
                    spec_transform =  get_transform(self.cfg, convert = False)
                    audio2return = spec_transform(audio2return)
                
                # Cache audio
                self.audio_cache[idx] = ( cam_img, gt_depth, audio2return, pointcloud_1, pointcloud_2, pointcloud_3, queries, spec)
                self.audio_cache[audio_path_right] = waveform_right
                
                pbar.update(1)

    def __getitem__(self, idx):
        # Access instance
        if self.initialize_flag == True:
            ( cam_img, gt_depth, audio2return, pointcloud_1, pointcloud_2, pointcloud_3, queries, spec) = self.audio_cache[idx]
            return {'img': cam_img, 'depth':gt_depth, 'audio':audio2return, 'raw_pointcloud_1':pointcloud_1, 'raw_pointcloud_2':pointcloud_2, 'raw_pointcloud_3':pointcloud_3, 'queries':queries, 'spec': spec}
        else:
            # Access instance 
            instance = self.instances.iloc[idx]
            
            # Load path
            depth_path = os.path.join(self.root_dir,instance['depth path'])
            audio_path_left = os.path.join(self.root_dir,instance['audio path left'])
            audio_path_right = os.path.join(self.root_dir,instance['audio path right'])
            cam_img_path = os.path.join(self.root_dir, instance['camera path left'])
            
            ## Depth
            # Load depth map
            depth = np.load(depth_path)

            # Set nan value to 0
            depth = np.nan_to_num(depth)
            depth[depth == -np.inf] = 0
            depth[depth == np.inf] = 0
            
            depth = depth / 1000 # to go from mm to m
            depth[depth > self.cfg.max_depth] = self.cfg.max_depth 
            depth[depth < 0.0] = 0.0
            
            # Transform 
            depth_transform = get_transform(self.cfg, convert =  True, depth_norm = self.cfg.depth_norm)
            gt_depth = depth_transform(depth)
            
            ## Audio
            # Load audio binaural waveform
            waveform_left = np.load(audio_path_left).astype(np.float32)
            waveform_right = np.load(audio_path_right).astype(np.float32)
            waveform = np.stack((waveform_left,waveform_right))

            # STFT parameters for full length audio
            win_length = 200 
            n_fft = 400
            hop_length = 100
            sr = 44100

            # Cut audio to fit max depth
            if self.cfg.max_depth:
                cut = int((2*self.cfg.max_depth / 343) * sr)
                waveform = waveform[:, 970:cut+970]

            # Spectrogram
            audio2return = self.generate_spectrogram(waveform[0, :], waveform[1, :], winl=64)
            audio2return = audio2return[:, :, :self.cfg.audio_shape[2]]

            # Generate point cloud
            pointcloud_1 = torch.FloatTensor(generate_pointcloud(self.d, waveform, self.cfg.chirp_params, thresh=0.15, fov=80, plot_peaks=False)) # 0.06 for vis, 0.15 for train
            pointcloud_1 = pointcloud_1.permute(1, 0)
            pointcloud_2 = torch.FloatTensor(generate_pointcloud(self.d, waveform, self.cfg.chirp_params, thresh=0.2, fov=80, plot_peaks=False)) # 0.04 for vis, 0.2 for train
            pointcloud_2 = pointcloud_2.permute(1, 0)
            pointcloud_3 = torch.FloatTensor(generate_pointcloud(self.d, waveform, self.cfg.chirp_params, thresh=0.25, fov=80, plot_peaks=False)) # 0.05 for vis, 0.25 for train
            pointcloud_3 = pointcloud_3.permute(1, 0)

            _, spec = bss_locate_spec(waveform.T, self.cfg.audio_sampling_rate, self.d, 1, self.local, self.pooling, self.tau_grid)
            peaks, _ = find_peaks(spec - min(spec))
            if len(peaks) > 15:
                top_peaks = np.argsort(spec[peaks])[-15:]
            else:
                top_peaks = np.arange(len(peaks))

            queries = np.zeros((self.num_queries, 2))
            queries[:len(top_peaks), 0] = self.angles[peaks][top_peaks]
            queries[:len(top_peaks), 1] = spec[peaks][top_peaks]
            queries = np.sort(queries)
            spec = torch.FloatTensor(spec)
            queries = torch.FloatTensor(queries)

            ## Camera
            # Load camera image
            cam_img = cv2.imread(cam_img_path)
            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            cam_img = np.transpose(cam_img, (2,0,1))
            cam_img = torch.from_numpy(cam_img)

            gt_depth = torch.FloatTensor(gt_depth)
            audio2return = torch.FloatTensor(audio2return)
            pointcloud_1 = torch.FloatTensor(pointcloud_1)
            pointcloud_2 = torch.FloatTensor(pointcloud_2)
            pointcloud_3 = torch.FloatTensor(pointcloud_3)
            queries = torch.FloatTensor(queries)
            spec = torch.FloatTensor(spec)

            if self.cfg.model_baseline == 'batvision':
                spec_transform =  get_transform(self.cfg, convert = False)
                audio2return = spec_transform(audio2return)

            return {'img': cam_img, 'depth':gt_depth, 'audio':audio2return, 'raw_pointcloud_1':pointcloud_1, 'raw_pointcloud_2':pointcloud_2, 'raw_pointcloud_3':pointcloud_3, 'queries':queries, 'spec': spec}

    # audio transformation
    def _get_spectrogram(self, waveform, n_fft = 400, power = 1.0, win_length = 400, hop_length=100): 

        spectrogram = T.Spectrogram(
          n_fft=n_fft,
          win_length=win_length,
          power=power,
          hop_length=hop_length
        )
        #db = T.AmplitudeToDB(stype = 'magnitude') # better results without dB conversion
        return spectrogram(waveform)
    
    def generate_spectrogram(self, audioL, audioR, winl=32):
        channel_1_spec = librosa.stft(audioL, n_fft=512, win_length=winl)
        channel_2_spec = librosa.stft(audioR, n_fft=512, win_length=winl)
    
        spectro_two_channel = np.concatenate((np.expand_dims(np.abs(channel_1_spec), axis=0), np.expand_dims(np.abs(channel_2_spec), axis=0)), axis=0)
        # print("spectro_two_channel.shape = ", spectro_two_channel.shape)
        return spectro_two_channel
    

if __name__ == '__main__':
    from utils.Opt_ import Opt
    cfg = Opt('config/config.yaml')
    cfg.mode = 'val'
    dataset = BatvisionV1Dataset(cfg)
    dataset.initialize()
    print(len(dataset))
    val_data = dataset[5]
    print(val_data['img'].shape)
    print(val_data['depth'].shape)
    print(val_data['audio'].shape)
    print(val_data['raw_pointcloud_1'].shape)
    print(val_data['raw_pointcloud_2'].shape)
    print(val_data['raw_pointcloud_3'].shape)
    print(val_data['queries'].shape)
    print(val_data['spec'].shape)
    print('Done')
    # visualize_data(dataset, 0)
    import matplotlib.pyplot as plt

    # Visualize the data
    def visualize_data(val_data):
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        # Plot camera image
        axs[0, 0].imshow(val_data['img'].permute(1, 2, 0).numpy())
        axs[0, 0].set_title('Camera Image')

        # Plot depth map
        axs[0, 1].imshow(val_data['depth'][0].numpy(), cmap='viridis')
        axs[0, 1].set_title('Depth Map')

        # Plot audio spectrogram
        axs[1, 0].imshow(val_data['audio'][0].numpy(), aspect='auto', origin='lower')
        axs[1, 0].set_title('Audio Spectrogram - Channel 1')
        axs[1, 1].imshow(val_data['audio'][1].numpy(), aspect='auto', origin='lower')
        axs[1, 1].set_title('Audio Spectrogram - Channel 2')

        # Plot point clouds
        
        axs[2, 0].set_xlim([0, 11])
        axs[2, 0].set_ylim([-5, 5])
        axs[2, 0].set_title('Point Cloud 1')
        axs[2, 0].scatter(val_data['raw_pointcloud_1'][0, :], val_data['raw_pointcloud_1'][1, :], s=70)
        
        axs[2, 1].set_xlim([0, 11])
        axs[2, 1].set_ylim([-5, 5])
        axs[2, 1].set_title('Point Cloud 2')
        axs[2, 1].scatter(val_data['raw_pointcloud_3'][0, :], val_data['raw_pointcloud_3'][1, :], s=70)

        plt.show()

    visualize_data(val_data)
    print(val_data['raw_pointcloud_2'].T)
    