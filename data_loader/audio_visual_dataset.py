import os.path
import shutil
import torch.nn as nn
import librosa
import h5py
import random
import math
import numpy as np
import glob
import torch
import pickle
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm

from GCC_PCL import generate_pointcloud
from bss_locate_spec import bss_locate_spec
from scipy.signal import find_peaks


def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples

def generate_spectrogram(audioL, audioR, winl=32):
    channel_1_spec = librosa.stft(audioL, n_fft=512, win_length=winl)
    channel_2_spec = librosa.stft(audioR, n_fft=512, win_length=winl)
    
    spectro_two_channel = np.concatenate((np.expand_dims(np.abs(channel_1_spec), axis=0), np.expand_dims(np.abs(channel_2_spec), axis=0)), axis=0)
    # print("spectro_two_channel.shape = ", spectro_two_channel.shape)
    return spectro_two_channel

def generate_batvision_spectrogram(audioL, audioR, winl=64):
    # For batvision
    
    channel_1_spec = librosa.stft(audioL, n_fft=512, win_length=winl, hop_length=64//4)
    channel_2_spec = librosa.stft(audioR, n_fft=512, win_length=winl, hop_length=64//4)
    
    spectro_two_channel = np.concatenate((np.expand_dims(np.abs(channel_1_spec), axis=0), np.expand_dims(np.abs(channel_2_spec), axis=0)), axis=0)
    return spectro_two_channel

def process_image(rgb, augment):
    if augment:
        enhancer = ImageEnhance.Brightness(rgb)
        rgb = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(rgb)
        rgb = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Contrast(rgb)
        rgb = enhancer.enhance(random.random()*0.6 + 0.7)
    return rgb


def parse_all_data(root_path, scenes):
    data_idx_all = []
    with open(root_path, 'rb') as f:
        data_dict = pickle.load(f)
    for scene in scenes:
        print(scene)
        data_idx_all += ['/'.join([scene, str(loc), str(ori)]) \
            for (loc,ori) in list(data_dict[scene].keys())]
        print(len(data_idx_all))    
    
    return data_idx_all, data_dict
  
    
class AudioVisualDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.audio_cache = {}
        if self.opt.dataset == 'replica':
            self.data_idx, self.data = parse_all_data(self.opt.img_path, 
                                                        self.opt.scenes[opt.mode])   
            self.win_length = 64
        self.normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), self.normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.base_audio_path = self.opt.audio_path
        if self.opt.dataset == 'replica':
            self.audio_type = '3ms_sweep'
        
        self.d = 0.1
        self.tau_grid = np.linspace(-self.d/343, self.d/343, 181)
        self.local = 'GCC-NONLIN'
        self.pooling = 'max'

        self.angles = np.arcsin(self.tau_grid * 343 / self.d) * 180 / np.pi
        self.num_queries = 15

        # Load entire dataset into memory
        self.audio_cache = {}
        terminal_width = shutil.get_terminal_size().columns
        with tqdm(total=len(self.data_idx), ncols=int(0.7*terminal_width)) as pbar:
            for idx in self.data_idx:
                scene, loc, orn = idx.split('/')
                cache_key = f"{scene}_{loc}_{orn}_" + self.opt.model_baseline
                audio_path = os.path.join(self.base_audio_path, scene, self.audio_type, orn, loc+'.wav')
                audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate, mono=False, duration=self.opt.audio_length)
                if self.opt.audio_normalize:
                    audio = normalize(audio)
                if self.opt.model_baseline == 'batvision':
                    # For batvision
                    if opt.raw_audio:
                        audio_spec_both = torch.FloatTensor([audio[0,:], audio[1,:]])
                    else:
                        audio_spec_both = torch.FloatTensor(generate_batvision_spectrogram(audio[0, 0:3113], audio[1, 0:3113], self.win_length))
                else:
                    audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0,:], audio[1,:], self.win_length))  

                # Generate point cloud
                pointcloud_1 = torch.FloatTensor(generate_pointcloud(self.d, audio, self.opt.chirp_params, thresh=0.15, fov=80, plot_peaks=False)) 
                pointcloud_1 = pointcloud_1.permute(1, 0)
                pointcloud_2 = torch.FloatTensor(generate_pointcloud(self.d, audio, self.opt.chirp_params, thresh=0.2, fov=80, plot_peaks=False)) 
                pointcloud_2 = pointcloud_2.permute(1, 0)
                pointcloud_3 = torch.FloatTensor(generate_pointcloud(self.d, audio, self.opt.chirp_params, thresh=0.25, fov=80, plot_peaks=False)) 
                pointcloud_3 = pointcloud_3.permute(1, 0)

                _, spec = bss_locate_spec(audio.T, self.opt.audio_sampling_rate, self.d, 1, self.local, self.pooling, self.tau_grid)
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

                self.audio_cache[cache_key] = (audio_spec_both, pointcloud_1, pointcloud_2, pointcloud_3, spec, queries)
                pbar.update(1)

                
    def __getitem__(self, index):
        #load audio
        scene, loc, orn = self.data_idx[index].split('/')
        
        cache_key = f"{scene}_{loc}_{orn}_" + self.opt.model_baseline
        audio_spec_both, pointcloud_1, pointcloud_2, pointcloud_3, spec, queries = self.audio_cache[cache_key]

        #get the rgb image and depth image
        img = Image.fromarray(self.data[scene][(int(loc),int(orn))][('rgb')]).convert('RGB')
        
        if self.opt.mode == "train":
           img = process_image(img, self.opt.enable_img_augmentation)

        if self.opt.image_transform:
            img = self.vision_transform(img)

        depth = torch.FloatTensor(self.data[scene][(int(loc),int(orn))][('depth')])
        depth = depth.unsqueeze(0)

        
        if self.opt.mode == "train":
            if self.opt.enable_cropping:
                RESOLUTION = self.opt.image_resolution
                w_offset =  RESOLUTION - 128
                h_offset = RESOLUTION - 128
                left = random.randrange(0, w_offset + 1)
                upper = random.randrange(0, h_offset + 1)
                img = img[:, left:left+128,upper:upper+128]
                depth = depth[:, left:left+128,upper:upper+128]
        
        return {'img': img, 'depth':depth, 'audio':audio_spec_both, 'raw_pointcloud_1':pointcloud_1, 'raw_pointcloud_2':pointcloud_2, 'raw_pointcloud_3':pointcloud_3, 'queries':queries, 'spec': spec}

    def __len__(self):
        return len(self.data_idx)

    def name(self):
        return 'AudioVisualDataset'