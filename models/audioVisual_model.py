import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F


class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_audio = nets
        if self.opt.model_baseline == 'audiodepth_mix_raw_pointnet' or self.opt.model_baseline == 'fcrn_mix_raw_pointnet' or self.opt.model_baseline == 'fcrn_mix_pointransformer':
            self.net_audio, self.net_attention_fusion, self.net_mixfeatUpSample, self.net_pointnet = nets
        elif self.opt.model_baseline == 'fcrn_mix_multi_pointnet':
            self.net_audio, self.net_attention_fusion, self.net_mixfeatUpSample, self.net_pointnet = nets
        elif self.opt.model_baseline == 'fcrn_angular_q_mix_multi_pointnet':
            self.net_audio, self.net_attention_fusion, self.net_mixfeatUpSample, self.net_pointnet, self.angularqencoder = nets
        elif self.opt.model_baseline == 'fcrn_decoder':
            self.net_audio, self.net_mixfeatUpSample = nets

        

    def forward(self, input, volatile=False):
        rgb_input = input['img']
        audio_input = input['audio']
        depth_gt = input['depth']

        if self.opt.model_baseline == 'batvision':
            audio_depth = self.net_audio(audio_input)
        elif self.opt.model_baseline == 'audiodepth':
            audio_depth, audio_feat = self.net_audio(audio_input)

        elif self.opt.model_baseline == 'fcrn_decoder':
            audio_feat = self.net_audio(audio_input)
            audio_depth = self.net_mixfeatUpSample(audio_feat)

        elif self.opt.model_baseline == 'fcrn_mix_multi_pointnet':
            raw_pointcloud_1, raw_pointcloud_2, raw_pointcloud_3  = input['raw_pointcloud_1'], input['raw_pointcloud_2'], input['raw_pointcloud_3']
            audio_feat = self.net_audio(audio_input)

            pointfeat_1, _, _ = self.net_pointnet(raw_pointcloud_1)
            pointfeat_2, _, _ = self.net_pointnet(raw_pointcloud_2)
            pointfeat_3, _, _ = self.net_pointnet(raw_pointcloud_3)



            fused_feat = self.net_attention_fusion(pointfeat_2, audio_feat)
            audio_depth = self.net_mixfeatUpSample(fused_feat)
        
        elif self.opt.model_baseline == 'fcrn_angular_q_mix_multi_pointnet':
            raw_pointcloud_1, raw_pointcloud_2, raw_pointcloud_3  = input['raw_pointcloud_1'], input['raw_pointcloud_2'], input['raw_pointcloud_3']
            queries = input['queries']
            audio_feat = self.net_audio(audio_input)
            SPEC_SA_feat = self.angularqencoder(queries, audio_feat.squeeze(-1).squeeze(-1))

            if self.opt.mode == 'train':
                pointfeat_1, _, _ = self.net_pointnet(raw_pointcloud_1)
                pointfeat_2, _, _ = self.net_pointnet(raw_pointcloud_2)
                pointfeat_3, _, _ = self.net_pointnet(raw_pointcloud_3)
            else:
                pointfeat_2, _, _ = self.net_pointnet(raw_pointcloud_1)
                pointfeat_3, pointfeat_1 = pointfeat_2, pointfeat_2

            fused_feat = self.net_attention_fusion(pointfeat_2, SPEC_SA_feat.unsqueeze(-1).unsqueeze(-1))
            audio_depth = self.net_mixfeatUpSample(fused_feat)

        if 'pointnet' in self.opt.model_baseline:
            if self.opt.model_baseline == 'fcrn_mix_multi_pointnet' or self.opt.model_baseline == 'fcrn_angular_q_mix_multi_pointnet':
                output =  { 'audio_depth': audio_depth * self.opt.max_depth,
                            'img': rgb_input,
                            'audio': audio_input,
                            'depth_gt': depth_gt,
                            'raw_pointcloud': raw_pointcloud_2,
                            'pointfeats': [pointfeat_1, pointfeat_2, pointfeat_3]}
            else:
                output =  { 'audio_depth': audio_depth * self.opt.max_depth,
                            'img': rgb_input,
                            'audio': audio_input,
                            'depth_gt': depth_gt,
                            'raw_pointcloud': raw_pointcloud_2}
        else:
            output =  { 'audio_depth': audio_depth if self.opt.dataset == 'batvision' else audio_depth * self.opt.max_depth,
                        'img': rgb_input,
                        'audio': audio_input,
                        'depth_gt': depth_gt}
        return output
