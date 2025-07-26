import os
import torch

from models.PointNet import MixfeatUpSample
from models.fcrn_encoder import FCRN_encoder
from models.unetbaseline_model import define_G
from .networks import  weights_init, \
    SimpleAudioDepthNet

class ModelBuilder():
    def __init__(self, opt=None):
        self.opt = opt

    # builder for audio stream
    def build_audiodepth(self, audio_shape=[2,257,121], weights=''):
        net = SimpleAudioDepthNet(8, audio_shape=audio_shape, audio_feature_length=512)
        net.apply(weights_init)
        if len(weights) > 0:
            # print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights))
        return net

    def build_batvison_model(self, opt, weights=''):
        model = define_G(opt, input_nc = 2, output_nc = 1, ngf = 64, netG = 'unet_128', norm = 'batch',
                                    use_dropout = False, init_type='normal', init_gain=0.02, gpu_ids = opt.gpu_ids)
        print('Model used:', 'unet_128')
        # model.apply(weights_init)
        if len(weights) > 0:
            # print('Loading weights for audio stream')
            model.load_state_dict(torch.load(weights))
        return model

    
    def build_mixfeatUpSample_net(self, feat_dim=512, output_nc=1, weights=''):
        net = MixfeatUpSample(feat_dim, output_nc)
        # net.apply(weights_init)
        if len(weights) > 0:
            net.load_state_dict(torch.load(weights))
        return net
    
    def build_PointNetfeat_net(self, global_feat=True, feature_transform=False, global_feature_dim=512, weights=''):
        from models.PointNet import PointNetfeat
        net = PointNetfeat(global_feat=global_feat, feature_transform=feature_transform, global_feature_dim=global_feature_dim)
        # net.apply(weights_init)
        if len(weights) > 0:
            net.load_state_dict(torch.load(weights))

        return net
    
    def build_attention_fusion_net(self, feature_dim=512, weights=''):
        from models.PointNet import AttentionFusion
        net = AttentionFusion(feature_dim)
        # net.apply(weights_init)
        if len(weights) > 0:
            net.load_state_dict(torch.load(weights))

        return net

    
    def build_fcrn_encoder_net(self, layers=50, output_feature_dim = 512, weights=''):
        net = FCRN_encoder(layers=layers,output_feature_dim=output_feature_dim, pretrained=True)
        if len(weights) > 0:
            if os.path.isfile(weights):
                print("=> loading checkpoint '{}'".format(weights))
                checkpoint = torch.load(weights, map_location=lambda storage, loc: storage.cuda(self.opt.gpu_ids[0] if torch.cuda.is_available() else 'cpu'))
                net.load_state_dict(checkpoint)

        return net
    
    def build_angular_q_encoder_net(self, d_query=512, weights=''):
        from models.fcrn_encoder import AngularQEncoder
        net = AngularQEncoder(d_query=d_query)
        # net.apply(weights_init)
        if len(weights) > 0:
            net.load_state_dict(torch.load(weights))
        return net
