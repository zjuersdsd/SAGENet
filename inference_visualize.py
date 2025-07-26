import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from models.audioVisual_model import AudioVisualModel
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from utils.Opt_ import Opt
from models.model_builder import ModelBuilder
from multiprocessing import freeze_support
from train import evaluate
from PIL import Image



if __name__ == '__main__':
    freeze_support()
    vis_index = 480
    vis_mode = 'Auto'
    save_res = False
    
    opt_file = 'config/config.yaml'
    opt = Opt(opt_file)
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        opt.device = torch.device(f"cuda:{opt.gpu_ids[0]}")
    else:
        opt.device = torch.device("cpu")
    
    builder = ModelBuilder(opt)
    opt.mode = 'test'
    checkpoint_ep_num = opt.checkpoint_ep_num
    
    # Load batvision model
    weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir, 
                        'audiodepth_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')
    print('Model baseline: ', opt.model_baseline)
    print(f'Loading weights from {weights_path}')
    
    if opt.model_baseline == 'batvision':
        net_audiodepth = builder.build_batvison_model(opt, weights=weights_path)

    elif opt.model_baseline == 'audiodepth':
        net_audiodepth = builder.build_audiodepth(opt.audio_shape, weights=weights_path)

    elif opt.model_baseline == 'audiodepth_mix_raw_pointnet':
        audiodepth_weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir, 
								'audio_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')
        attention_fusion_weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir, 
								'attention_fusion_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')
        mixfeatUpSample_weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir,
								'mixfeatUpSample_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')
        pointnet_weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir,
								'pointnet_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')
		
        net_audio = builder.build_audiodepth(opt.audio_shape, weights=audiodepth_weights_path)
        net_attention_fusion = builder.build_attention_fusion_net(feature_dim=512, weights=attention_fusion_weights_path)
        net_mixfeatUpSample = builder.build_mixfeatUpSample_net(feat_dim=512, output_nc=1, weights=mixfeatUpSample_weights_path)    
        net_pointnet = builder.build_PointNetfeat_net(global_feat=True, feature_transform=False, weights=pointnet_weights_path)
        net_audiodepth = (net_audio, net_attention_fusion, net_mixfeatUpSample, net_pointnet)

    elif opt.model_baseline == 'fcrn_decoder':
        mixfeatUpSample_weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir,
                                'mixfeatUpSample_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')
        net_mixfeatUpSample = builder.build_mixfeatUpSample_net(feat_dim=512, output_nc=1, weights=mixfeatUpSample_weights_path)
        net_audiodepth = net_mixfeatUpSample
    
    elif opt.model_baseline == 'fcrn_angular_q_mix_multi_pointnet':
        audiodepth_weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir, 
								'audio_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')
        attention_fusion_weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir, 
								'attention_fusion_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')
        mixfeatUpSample_weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir,
								'mixfeatUpSample_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')
        pointnet_weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir,
								'pointnet_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')
        angular_q_encoder_weights_path = os.path.join(opt.test_checkpoints_dir if checkpoint_ep_num is not None else opt.test_checkpoints_dir,
								'angular_q_encoder_' + opt.dataset + ('_epoch_' + str(checkpoint_ep_num) if checkpoint_ep_num is not None else '') + '.pth')

        net_audio = builder.build_fcrn_encoder_net(output_feature_dim=256, weights=audiodepth_weights_path )
        net_attention_fusion = builder.build_attention_fusion_net(feature_dim=256, weights=attention_fusion_weights_path )
        net_mixfeatUpSample = builder.build_mixfeatUpSample_net(feat_dim=256, output_nc=1, weights=mixfeatUpSample_weights_path )
        net_pointnet = builder.build_PointNetfeat_net(global_feat=True, global_feature_dim=256, feature_transform=False, weights=pointnet_weights_path)
        net_angular_q_encoder = builder.build_angular_q_encoder_net(d_query=256, weights=angular_q_encoder_weights_path)

        net_audiodepth = (net_audio, net_attention_fusion, net_mixfeatUpSample, net_pointnet, net_angular_q_encoder)

    # construct our audio-visual model
    model = AudioVisualModel(net_audiodepth, opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    # model.to('cuda')
    model.eval()

    
    dataloader_val = CustomDatasetDataLoader()
    dataloader_val.initialize(opt)
    dataset_val = dataloader_val.load_data()
    dataset_size_val = len(dataloader_val)
    print('#validation clips = %d' % dataset_size_val)

    while True:
        try:
            if vis_mode != 'Auto':
                vis_index = int(input("Enter vis_index (negative number to exit): "))
            else:
                vis_index = vis_index + 1
            if vis_index < 0 or vis_index >= len(dataloader_val.dataset):
                print("Invalid index. Exiting.")
                break
            
            # # dataset_val.dataset[0]
            val_data = dataset_val.dataset.__getitem__(vis_index)
            # Ensure the input tensor has the correct number of dimensions
            for key in val_data:
                if isinstance(val_data[key], torch.Tensor) :
                    val_data[key] = val_data[key].unsqueeze(0)
            
            with torch.no_grad():
                try:
                    output = model.forward(val_data)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
                depth_gt = output['depth_gt']
                audio_depth = output['audio_depth']

                audio = val_data['audio']
                if 'pointnet' in opt.model_baseline:
                    pointcloud = output['raw_pointcloud']


            # Visualize the results
            fig, axs = plt.subplots(2, 3, figsize=(9, 6))

            # Visualize ground truth depth
            axs[0, 0].imshow(depth_gt.squeeze().cpu().numpy(), cmap='jet')
            axs[0, 0].set_title('Ground Truth Depth')
            axs[0, 0].axis('off')

            # Visualize audio depth
            audio_depth = audio_depth.squeeze().cpu().numpy()
            axs[0, 1].imshow(audio_depth, cmap='jet')
            axs[0, 1].set_title('Audio Depth')
            axs[0, 1].axis('off')

            # Visualize the input image
            img = val_data['img'].squeeze().cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            img = Image.fromarray((img * 255).astype(np.uint8))
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
            img = np.array(img) / 255.0
            axs[1, 0].imshow(img)
            axs[1, 0].set_title('Input Image')
            axs[1, 0].axis('off')

            # Visualize the original audio
            if opt.raw_audio:
                audio = audio.squeeze().cpu().numpy()
                axs[1, 1].plot(audio[0, :], label='Channel 1')
                axs[1, 1].plot(audio[1, :], label='Channel 2')
                axs[1, 1].set_title('Original Audio')
                axs[1, 1].legend()
                axs[1, 1].axis('off')
            else:
                audio = audio.squeeze().cpu().numpy()
                audio_L = audio[0, :, :] # Left channel
                audio_R = audio[1, :, :]  

                axs[1, 1].imshow(audio_L, aspect='auto', origin='lower', cmap='viridis')
                axs[1, 1].set_title('Audio Spectrogram (Left Channel)')
                axs[1, 1].axis('off')

                axs[1, 2].imshow(audio_R, aspect='auto', origin='lower', cmap='viridis')
                axs[1, 2].set_title('Audio Spectrogram (Right Channel)')
                axs[1, 2].axis('off')

            # Visualize the point cloud
            if 'pointnet' in opt.model_baseline:
                pointcloud = pointcloud.squeeze().cpu().numpy().transpose(1, 0)
                axs[0, 2].scatter(pointcloud[:, 0], pointcloud[:, 1], c='b', marker='o')
                axs[0, 2].set_title('Point Cloud')
                axs[0, 2].axis('on')
                axs[0, 2].set_xlim([0, 11])
                axs[0, 2].set_ylim([-5, 5])
                axs[0, 2].set_xlabel('x (meters)')
                axs[0, 2].set_ylabel('y (meters)')

            fig.suptitle(f'Visualization for index {vis_index}')
            fig.tight_layout()
            fig.show()
            # print("Press Space to continue...")
            while True:
                if plt.waitforbuttonpress():
                    break
            plt.close(fig)
            # plt.show()

            if save_res:
                # Save the visualizations
                output_dir = 'visualization_temp'
                os.makedirs(output_dir, exist_ok=True)

                # Save ground truth depth
                plt.imsave(os.path.join(output_dir, f'depth_gt_{vis_index}.png'), depth_gt.squeeze().cpu().numpy(), cmap='viridis')

                # Save audio depth
                plt.imsave(os.path.join(output_dir, f'audio_depth_{vis_index}.png'), audio_depth.squeeze().cpu().numpy(), cmap='viridis')

                # Save input image
                plt.imsave(os.path.join(output_dir, f'input_image_{vis_index}.png'), img)

                # Save original audio or audio spectrogram
                if opt.raw_audio:
                    plt.figure()
                    plt.plot(audio[0, :], label='Channel 1')
                    plt.plot(audio[1, :], label='Channel 2')
                    plt.title('Original Audio')
                    plt.legend()
                    plt.axis('off')
                    plt.savefig(os.path.join(output_dir, f'original_audio_{vis_index}.png'))
                    plt.close()
                else:
                    plt.imsave(os.path.join(output_dir, f'audio_spectrogram_L_{vis_index}.png'), audio_L, cmap='viridis')
                    plt.imsave(os.path.join(output_dir, f'audio_spectrogram_R_{vis_index}.png'), audio_R, cmap='viridis')

        except ValueError:
            print("Invalid input. Please enter a valid integer index.")

    