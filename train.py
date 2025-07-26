from multiprocessing import freeze_support
import os
import shutil
import torch
from tqdm import tqdm
from models.model_builder import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from utils.util import TextWrite, compute_errors
import numpy as np
from models import criterion 
from torch.utils.tensorboard import SummaryWriter
from utils.Opt_ import Opt
from utils.utils_tensorboard import tensorboard_display_input_pred

def create_optimizer(nets, opt):
	if opt.model_baseline == 'fcrn_decoder':
		(net_audio, net_mixfeatUpSample) = nets
		param_groups = [
						{'params': net_audio.parameters(), 'lr': opt.lr_audio},
						{'params': net_mixfeatUpSample.parameters(), 'lr': opt.lr_audio}
						]
	elif opt.model_baseline == 'fcrn_mix_multi_pointnet':
		# (net_audio, net_attention_fusion, net_mixfeatUpSample, net_pointnet, net_CBAM) = nets
		(net_audio, net_attention_fusion, net_mixfeatUpSample, net_pointnet) = nets
		param_groups = [
						{'params': net_audio.parameters(), 'lr': opt.lr_audio},
						{'params': net_attention_fusion.parameters(), 'lr': opt.lr_audio},
						{'params': net_mixfeatUpSample.parameters(), 'lr': opt.lr_audio},
						{'params': net_pointnet.parameters(), 'lr': opt.lr_audio},
						# {'params': net_CBAM.parameters(), 'lr': opt.lr_audio}
						]
	elif opt.model_baseline == 'fcrn_angular_q_mix_multi_pointnet':
		(net_audio, net_attention_fusion, net_mixfeatUpSample, net_pointnet, net_angular_q_encoder) = nets
		param_groups = [
						{'params': net_audio.parameters(), 'lr': opt.lr_audio},
						{'params': net_attention_fusion.parameters(), 'lr': opt.lr_audio},
						{'params': net_mixfeatUpSample.parameters(), 'lr': opt.lr_audio},
						{'params': net_pointnet.parameters(), 'lr': opt.lr_audio},
						{'params': net_angular_q_encoder.parameters(), 'lr': opt.lr_audio}
						]
	else:
		net_audiodepth = nets
		param_groups = [
						{'params': net_audiodepth.parameters(), 'lr': opt.lr_audio},
						]
	
	if opt.optimizer == 'sgd':
		return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
	elif opt.optimizer == 'adam':
		return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor
		
def evaluate(model, loss_criterion, dataset_val, writer, epoch):
	losses = []
	errors = []
	with torch.no_grad():
		for i, val_data in enumerate(dataset_val):
			val_data['audio'] = val_data['audio'].to(opt.device)
			val_data['depth'] = val_data['depth'].to(opt.device)
			if 'pointnet' in opt.model_baseline:
				if opt.model_baseline == 'fcrn_mix_multi_pointnet' or opt.model_baseline == 'fcrn_angular_q_mix_multi_pointnet':
					val_data['raw_pointcloud_1'] = val_data['raw_pointcloud_1'].to(opt.device)
					val_data['raw_pointcloud_2'] = val_data['raw_pointcloud_2'].to(opt.device)
					val_data['raw_pointcloud_3'] = val_data['raw_pointcloud_3'].to(opt.device)
					val_data['queries'] = val_data['queries'].to(opt.device)
				else:
					val_data['raw_pointcloud_2'] = val_data['raw_pointcloud_2'].to(opt.device)
			output = model.forward(val_data)
			# depth_predicted = output['depth_predicted']
			depth_predicted = output['audio_depth']
			depth_gt = output['depth_gt']
			if loss_criterion.__class__.__name__ == 'L1_LPIPS_pointshape_Loss':
				loss = loss_criterion(depth_predicted, depth_gt, output['pointfeats'])
			else:
				loss = loss_criterion(depth_predicted[depth_gt!=0], depth_gt[depth_gt!=0])
			losses.append(loss.item())
			for idx in range(depth_predicted.shape[0]):
				errors.append(compute_errors(depth_gt[idx].cpu().numpy(), 
								depth_predicted[idx].cpu().numpy()))
	
	mean_loss = sum(losses)/len(losses)
	mean_errors = np.array(errors).mean(0)	
	print('Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_errors[1])) 
	val_errors = {}
	val_errors['ABS_REL'], val_errors['RMSE'] = mean_errors[0], mean_errors[1]
	val_errors['DELTA1'] = mean_errors[2] 
	val_errors['DELTA2'] = mean_errors[3]
	val_errors['DELTA3'] = mean_errors[4]
	val_errors['LOG_10'] = mean_errors[5]

	tensorboard_display_input_pred(writer, val_data, depth_predicted, depth_gt, 'Val', epoch)
	return mean_loss, val_errors 

def clear_tensorboard_cache(log_dir):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"TensorBoard cache at {log_dir} has been cleared.")
    else:
        print(f"No TensorBoard cache found at {log_dir}.")


if __name__ == '__main__':
	freeze_support()
	# torch.cuda.empty_cache()

	opt = Opt('config/config.yaml')
	# Save the options to a text file
	opt_file = os.path.join(opt.expr_dir, 'train_config.txt')
	with open(opt_file, 'w') as f:
		for key, value in vars(opt).items():
			f.write(f'{key}: {value}\n')
	if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
		opt.device = torch.device(f"cuda:{opt.gpu_ids[0]}")
	else:
		opt.device = torch.device("cpu")

	loss_criterion = criterion.L1_LPIPS_pointshape_Loss().to(opt.device)

	#### Log the results ####
	loss_list = ['step', 'loss']
	err_list = ['step', 'RMSE', 'ABS_REL', 'DELTA1', 'DELTA2', 'DELTA3', 'LOG_10']

	train_loss_file = TextWrite(os.path.join(opt.expr_dir, 'train_loss.csv'))
	train_loss_file.add_line_csv(loss_list)
	train_loss_file.write_line()

	val_loss_file = TextWrite(os.path.join(opt.expr_dir, 'val_loss.csv'))
	val_loss_file.add_line_csv(loss_list)
	val_loss_file.write_line()

	val_error_file = TextWrite(os.path.join(opt.expr_dir, 'val_error.csv'))
	val_error_file.add_line_csv(err_list)
	val_error_file.write_line()
	
    ##################### 
	
    # network builders
	builder = ModelBuilder(opt=opt)
	# Load batvision model
	if opt.load_pretrained:
		pretrained_ep_num = opt.pretrained_ep_num
		weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir, 
									'audiodepth_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
		print("Using pretrained weights from: ", weights_path)
	# for batvision model
	if opt.model_baseline == 'batvision':
		net_audiodepth = builder.build_batvison_model(opt, weights=weights_path if opt.load_pretrained is True else '')

	elif opt.model_baseline == 'audiodepth':
		net_audiodepth = builder.build_audiodepth(opt.audio_shape, weights=weights_path if opt.load_pretrained is True else '')

	elif opt.model_baseline == 'fcrn_decoder':
		if opt.load_pretrained:
			audiodepth_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir, 
									'audio_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
			mixfeatUpSample_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir,
									'mixfeatUpSample_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
		
		net_audio = builder.build_fcrn_encoder_net(output_feature_dim=256, weights=audiodepth_weights_path if opt.load_pretrained is True else '')
		net_mixfeatUpSample = builder.build_mixfeatUpSample_net(feat_dim=256, output_nc=1, weights=mixfeatUpSample_weights_path if opt.load_pretrained is True else '')
		net_audiodepth = (net_audio, net_mixfeatUpSample)

	elif opt.model_baseline == 'fcrn_mix_multi_pointnet':
		if opt.load_pretrained:
			audiodepth_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir, 
									'audio_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
			attention_fusion_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir, 
									'attention_fusion_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
			mixfeatUpSample_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir,
									'mixfeatUpSample_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
			pointnet_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir,
									'pointnet_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
		
		net_audio = builder.build_fcrn_encoder_net(output_feature_dim=512, weights=audiodepth_weights_path if opt.load_pretrained is True else '')
		net_attention_fusion = builder.build_attention_fusion_net(feature_dim=512, weights=attention_fusion_weights_path if opt.load_pretrained is True else '')
		net_mixfeatUpSample = builder.build_mixfeatUpSample_net(feat_dim=512, output_nc=1, weights=mixfeatUpSample_weights_path if opt.load_pretrained is True else '')
		net_pointnet = builder.build_PointNetfeat_net(global_feat=True, global_feature_dim=512, feature_transform=False, weights=pointnet_weights_path if opt.load_pretrained is True else '')

		net_audiodepth = (net_audio, net_attention_fusion, net_mixfeatUpSample, net_pointnet)
	
	elif opt.model_baseline == 'fcrn_angular_q_mix_multi_pointnet':
		if opt.load_pretrained:
			audiodepth_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir, 
									'audio_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
			attention_fusion_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir, 
									'attention_fusion_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
			mixfeatUpSample_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir,
									'mixfeatUpSample_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
			pointnet_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir,
									'pointnet_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')
			angular_q_encoder_weights_path = os.path.join(opt.checkpoints_dir if pretrained_ep_num is not None else opt.checkpoints_dir,
									'angular_q_encoder_' + opt.dataset + ('_epoch_' + str(pretrained_ep_num) if pretrained_ep_num is not None else '') + '.pth')

		net_audio = builder.build_fcrn_encoder_net(output_feature_dim=256, weights=audiodepth_weights_path if opt.load_pretrained is True else '')
		net_attention_fusion = builder.build_attention_fusion_net(feature_dim=256, weights=attention_fusion_weights_path if opt.load_pretrained is True else '')
		net_mixfeatUpSample = builder.build_mixfeatUpSample_net(feat_dim=256, output_nc=1, weights=mixfeatUpSample_weights_path if opt.load_pretrained is True else '')
		net_pointnet = builder.build_PointNetfeat_net(global_feat=True, global_feature_dim=256, feature_transform=False, weights=pointnet_weights_path if opt.load_pretrained is True else '')
		net_angular_q_encoder = builder.build_angular_q_encoder_net(d_query=256, weights=angular_q_encoder_weights_path if opt.load_pretrained is True else '')
		net_audiodepth = (net_audio, net_attention_fusion, net_mixfeatUpSample, net_pointnet, net_angular_q_encoder)


	model = AudioVisualModel(net_audiodepth, opt)
	model = model.to(opt.device)

	opt.mode = 'train'
	dataloader = CustomDatasetDataLoader()
	dataloader.initialize(opt)
	dataset = dataloader.load_data()
	dataset_size = len(dataloader)
	print('#training clips = %d' % dataset_size)

	
	if opt.validation_on:
		opt.mode = 'test'
		dataloader_val = CustomDatasetDataLoader()
		dataloader_val.initialize(opt)
		dataset_val = dataloader_val.load_data()
		dataset_size_val = len(dataloader_val)
		print('#validation clips = %d' % dataset_size_val)
		opt.mode = 'train'
		
    # create optimizer
	nets = net_audiodepth
	optimizer = create_optimizer(nets, opt)
	# initialization
	total_steps = pretrained_ep_num * dataset_size if opt.load_pretrained else dataset_size  
	batch_loss = []
	best_rmse = float("inf")
	best_loss = float("inf")
	best_absrel = float("inf")
	
	writer = SummaryWriter(log_dir=os.path.join(opt.expr_dir, 'logs'))
	terminal_width = shutil.get_terminal_size().columns
	# clear_tensorboard_cache(opt.expr_dir + 'logs')
	
	for epoch in range(pretrained_ep_num if opt.load_pretrained else 1, opt.niter+1):
		if len(opt.gpu_ids) > 1:
			torch.cuda.synchronize()
		batch_loss = []

		with tqdm(total=dataset_size, ncols=int(0.7*terminal_width)) as t:
			t.set_description('Epoch %d' % epoch)
			for i, data in enumerate(dataset):

				total_steps += opt.batchSize
				data['audio'] = data['audio'].to(opt.device)
				data['depth'] = data['depth'].to(opt.device)
				if 'pointnet' in opt.model_baseline:
					if opt.model_baseline == 'fcrn_mix_multi_pointnet' or opt.model_baseline == 'fcrn_angular_q_mix_multi_pointnet':
						data['raw_pointcloud_1'] = data['raw_pointcloud_1'].to(opt.device)
						data['raw_pointcloud_2'] = data['raw_pointcloud_2'].to(opt.device)
						data['raw_pointcloud_3'] = data['raw_pointcloud_3'].to(opt.device)
						data['queries'] = data['queries'].to(opt.device)

					else:
						data['raw_pointcloud_2'] = data['raw_pointcloud_2'].to(opt.device)

				# forward pass
				model.zero_grad()
				output = model.forward(data)

				# calculate loss
				depth_predicted = output['audio_depth']
				depth_gt = output['depth_gt']
				
				if loss_criterion.__class__.__name__ == 'L1WithLPIPSLoss':
					loss = loss_criterion(depth_predicted, depth_gt)
				elif loss_criterion.__class__.__name__ == 'L1_LPIPS_pointshape_Loss':
					pointfeats = output['pointfeats']
					loss = loss_criterion(depth_predicted, depth_gt, pointfeats)
				else:
					loss = loss_criterion(depth_predicted[depth_gt!=0], depth_gt[depth_gt!=0])

				batch_loss.append(loss.item())

				# update optimizer
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if(total_steps // opt.batchSize % opt.display_freq == 0):
					# print('Display training progress at (epoch %d, steps %d)' % (epoch, total_steps // opt.batchSize))
					avg_loss = sum(batch_loss) / len(batch_loss)
					t.set_postfix(loss=avg_loss)
					# print('Average loss: %.5f' % (avg_loss))
					batch_loss = []
					# print('end of display \n')
					train_loss_file.add_line_csv([total_steps // opt.batchSize, avg_loss])
					train_loss_file.write_line()
					writer.add_scalar('Loss/train', avg_loss, total_steps)

				if(total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on):
					model.eval()
					opt.mode = 'test'
					# print('Display validation results at (epoch %d, steps %d)' % (epoch, total_steps // opt.batchSize))
					val_loss, val_err = evaluate(model, loss_criterion, dataset_val, writer, epoch)

					# print('end of display \n')
					
					model.train()
					opt.mode = 'train'

					# save the model that achieves the smallest validation error
					if (val_err['RMSE'] < best_rmse and val_err['RMSE'] < 0.93) or (val_err['ABS_REL'] < best_absrel and val_err['ABS_REL'] <  0.58):
						best_rmse = val_err['RMSE'] if val_err['RMSE'] < best_rmse else best_rmse
						best_absrel = val_err['ABS_REL'] if val_err['ABS_REL'] < best_absrel else best_absrel
						print('saving the best model (epoch %d) with validation RMSE %.5f\n' % (epoch, val_err['RMSE']))
						if opt.model_baseline == 'batvision':
							torch.save(net_audiodepth.state_dict(), os.path.join(opt.expr_dir, 'audiodepth_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
						elif opt.model_baseline == 'audiodepth':
							torch.save(net_audiodepth.state_dict(), os.path.join(opt.expr_dir, 'audio_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
						elif opt.model_baseline == 'fcrn_mix_multi_pointnet':
							torch.save(net_audio.state_dict(), os.path.join(opt.expr_dir, 'audio_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
							torch.save(net_attention_fusion.state_dict(), os.path.join(opt.expr_dir, 'attention_fusion_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
							torch.save(net_mixfeatUpSample.state_dict(), os.path.join(opt.expr_dir, 'mixfeatUpSample_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
							torch.save(net_pointnet.state_dict(), os.path.join(opt.expr_dir, 'pointnet_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
						elif opt.model_baseline == 'fcrn_decoder':
							torch.save(net_audio.state_dict(), os.path.join(opt.expr_dir, 'audio_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
							torch.save(net_mixfeatUpSample.state_dict(), os.path.join(opt.expr_dir, 'mixfeatUpSample_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
						elif opt.model_baseline == 'fcrn_angular_q_mix_multi_pointnet':
							torch.save(net_audio.state_dict(), os.path.join(opt.expr_dir, 'audio_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
							torch.save(net_attention_fusion.state_dict(), os.path.join(opt.expr_dir, 'attention_fusion_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
							torch.save(net_mixfeatUpSample.state_dict(), os.path.join(opt.expr_dir, 'mixfeatUpSample_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
							torch.save(net_pointnet.state_dict(), os.path.join(opt.expr_dir, 'pointnet_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
							torch.save(net_angular_q_encoder.state_dict(), os.path.join(opt.expr_dir, 'angular_q_encoder_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
					
					#### Logging the values for the val set
					val_loss_file.add_line_csv([total_steps // opt.batchSize, val_loss])
					val_loss_file.write_line()
					writer.add_scalar('Loss/val', val_loss, total_steps)

					err_list = [total_steps // opt.batchSize, \
						val_err['RMSE'], val_err['ABS_REL'], \
						val_err['DELTA1'], val_err['DELTA2'], val_err['DELTA3'], val_err['LOG_10']]
					val_error_file.add_line_csv(err_list)
					val_error_file.write_line()
					writer.add_scalar('Metrics/RMSE', val_err['RMSE'], total_steps)
					writer.add_scalar('Metrics/ABS_REL', val_err['ABS_REL'], total_steps)
					writer.add_scalar('Metrics/DELTA1', val_err['DELTA1'], total_steps)
					writer.add_scalar('Metrics/DELTA2', val_err['DELTA2'], total_steps)
					writer.add_scalar('Metrics/DELTA3', val_err['DELTA3'], total_steps)
				
				t.update(opt.batchSize)

		if epoch % opt.epoch_save_freq == 0:
			print('saving the model at epoch %d' % epoch)
			if opt.model_baseline == 'batvision':
				torch.save(net_audiodepth.state_dict(), os.path.join(opt.expr_dir, 'audiodepth_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
			elif opt.model_baseline == 'audiodepth':
				torch.save(net_audiodepth.state_dict(), os.path.join(opt.expr_dir, 'audiodepth_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
			elif opt.model_baseline == 'fcrn_mix_multi_pointnet':
				torch.save(net_audio.state_dict(), os.path.join(opt.expr_dir, 'audio_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
				torch.save(net_attention_fusion.state_dict(), os.path.join(opt.expr_dir, 'attention_fusion_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
				torch.save(net_mixfeatUpSample.state_dict(), os.path.join(opt.expr_dir, 'mixfeatUpSample_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
				torch.save(net_pointnet.state_dict(), os.path.join(opt.expr_dir, 'pointnet_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
			elif opt.model_baseline == 'fcrn_decoder':
				torch.save(net_audio.state_dict(), os.path.join(opt.expr_dir, 'audio_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
				torch.save(net_mixfeatUpSample.state_dict(), os.path.join(opt.expr_dir, 'mixfeatUpSample_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
			elif opt.model_baseline == 'fcrn_angular_q_mix_multi_pointnet':
				torch.save(net_audio.state_dict(), os.path.join(opt.expr_dir, 'audio_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
				torch.save(net_attention_fusion.state_dict(), os.path.join(opt.expr_dir, 'attention_fusion_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
				torch.save(net_mixfeatUpSample.state_dict(), os.path.join(opt.expr_dir, 'mixfeatUpSample_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
				torch.save(net_pointnet.state_dict(), os.path.join(opt.expr_dir, 'pointnet_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
				torch.save(net_angular_q_encoder.state_dict(), os.path.join(opt.expr_dir, 'angular_q_encoder_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))


		# decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
		if(opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0):
			decrease_learning_rate(optimizer, opt.decay_factor)
			print('decreased learning rate by ', opt.decay_factor)

	writer.close()
