from matplotlib import pyplot as plt
# from data_loader.audio_visual_dataset import AudioVisualDataset
import yaml


class Opt:
    def __init__(self, opt_file):
        self.load_from_file(opt_file)

    def load_from_file(self, opt_file):
        with open(opt_file, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(self, key, value)

def visualize_data(dataset, index):
    data = dataset[index]
    img = data['img']
    depth = data['depth']
    audio = data['audio']

    # Convert tensor to numpy array for visualization
    img_np = img.permute(1, 2, 0).numpy()
    depth_np = depth.squeeze(0).numpy()
    audio_np = audio.numpy()

    # Plot the images and audio spectrogram
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))

    axs[0, 0].imshow(img_np)
    axs[0, 0].set_title('RGB Image')

    axs[0, 1].imshow(depth_np, cmap='gray')
    axs[0, 1].set_title('Depth Image')

    axs[1, 0].imshow(audio_np[0], aspect='auto', origin='lower', cmap='inferno')
    axs[1, 0].set_title('Audio Spectrogram - Channel 1')

    axs[1, 1].imshow(audio_np[1], aspect='auto', origin='lower', cmap='inferno')
    axs[1, 1].set_title('Audio Spectrogram - Channel 2')

    # axs[1, 0].plot(audio_np[0])
    # axs[1, 0].set_title('Audio Signal - Channel 1')

    # axs[1, 1].plot(audio_np[1])
    # axs[1, 1].set_title('Audio Signal - Channel 2')


    plt.show()

if __name__ == '__main__':
    opt_file = 'config/config.yaml'
    opt = Opt(opt_file)
    print(opt.chirp_params['f_start'])
    # opt.audio_length = 1
    # opt.audio_shape = [2, 257, 2757]
    # opt.raw_audio = True
    
    # dataset = AudioVisualDataset()
    # dataset.initialize(opt)  # Assuming there is an initialize method
    # print(dataset.audio_type)
    # dataset.__getitem__(3334)
    # visualize_data(dataset, 3334)  # Visualize the first data point