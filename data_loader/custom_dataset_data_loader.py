#!/usr/bin/env python

import torch.utils.data

def CreateDataset(opt):
    dataset = None
    if opt.dataset == 'replica':
        from data_loader.audio_visual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
        dataset.initialize(opt)
    elif opt.dataset == 'batvision_v1':
        from data_loader.BatvisionV1_Dataset import BatvisionV1Dataset
        dataset = BatvisionV1Dataset(opt)
        dataset.initialize()
    return dataset

def collate_fn(batch, opt):
    audio = torch.stack([item['audio'] for item in batch])
    depth = torch.stack([item['depth'] for item in batch])
    image = torch.stack([item['img'] for item in batch])
    
    # Find the maximum length of pointclouds in the batch
    max_length = max(max(item['raw_pointcloud_1'].shape[1], item['raw_pointcloud_2'].shape[1], item['raw_pointcloud_3'].shape[1]) for item in batch)
    padded_pointclouds_1 = [torch.nn.functional.pad(item['raw_pointcloud_1'], (0, max_length - item['raw_pointcloud_1'].shape[1], 0, 0)) for item in batch]
    padded_pointclouds_2 = [torch.nn.functional.pad(item['raw_pointcloud_2'], (0, max_length - item['raw_pointcloud_2'].shape[1], 0, 0)) for item in batch]
    padded_pointclouds_3 = [torch.nn.functional.pad(item['raw_pointcloud_3'], (0, max_length - item['raw_pointcloud_3'].shape[1], 0, 0)) for item in batch]

    pointcloud_1 = torch.stack(padded_pointclouds_1)
    pointcloud_2 = torch.stack(padded_pointclouds_2)
    pointcloud_3 = torch.stack(padded_pointclouds_3)

    queries = torch.stack([item['queries'] for item in batch])

    return {'img':image, 'audio': audio, 'depth': depth, 'raw_pointcloud_1': pointcloud_1, 'raw_pointcloud_2': pointcloud_2, 'raw_pointcloud_3': pointcloud_3, 'queries': queries}

        
        
class CustomDatasetDataLoader():
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.dataset = CreateDataset(opt)
        shuff = False
        if opt.mode == "train":
            print('Shuffling the dataset....')
            shuff= True
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=shuff,
            num_workers=int(opt.nThreads),
            drop_last=True,
            collate_fn=lambda batch: collate_fn(batch, opt),
            pin_memory=True)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
