import torch
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Note: You should write a DataLoader suitable for your own Dataset!!!
class SimpleVideoDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.video_dir = os.path.join(self.root, 'data')
        videos = sorted(os.listdir(self.video_dir))
        self.video_folders = [os.path.join(self.video_dir, video) for video in videos]
        
    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, index):
        video_folder = self.video_folders[index]
        images = sorted(os.listdir(video_folder))
        image_list = [os.path.join(video_folder, image) for image in images]
        text = video_folder.split('/')[-1]
        prompt = text.replace('_', ' ')
        
        video = [Image.open(image).convert('RGB').resize((576, 320)) for image in image_list]
        video = [transforms.ToTensor()(image) for image in video]
        video = np.stack(video, axis=1)
        video = torch.from_numpy(np.ascontiguousarray(video)).float()

        # normalize
        video = video * 2. - 1.

        return {"video": video, "prompt": prompt}

if __name__ == '__main__':
    train_dataset = SimpleVideoDataset(root="./")
    print(train_dataset.__len__())

    train_data = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False)
    # B C H W
    for i, data in enumerate(train_data):
        print(i)
        print(data['video'].shape)
        print(data['prompt'])