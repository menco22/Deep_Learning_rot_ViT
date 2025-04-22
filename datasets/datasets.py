from utils.functions import *
from utils.imports import *

#used for the val rotation
class customRotation(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]['image']
        cls_label = self.dataset[index]['label']

        if self.transform:
            image = self.transform(image)

        rotation_label = random.choice([0, 1, 2, 3])
        image_rotated = rotate_img(image, rotation_label)

        rotation_label = torch.tensor(rotation_label).long()

        return image, image_rotated, rotation_label, torch.tensor(cls_label).long()


#1 image and 1 cls-label, 4 rot-images and 4 rot-labels, used for train
class customRotation2(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]['image']
        cls_label = self.dataset[index]['label']

        if self.transform:
            image = self.transform(image)

        images = [image for _ in range(4)]
        cls_labels = [cls_label for _ in range(4)]
        rot_images = [rotate_img(image, rot) for rot in range(4)]
        rot_labels = [torch.tensor(rot).long() for rot in range(4)]

        return images, rot_images, rot_labels, cls_labels
    

#used for val of a streamed dataset
class StreamingRotation(torch.utils.data.IterableDataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        for item in self.dataset:
            image = item['image']
            cls_label = item['label']

            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            if self.transform:
                image = self.transform(image)

            rotation_label = random.choice([0, 1, 2, 3])
            image_rotated = rotate_img(image, rotation_label)

            rotation_label = torch.tensor(rotation_label).long()

            yield image, image_rotated, rotation_label, torch.tensor(cls_label).long()


#used for train in streamed dataset
class StreamingRotation2(torch.utils.data.IterableDataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __iter__(self):
        for item in self.dataset:
            image = item['image']
            cls_label = item['label']

            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            images = [image for _ in range(4)]
            cls_labels = [cls_label for _ in range(4)]
            rot_images = [rotate_img(image, rot) for rot in range(4)]
            rot_labels = [torch.tensor(rot).long() for rot in range(4)]
            
            yield images, rot_images, rot_labels, cls_labels