import json
import random
import warnings

import torch

from pathlib import Path
from  PIL import Image

from configs import IMAGENET_MEAN, IMAGENET_STD

from torchvision import transforms as T
from torch.utils.data import Dataset, Sampler

class ImageDataset(Dataset):
    def __init__(
        self, 
        specs_file, 
        image_size, 
        transform = None, 
        training = False, 
        formats = None
    ):
        
        specs = self.load_specs(Path(specs_file))

        self.images, self.labels = self.list_data_instances(
            specs['class_roots'], formats
        )

        self.class_names = specs['class_names']

        self.transform = (
            transform if transform else self.set_transform(image_size, training)
        )

    @staticmethod
    def load_specs(specs_file):
        """
        Load specs from a JSON file
        """
        
        with open(specs_file, 'r') as file:
            specs = json.load(file)
        
        if 'class_names' not in specs.keys() or 'class_roots' not in specs.keys():
            raise ValueError('requires specs in a JSON with the keys class_names and class_roots')

        if len(specs['class_names']) != len(specs['class_roots']):
            raise ValueError('The number of class names does not equal with the nuber of class root directories')

        return specs

    @staticmethod
    def list_data_instances(class_roots, formats):
        """
        Returns image path and encoded label
        """
        if formats is None:
            formats = {',png', '.jpg', '.jpeg', '.bmp'}

        images, labels = [], []
        for class_id, class_root in enumerate(class_roots):
            class_images = [
                str(img_path)
                for img_path in sorted (Path(class_root).glob('*'))
                if img_path.is_file() and (img_path.suffix.lower() in formats)
            ]
            images += class_images
            labels += len(class_images) * [class_id]
            
        if len(labels) == 0:
            warnings.warn(UserWarning(
                'No images were found in the specified directory. The dataset will be empty'
            ))
        
        return images, labels

    @staticmethod
    def set_transform(image_size, training):

        return (
            T.Compose([
                T.Resize([image_size, image_size]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
            if training
            else T.Compose([
                T.Resize([image_size, image_size]),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        )
    
    def __getitem__(self, item):
        image = self.transform(Image.open(self.images[item]).convert('RGB'))
        label = self.labels[item]

        return image, label

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def number_of_classes(self):
        return len(self.class_names)


class FewShotBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        n_way,
        n_shot,
        n_query,
        n_task
    ):
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_task = n_task

        self.items_per_label = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self):
        return self.n_task

    def __iter__(self):
        for _ in range(self.n_task):
            yield torch.cat(
                [
                    torch.tensor(random.sample(
                        self.items_per_label[label], self.n_shot + self.n_query
                    ))
                    for label in random.sample(self.items_per_label.keys(), self.n_way)
                ]
            ).tolist()
    
    def collate_fn(self, input_data):

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot+self.n_query, *all_images.shape[1:])
        )

        true_class_ids = list({x[1] for x in input_data})

        all_labels = torch.tensor([true_class_ids.index(x[1]) for x in input_data])
        all_labels = all_labels.reshape(
            (self.n_way, self.n_shot + self.n_query)
        )

        support_images = all_images[:, :self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        support_labels = all_labels[:, :self.n_shot].flatten()

        query_images = all_images[:, self.n_shot:].reshape(
            (-1, *all_images.shape[2:])
        )
        query_labels = all_labels[:, self.n_shot:].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids
        )

