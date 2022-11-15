from pathlib import Path
from fewshotdataset import ImageDataset, FewShotBatchSampler

from configs import CAR_SPECS_DIR
from configs import IMAGE_SIZE
from configs import N_WAY, N_SHOT, N_QUERY, N_WORKERS, N_TRAINING_EPISODES, N_VALIDATION_TASK

from torch.utils.data import DataLoader

def CARS(split, image_size, **kwargs):
    specs_file = Path(CAR_SPECS_DIR) / '{}.json'.format(split)
    
    if specs_file.suffix != '.json':
        raise ValueError('requires specs in a JSON file')
    elif specs_file.is_file():
        return ImageDataset(specs_file=specs_file, image_size=image_size, **kwargs)
    else:
        raise ValueError(
            'couldn\'t find specs file {} in {}'.format(specs_file.name, Path(CAR_SPECS_DIR))
        )

def generate_loader(
    split,
    image_size=IMAGE_SIZE,
    n_way=N_WAY,
    n_shot=N_SHOT,
    n_query=N_QUERY,
    n_task=N_VALIDATION_TASK,
    n_workers=N_WORKERS,
    **kwargs
):
    training = True if split=='train' else False
    dataset = CARS(split=split, image_size=image_size, training=training, **kwargs)

    batch_sampler =  FewShotBatchSampler(
        dataset, n_way, n_shot, n_query, n_task
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=batch_sampler.collate_fn
    ) 