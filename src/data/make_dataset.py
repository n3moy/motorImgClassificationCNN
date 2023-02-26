import torch
from torchvision import transforms
import PIL
from pathlib import Path

from img_handler import MotorDataset


def stratified_split(dataset : torch.utils.data.Dataset, fraction, random_state=None):
    import random 
    from collections import defaultdict

    labels = [torch.argmax(i[1]).item() for i in dataset]
    if random_state: 
        random.seed(random_state)
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_inds, second_set_inds = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_inds.extend(random_indices_sample)
        second_set_inds.extend(set(indices) - set(random_indices_sample))

    return second_set_inds, first_set_inds


def train_test_split(folder_path, out_path, fraction, n_replications):
    '''
    Splits dataset into augmented train and test sets, saves to out_path

    '''
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([10, 180], interpolation=PIL.Image.BILINEAR)
    ])

    dataset = MotorDataset(folder_path)
    train_inds, test_inds = stratified_split(dataset, fraction)

    for ind in train_inds:
        img, _, img_id = dataset[ind]
        img.save(out_path / 'train' / img_id)
        for i_repl in range(n_replications):
            img_id = f'aug{i_repl}.'.join(img_id.split('.'))
            aug_img = aug_transform(img)
            aug_img.save(out_path / 'train' / img_id)

    for ind in test_inds:
        img, _, img_id = dataset[ind]
        img.save(out_path / 'test' / img_id)


if __name__ == '__main__':
    folder_path = Path(__file__).parent.parent.parent / 'data' / 'interim' / 'data_named'
    out_path = Path(__file__).parent.parent.parent / 'data' / 'interim' / 'augmented'
    fraction = 0.2
    n_replications = 10

    train_test_split(folder_path, out_path, fraction, n_replications)
