import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.datasets import ImageFolder

def preparing_dataset(path):
    target_size = (128, 128)
    data_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(path, transform=data_transforms)
    n_samples = len(dataset)
    val_frac = 0.2
    val_size = int(n_samples * val_frac)
    train_size = n_samples - val_size
    print('Number of points in train dataset - ',train_size)
    print('Number of points in test dataset - ',val_size)
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    batch_size = 32
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True
                                   ,drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size,drop_last=True)
    return train_loader, val_loader

# path = '/content/asl_alphabet_train/asl_alphabet_train/'
# train_loader, val_loader = preparing_dataset(path)

