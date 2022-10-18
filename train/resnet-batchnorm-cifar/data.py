import torch
import torchvision.transforms as T
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

def get_loaders(device_id=0):
    device = 'cuda:%d' % device_id
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]
    pre_p = [SimpleRGBImageDecoder()]
    post_p = [
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
    aug_p = [
        RandomHorizontalFlip(),
        RandomTranslate(padding=4),
        Cutout(12, tuple(map(int, CIFAR_MEAN))),
    ]

    train_aug_loader = Loader(f'/tmp/cifar_train.beton',
                          batch_size=100,
                          num_workers=8,
                          order=OrderOption.RANDOM,
                          drop_last=True,
                          pipelines={'image': pre_p+aug_p+post_p,
                                     'label': label_pipeline})
    train_noaug_loader = Loader(f'/tmp/cifar_train.beton',
                          batch_size=500,
                          num_workers=8,
                          order=OrderOption.RANDOM,
                          drop_last=True,
                          pipelines={'image': pre_p+post_p,
                                     'label': label_pipeline})
    test_loader = Loader(f'/tmp/cifar_test.beton',
                         batch_size=1000,
                         num_workers=8,
                         order=OrderOption.SEQUENTIAL,
                         drop_last=False,
                         pipelines={'image': pre_p+post_p,
                                    'label': label_pipeline})
    return train_aug_loader, train_noaug_loader, test_loader

