import importlib
from torchvision.transforms import Compose
from typing import Union
from torch import nn
from abc import ABCMeta


def get_callable(class_path):
    if callable(class_path):
        return class_path
    module_n, class_n = class_path.rsplit(".", 1)
    class_callable = getattr(importlib.import_module(module_n), class_n)
    return class_callable


def instantiate(config, class_key='class_path', load_class_key='load_class_path', *extra_init_args, **extra_init_kwargs):
    if isinstance(config, ABCMeta):
        return config(*extra_init_args, **extra_init_kwargs)
    elif type(config) is list:
        return [instantiate(elem, *extra_init_args, **extra_init_kwargs) for elem in config]
    elif type(config) is dict:
        if class_key in config or load_class_key in config:
            if ('init_args' in config) and (type(config['init_args']) is dict):
                for key in config['init_args']:
                    if type(config['init_args'][key]) is dict and ((class_key in config['init_args'][key]) or (load_class_key in config['init_args'][key])):
                        config['init_args'][key] = instantiate(config['init_args'][key])
            if class_key in config:
                return get_callable(config[class_key])(*extra_init_args, **extra_init_kwargs, **config.get('init_args', {}))
            elif load_class_key in config:
                module = get_callable(config[load_class_key]).load_from_checkpoint(*extra_init_args, **extra_init_kwargs, **config.get('init_args', {}))
                if 'attr' in config:
                    for attribute in config['attr'].split('.'):
                        module = getattr(module, attribute)
                return module
        for key in config:
            if type(config[key]) is dict:
                config[key] = instantiate(config[key])
        return config
    else:
        return config


def create_runtime_inheritance_dataset(class_callable):
    class DynamicDataset(class_callable):
        def __init__(self, getitem_transform: Union[Compose, nn.Module], *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert getitem_transform is not None
            self.getitem_transforms = getitem_transform

        def __getitem__(self, index):
            return self.getitem_transforms(super().__getitem__(index))

    DynamicDataset.__name__ = class_callable.__name__
    return DynamicDataset
