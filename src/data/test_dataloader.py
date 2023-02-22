import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import pytest
from dataloader import ReflowDataset, generate_dataloader


def test_main():
    geom="/home/junkataoka/reflownet_ver2/data/processed/train-tar-GEOM"
    recipe="/home/junkataoka/reflownet_ver2/data/processed/train-tar-RECIPE"
    heatmap="/home/junkataoka/reflownet_ver2/data/processed/train-tar-HEATMAP"

    dataloader = generate_dataloader(geom, heatmap, recipe, batch_size=24, train=True, M=452)

    x, y = next(iter(dataloader))


if __name__ == '__main__':
    test_main()
