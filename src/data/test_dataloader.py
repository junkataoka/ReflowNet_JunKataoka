import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import pytest
from dataloader import ReflowDataset, generate_dataloader


def test_main():
    geom="../data/processed/train-src-GEOM"
    recipe="../data/processed/train-src-RECIPE"
    heatmap="../data/processed/train-src-HEATMAP"
    dataloader = generate_dataloader(geom, heatmap, recipe, batch_size=34, train=True)
    x, y = next(iter(dataloader))


    geom="../data/processed/train-tar-GEOM"
    recipe="../data/processed/train-tar-RECIPE"
    heatmap="../data/processed/train-tar-HEATMAP"
    dataloader = generate_dataloader(geom, heatmap, recipe, batch_size=34, train=True)
    x, y = next(iter(dataloader))

    geom="../data/processed/test-tar-GEOM"
    recipe="../data/processed/test-tar-RECIPE"
    heatmap="../data/processed/test-tar-HEATMAP"
    dataloader = generate_dataloader(geom, heatmap, recipe, batch_size=1, train=False)
    x, y = next(iter(dataloader))

if __name__ == '__main__':
    test_main()
