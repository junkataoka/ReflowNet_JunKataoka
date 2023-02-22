import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import pytest
from dataloader import ReflowDataset


def test_main():
    dataset = ReflowDataset(
        geom="/home/junkataoka/reflownet_ver2/data/processed/train-src-GEOM",
        recipe="/home/junkataoka/reflownet_ver2/data/processed/train-src-RECIPE",
        heatmap="/home/junkataoka/reflownet_ver2/data/processed/train-src-HEATMAP",
    )
    x, y = dataset[0]

if __name__ == '__main__':
    test_main()
