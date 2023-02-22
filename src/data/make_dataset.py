# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import os
import torch


def generate_target(root_recipe, num_area, num_geom, num_recipe, remove_geom):

    res = []
    for j in range(num_geom):
        if j not in remove_geom:
            out = np.empty((num_recipe, num_area))
            for i in range(num_recipe):
                for k in range(num_area):
                    recipe_path = f"recipe_{i+1}_{k+1}.csv"
                    recipe_img = np.genfromtxt(
                        os.path.join(root_recipe, recipe_path), delimiter=","
                        )
                    out[i, k] = recipe_img[0, 0]
            res.append(out)

    res = np.concatenate(res, axis=0)

    return res


def generate_input(root_geom, root_heatmap, seq_len, num_geom, num_recipe, remove_geom):

    res = []

    for j in range(num_geom):
        if j not in remove_geom:

            out = np.zeros((num_recipe, seq_len, 2, 50, 50))
            die_path = f"M{j+1}_DIE.csv"
            pcb_path = f"M{j+1}_PCB.csv"
            trace_path = f"M{j+1}_Substrate.csv"

            die_img = np.genfromtxt(
                os.path.join(root_geom, die_path), delimiter=","
                )
            pcb_img = np.genfromtxt(
                os.path.join(root_geom, pcb_path), delimiter=","
                )
            trace_img = np.genfromtxt(
                os.path.join(root_geom, trace_path), delimiter=","
                )

            for i in range(num_recipe):
                for k in range(seq_len):
                    heatmap_path = f"IMG_{j+1}_{i+1}_{k+1}.csv"
                    heatmap_img = np.genfromtxt(
                        os.path.join(root_heatmap, heatmap_path), delimiter=","
                        )
                    out[i, k, 0] += die_img
                    out[i, k, 0] += pcb_img
                    out[i, k, 0] += trace_img
                    out[i, k, 1] = heatmap_img

            res.append(out)

    res = np.concatenate(res, axis=0)
    return res


def generate_tardomain_input(root_geom, root_heatmap, seq_len, geom_id, num_recipe):

    out = np.zeros((num_recipe, seq_len, 2, 50, 50))

    for i in range(num_recipe):
        for k in range(seq_len):

            die_path = f"M{geom_id+1}_DIE.csv"
            pcb_path = f"M{geom_id+1}_PCB.csv"
            trace_path = f"M{geom_id+1}_Substrate.csv"
            heatmap_path = f"IMG_{geom_id+1}_{i+1}_{k+1}.csv"

            die_img = np.genfromtxt(
                os.path.join(root_geom, die_path), delimiter=","
                )
            pcb_img = np.genfromtxt(
                os.path.join(root_geom, pcb_path), delimiter=","
                )
            trace_img = np.genfromtxt(
                os.path.join(root_geom, trace_path), delimiter=","
                )
            heatmap_img = np.genfromtxt(
                os.path.join(root_heatmap, heatmap_path), delimiter=","
                )
            out[i, k, 0] += die_img
            out[i, k, 0] += pcb_img
            out[i, k, 0] += trace_img
            out[i, k, 1] = heatmap_img

    return out


def SaveInputAsText(array, output_filepath, domain, train):

    b, s, c, w, h = array.shape
    assert c == 2

    for i in range(b):
        for j in range(s):
            for k in range(c):
                if k == 0:
                    imgtype = f"{train}-{domain}-GEOM"
                else:
                    imgtype = f"{train}-{domain}-HEATMAP"

                img_path = os.path.join(output_filepath, imgtype)
                if not os.path.exists(img_path):
                    os.mkdir(img_path)
                np.savetxt(os.path.join(img_path, f'{i}-{j}.txt'), array[i, j, k], fmt='%f')


def SaveOutputAsText(recipe, output_filepath, domain, train):

    b, a = recipe.shape
    imgtype = f"{train}-{domain}-RECIPE"
    img_path = os.path.join(output_filepath, imgtype)
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    for i in range(b):
        np.savetxt(os.path.join(img_path, f'{i}.txt'), recipe[i], fmt='%f')


def MinMaxNormalize(input_tensor):
    b, s, c, w, h = input_tensor.shape
    res_max = torch.zeros(c).cuda()
    res_min = torch.ones(c) * 1e+10
    res_min = res_min.cuda()
    for i in range(input_tensor.shape[0]):

        for j in range(input_tensor.shape[1]):
            minimum = torch.min(input_tensor[i, j, :].view(c, -1), dim=1)[0]
            maximum = torch.max(input_tensor[i, j, :].view(c, -1), dim=1)[0]

            res_max = torch.FloatTensor([max(res_max[i], maximum[i]) for i in range(len(res_max))])
            res_min = torch.FloatTensor([min(res_min[i], minimum[i]) for i in range(len(res_min))])

    return res_min, res_max


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--rm_geom', multiple=True, type=int, default=[None])
@click.option('--target_geom', type=int, default=0)
@click.option('--test_recipe', type=int, default=0)
def main(input_filepath, output_filepath, rm_geom, target_geom, test_recipe):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Create src domain output data
    a = generate_target(
        root_recipe=os.path.join(input_filepath, "recipe_simulation"),
        num_area=7, num_geom=12, num_recipe=81, remove_geom=rm_geom)

    target_tensor = torch.tensor(a).cpu().numpy()
    SaveOutputAsText(target_tensor, output_filepath, "src", "train")

    # Create tar domain output data for testing
    a = generate_target(
        root_recipe=os.path.join(input_filepath, "recipe_experiment"),
        num_area=7, num_geom=12, num_recipe=3,
        remove_geom=[i for i in range(12) if i != target_geom]
        )
    target_tensor = torch.tensor(a).cpu()
    target_test = torch.index_select(
        target_tensor, dim=0, index=torch.tensor([test_recipe])
        )
    SaveOutputAsText(target_test, output_filepath, "tar", "test")

    # Create tar domain output data for training
    target_train = torch.index_select(
        target_tensor, dim=0,
        index=torch.tensor([i for i in range(len(target_tensor)) if i != test_recipe])
        )
    SaveOutputAsText(target_train, output_filepath, "tar", "train")

    # Create src domain input data
    a = generate_input(
        root_geom=os.path.join(input_filepath, "geo_img"),
        root_heatmap=os.path.join(input_filepath, "heatmap_simulation"),
        seq_len=15, num_geom=12, num_recipe=81,
        remove_geom=rm_geom)
    input_tensor = torch.FloatTensor(a).cuda()
    res_min, res_max = MinMaxNormalize(input_tensor)
    res_min, res_max = res_min.cuda(), res_max.cuda()
    src_input = (input_tensor - res_min[None, None, :, None, None]) / \
        (res_max[None, None, :, None, None] - res_min[None, None, :, None, None])
    src_input = src_input.cpu().numpy()
    SaveInputAsText(src_input, output_filepath, "src", "train")

    # Create tar domain input data
    a = generate_tardomain_input(
        root_geom=os.path.join(input_filepath, "geo_img"),
        root_heatmap=os.path.join(input_filepath, "heatmap_experiment"),
        seq_len=15, geom_id=target_geom, num_recipe=3)

    input_tensor = torch.FloatTensor(a).cuda()
    input_tensor = (input_tensor - res_min[None, None, :, None, None]) / \
        (res_max[None, None, :, None, None] - res_min[None, None, :, None, None])

    # Split tar domain input data (training)
    tar_input_train = torch.index_select(
        input_tensor, dim=0, 
        index=torch.tensor([i for i in range(len(input_tensor)) if i != test_recipe]).cuda()
        )
    tar_input_train = tar_input_train.cpu().numpy()
    SaveInputAsText(tar_input_train, output_filepath, "tar", "train")

    # Split tar domain input data (testing)
    tar_input_test = torch.index_select(
        input_tensor, dim=0,
        index=torch.tensor([test_recipe]).cuda()
        )
    tar_input_test = tar_input_test.cpu().numpy()
    SaveInputAsText(tar_input_test, output_filepath, "tar", "test")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
