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
                    recipe_img = np.genfromtxt(os.path.join(root_recipe, recipe_path), delimiter=",")
                    out[i, k] = recipe_img[0, 0]
            res.append(out)


    res = np.concatenate(res, axis=0)

    return res


def generate_input(root_geom, root_heatmap, seq_len, num_geom, num_recipe, remove_geom):

    res = []

    for j in range(num_geom):
        if j not in remove_geom:
            out = np.empty((num_recipe, seq_len, 4, 50, 50))
            die_path = f"M{j+1}_DIE.csv"
            pcb_path = f"M{j+1}_PCB.csv"
            trace_path = f"M{j+1}_Substrate.csv"
            die_img = np.genfromtxt(os.path.join(root_geom, die_path), delimiter=",")
            pcb_img = np.genfromtxt(os.path.join(root_geom, pcb_path), delimiter=",")
            trace_img = np.genfromtxt(os.path.join(root_geom, trace_path), delimiter=",")
            print(die_img.max())

            for i in range(num_recipe):
                for k in range(seq_len):
                    heatmap_path = f"IMG_{j+1}_{i+1}_{k+1}.csv"
                    heatmap_img = np.genfromtxt(os.path.join(root_heatmap, heatmap_path), delimiter=",")
                    out[i, k, 0] = die_img
                    out[i, k, 1] = pcb_img
                    out[i, k, 2] = trace_img
                    out[i, k, 3] = heatmap_img
            res.append(out)
    
    res = np.concatenate(res, axis=0)
    return res

def generate_tardomain_input(root_geom, root_heatmap, seq_len, geom_id, num_recipe):

    res = []

    out = np.empty((num_recipe, seq_len, 4, 50, 50))

    for i in range(num_recipe):
        for k in range(seq_len):

            die_path = f"M{geom_id+1}_DIE.csv"
            pcb_path = f"M{geom_id+1}_PCB.csv"
            trace_path = f"M{geom_id+1}_Substrate.csv"
            heatmap_path = f"IMG_{geom_id+1}_{i+1}_{k+1}.csv"

            die_img = np.genfromtxt(os.path.join(root_geom, die_path), delimiter=",")
            pcb_img = np.genfromtxt(os.path.join(root_geom, pcb_path), delimiter=",")
            trace_img = np.genfromtxt(os.path.join(root_geom, trace_path), delimiter=",")
            heatmap_img = np.genfromtxt(os.path.join(root_heatmap, heatmap_path), delimiter=",")
            out[i, k, 0] = die_img
            out[i, k, 1] = pcb_img
            out[i, k, 2] = trace_img
            out[i, k, 3] = heatmap_img

        res.append(out)


    res = np.concatenate(res, axis=0)

    return res
    
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    remove_geom_list = []

    a = generate_target(root_recipe=os.path.join(input_filepath,"recipe_simulation"), num_area=7, num_geom=12, num_recipe=81, remove_geom=remove_geom_list)
    target_tensor = torch.tensor(a).cuda()
    torch.save(target_tensor, os.path.join(output_filepath,"source_target.pt"))

    a = generate_target(root_recipe=os.path.join(input_filepath, "recipe_experiment"), num_area=7, num_geom=12, num_recipe=3, remove_geom=[i for i in range(12) if i != 0])
    target_tensor = torch.tensor(a).cuda()
    torch.save(target_tensor, os.path.join(output_filepath, "target_target.pt"))

    target_target_cv1_train = torch.index_select(target_tensor, dim=0, index=torch.tensor([0,1]).cuda())
    target_target_cv1_test = torch.index_select(target_tensor, dim=0, index=torch.tensor([2]).cuda())
    torch.save(target_target_cv1_train, os.path.join(output_filepath, "target_target_cv1_train.pt"))
    torch.save(target_target_cv1_test, os.path.join(output_filepath, "target_target_cv1_test.pt"))

    target_target_cv2_train = torch.index_select(target_tensor, dim=0, index=torch.tensor([0,2]).cuda())
    target_target_cv2_test = torch.index_select(target_tensor, dim=0, index=torch.tensor([1]).cuda())
    torch.save(target_target_cv2_train, os.path.join(output_filepath, "target_target_cv2_train.pt"))
    torch.save(target_target_cv2_test, os.path.join(output_filepath, "target_target_cv2_test.pt"))

    target_target_cv3_train = torch.index_select(target_tensor, dim=0, index=torch.tensor([1,2]).cuda())
    target_target_cv3_test = torch.index_select(target_tensor, dim=0, index=torch.tensor([0]).cuda())
    torch.save(target_target_cv3_train, os.path.join(output_filepath, "target_target_cv3_train.pt"))
    torch.save(target_target_cv3_test, os.path.join(output_filepath, "target_target_cv3_test.pt"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
