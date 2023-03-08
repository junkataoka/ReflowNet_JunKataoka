#!/bin/bash
#SBATCH --job-name=reflownet
#SBATCH --output=reflownet_output.txt
#SBATCH --error=reflownet_error.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpucompute
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

module load cuda11.1/toolkit/11.1.1

if [[ "$1" != "" ]]; then
    SRC_P=$1
else
    SRC_P=1
fi
WANdB_APIT_KEY=a78d9834d7b1cc2bacf2a7aca59aed42b4b2cd78

make delete
srun -n1 --gpus=1 --exclusive -c1 python src/data/make_dataset.py data/raw data/processed --test_recipe=0 --src_p=$1
srun -n1 --gpus=1 --exclusive -c1 python src/pretrain_model.py data/processed --log=test0_notar_all --epoch_size=200
srun -n1 --gpus=1 --exclusive -c1 python src/train_model.py data/processed --log=test0_notar_all --epoch_size=100
srun -n1 --gpus=1 --exclusive -c1 python src/predict_model.py data/processed --log=test0_notar_all

cp /data/home/jkataok1/ReflowNet_ver2/models/test0_notar_all/pretrained.ckpt /data/home/jkataok1/ReflowNet_ver2/models/test1_notar_all/pretrained.ckpt 
cp /data/home/jkataok1/ReflowNet_ver2/models/test0_notar_all/pretrained.ckpt /data/home/jkataok1/ReflowNet_ver2/models/test2_notar_all/pretrained.ckpt 
cp /data/home/jkataok1/ReflowNet_ver2/models/test0_notar_all/pretrained.ckpt /data/home/jkataok1/ReflowNet_ver2/models/test0_withtar_all/pretrained.ckpt 
cp /data/home/jkataok1/ReflowNet_ver2/models/test0_notar_all/pretrained.ckpt /data/home/jkataok1/ReflowNet_ver2/models/test1_withtar_all/pretrained.ckpt 
cp /data/home/jkataok1/ReflowNet_ver2/models/test0_notar_all/pretrained.ckpt /data/home/jkataok1/ReflowNet_ver2/models/test2_withtar_all/pretrained.ckpt 

make delete
srun -n1 --gpus=1 --exclusive -c1 python src/data/make_dataset.py data/raw data/processed --test_recipe=1 --src_p=$1
srun -n1 --gpus=1 --exclusive -c1 python src/train_model.py data/processed --log=test1_notar_all --epoch_size=100
srun -n1 --gpus=1 --exclusive -c1 python src/predict_model.py data/processed --log=test1_notar_all

make delete
srun -n1 --gpus=1 --exclusive -c1 python src/data/make_dataset.py data/raw data/processed --test_recipe=2 --src_p=$1
srun -n1 --gpus=1 --exclusive -c1 python src/train_model.py data/processed --log=test2_notar_all --epoch_size=100
srun -n1 --gpus=1 --exclusive -c1 python src/predict_model.py data/processed --log=test2_notar_all

make delete
srun -n1 --gpus=1 --exclusive -c1 python src/data/make_dataset.py data/raw data/processed --test_recipe=0 --src_p=$1 --no_tar_geom=False
srun -n1 --gpus=1 --exclusive -c1 python src/train_model.py data/processed --log=test0_withtar_all --epoch_size=100
srun -n1 --gpus=1 --exclusive -c1 python src/predict_model.py data/processed --log=test0_withtar_all

make delete
srun -n1 --gpus=1 --exclusive -c1 python src/data/make_dataset.py data/raw data/processed --test_recipe=1 --src_p=$1 --no_tar_geom=False
srun -n1 --gpus=1 --exclusive -c1 python src/train_model.py data/processed --log=test1_withtar_all --epoch_size=100
srun -n1 --gpus=1 --exclusive -c1 python src/predict_model.py data/processed --log=test1_withtar_all

make delete
srun -n1 --gpus=1 --exclusive -c1 python src/data/make_dataset.py data/raw data/processed --test_recipe=2 --src_p=$1 --no_tar_geom=False
srun -n1 --gpus=1 --exclusive -c1 python src/train_model.py data/processed --log=test2_withtar_all_$1 --epoch_size=100
srun -n1 --gpus=1 --exclusive -c1 python src/predict_model.py data/processed --log=test2_withtar_all
