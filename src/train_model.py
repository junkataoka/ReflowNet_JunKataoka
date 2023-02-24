# import libraries
import torch
from models.reflownet import EncoderDecoderConvLSTM, da_cos_loss
import torch.nn as nn
import click
from data.dataloader import generate_dataloader
import os
import math
import wandb

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def load_model(arch, model_path):
    checkpoint = torch.load(model_path)
    arch.load_state_dict(checkpoint["model"])

def adjust_learning_rate(epoch, epoch_size, lr):
    lr = lr / math.pow(1+10 * epoch / epoch_size, 0.75)
    return lr

def train(src_x, src_y, tar_x, tar_y, model, optimizer, criterions, hyper_params, num_areas):

    optimizer.zero_grad()
    src_output, src_feat1, src_feat2 = model(src_x, domain="src", future_step=num_areas)
    tar_output, tar_feat1, tar_feat2 = model(tar_x, domain="tar", future_step=num_areas)

    src_loss = criterions["mse"](src_output, src_y)
    tar_loss = criterions["mse"](tar_output, tar_y)

    da_loss1 = criterions["da"](src_output, tar_output).mean(dim=-1)
    da_loss2 = criterions["da"](src_feat1, tar_feat1).mean(dim=-1)
    da_loss3 = criterions["da"](src_feat2, tar_feat2).mean(dim=-1)
    da_loss = (da_loss1.sum() + da_loss2.sum() + da_loss3.sum()) / 3

    loss = src_loss + hyper_params["lambda_tar"] * tar_loss + hyper_params["lambda_da"] * da_loss
    loss.backward()
    optimizer.step()
    return {"loss":loss, "src_loss": src_loss, "ttrain_tar_loss": tar_loss, "da_loss": da_loss}

def validate(tar_x_test, tar_y_test, criterions, model, num_areas):
    with torch.no_grad():
        tar_output, _, _ = model(tar_x_test, domain="tar", future_step=num_areas)

        tar_loss = criterions["mse"](tar_output, tar_y_test)

        return {"test_tar_loss": tar_loss}

@click.command()
@click.option('--n_hidden_dim', nargs=1, type=int, default=4)
@click.option('--lr', nargs=1, type=float, default=0.001)
@click.option('--batch_size', nargs=1, type=int, default=24)
@click.option('--epoch_size', nargs=1, type=int, default=100)
@click.option('-channel', nargs=1, type=int, default=2)
@click.option('-seq_len', nargs=1, type=int, default=15)
@click.option('-log_id', nargs=1, type=int, default=1)
@click.option('-num_areas', nargs=1, type=int, default=7)
@click.argument('model_path', nargs=1, type=click.Path())
@click.argument('data_path', nargs=1, type=click.Path(exists=True))
def main(n_hidden_dim, lr, batch_size, epoch_size, model_path, data_path, channel, seq_len, num_areas, log_id):
    # load model
    model = EncoderDecoderConvLSTM(nf=n_hidden_dim, in_chan=channel, seq_len=seq_len).double().cuda()
    if model_path != "nomodel":
        load_model(model, model_path)


    # load dataloaders
    src_dataloader = generate_dataloader(
        geom_path = os.path.join(data_path, "processed/train-src-GEOM"),
        heatmap_path = os.path.join(data_path, "processed/train-src-HEATMAP"),
        recipe_path = os.path.join(data_path, "processed/train-src-RECIPE"),
        batch_size=batch_size,
        train=True
    )
    train_tar_dataloader = generate_dataloader(
        geom_path = os.path.join(data_path, "processed/train-tar-GEOM"),
        heatmap_path = os.path.join(data_path, "processed/train-tar-HEATMAP"),
        recipe_path = os.path.join(data_path, "processed/train-tar-RECIPE"),
        batch_size=batch_size,
        train=True
    )
    test_tar_dataloader = generate_dataloader(
        geom_path = os.path.join(data_path, "processed/test-tar-GEOM"),
        heatmap_path = os.path.join(data_path, "processed/test-tar-HEATMAP"),
        recipe_path = os.path.join(data_path, "processed/test-tar-RECIPE"),
        batch_size=1,
        train=False
    )

    # define optimizer and hyperparameters
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)

    # define loss functions
    criterions = {"mse": nn.MSELoss(),
                  "da": da_cos_loss}

    run = wandb.init(project="reflownet", 
                     config={
                        "log_id":log_id
                     })

    for epoch in range(epoch_size):

        hyper_params = {"lambda_tar": 0.01, "lambda_da": 2 / (1+math.exp(-1*10*epoch/epoch_size)) - 1}
        run.log(hyper_params)

        for i, data in enumerate(zip(src_dataloader, train_tar_dataloader)):

            (src_x, src_y), (tar_x, tar_y) = data
            src_x = src_x.double().cuda()
            src_y = torch.log(src_y).double().cuda()
            tar_x = tar_x.double().cuda()
            tar_y = torch.log(tar_y).double().cuda()
            loss_dict_train = train(src_x, src_y, tar_x, tar_y, model, optimizer, criterions, hyper_params, num_areas=num_areas) 
            run.log(loss_dict_train)
        
        for i, data in enumerate(test_tar_dataloader):

            (tar_x, tar_y) = data
            tar_x = tar_x.double().cuda()
            tar_y = torch.log(tar_y).double().cuda()
            loss_dict_val = validate(tar_x, tar_y, criterions=criterions, model=model, num_areas=num_areas)
            run.log(loss_dict_val)

        adjust_learning_rate(epoch, epoch_size, lr)
        run.log({"lr": lr})
        run.log({"epoch": epoch})
        
    run.finish()
    
if __name__ == '__main__':
    main()
