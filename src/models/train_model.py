# import libraries
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from reflownet import EncoderDecoderConvLSTM
from pytorch_lightning.loggers import NeptuneLogger
import argparse
import torch.nn as nn
import click

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
# parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
# parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
# parser.add_argument('--batch_size', default=12, type=int, help='batch size')
# parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
# parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
# parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes')
# parser.add_argument('--n_hidden_dim', type=int, default=8, help='number of hidden dim for ConvLSTM layers')
# parser.add_argument('--log_images', action='store_true', help='Whether to log images')
# parser.add_argument('--is_distributed', action='store_true', help='Whether to used distributeds dataloader')

# parser.add_argument('--root', type=str, default="./dataset")
# parser.add_argument('--src_input_file', type=str, default="source_input.pt")
# parser.add_argument('--src_target_file', type=str, default="source_target.pt")
# parser.add_argument('--tar_input_file', type=str, default="target_input.pt")
# parser.add_argument('--tar_target_file', type=str, default="sp_target_target.pt")
# parser.add_argument('--time_steps', type=int, default=7)

# parser.add_argument('--model_path', type=str, default="checkpoints/lstm_ac.ckpt")
# parser.add_argument('--out_model_path', type=str, default="checkpoints/lstm_ac.ckpt")
# parser.add_argument('--retrain', action='store_true', help='Whether to retrain the model or not')
# parser.add_argument('--neptune_logger', action='store_true', help='Whether to use neptune.ai logger')
# parser.add_argument('--api_key', type=str,
#                     default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOTE0MGFjYy02NzMwLTRkODQtYTU4My1lNjk0YWEzODM3MGIifQ==")

# parser.add_argument('--run_type', type=str, default="train")
# parser.add_argument('--source_only', action="store_true", help="Wether training model using source only or not")
# parser.add_argument('--single_f', action="store_true", help="Wether model uses signle feature extractor")


def load_model(arch, model_path):
    checkpoint = torch.load(model_path)
    arch.load_state_dict(checkpoint["model"])


def count_epoch_on_large_dataset(train_loader_target, train_loader_source):
    batch_number_t = len(train_loader_target)
    batch_number = batch_number_t
    batch_number_s = len(train_loader_source)
    if batch_number_s > batch_number_t:
        batch_number = batch_number_s
    return batch_number


@click.command()
@click.option('--beta', nargs=2, type=float, default=[0.9, 0.98])
@click.option('-channel', nargs=1, type=int, default=2)
@click.option('--n_hidden_dim', nargs=1, type=int, default=8)
@click.option('--lr', nargs=1, type=float, default=0.001)
@click.option('--batch_size', nargs=1, type=float, default=0.001)
@click.option('--epoch_size', nargs=1, type=float, default=0.001)
@click.argument('--model_path', nargs=1, type=click.Path(exists=True))
def main(beta, n_hidden_dim, lr, model_path, channel):
    model = EncoderDecoderConvLSTM(nf=n_hidden_dim, in_chan=channel)
    load_model(model, model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta[0], beta[1]))
    batch_number = count_epoch_on_large_dataset(train_loader_target, train_loader_source, args)
    num_itern_total = args.epochs * batch_number

 

class OvenLightningModule(pl.LightningModule):

    def __init__(self, opt):

        super(OvenLightningModule, self).__init__()

        self.save_hyperparameters()
        self.opt = opt
        self.normalize = False

        self.model_src = EncoderDecoderConvLSTM(
            nf=self.opt.n_hidden_dim, in_chan=opt.channel
            )
        self.model_tar = EncoderDecoderConvLSTM(
            nf=self.opt.n_hidden_dim, in_chan=opt.channel
            )

        self.log_images = self.opt.log_images
        self.criterion = nn.MSELoss()
        self.pdist = nn.PairwiseDistance(p=2)
        self.batch_size = self.opt.batch_size
        self.time_steps = self.opt.time_steps
        self.epoch = 0
        self.step = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr,
            betas=(opt.beta_1, opt.beta_2)
            )

        return optimizer

    def load_model(self):
        checkpoint = torch.load(self.opt.model_path)
        if self.opt.singe_f:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model1.load_state_dict(checkpoint["model1"])
            self.model2.load_state_dict(checkpoint["model1"])

    def forward(self, x, model, source_only, domain):

        output, f1, f2 = model(x, future_step=self.time_steps, source_only=self.opt.source_only, domain=domain)

        return output, f1, f2


    def training_step(self, batch, batch_idx):

        src_batch = batch[0]
        tar_batch = batch[1]
        src_x, src_y = src_batch
        tar_x, tar_y = tar_batch

        if self.opt.single_f:
            src_y_hat, src_ln_1, src_ln_2 = self.forward(src_x, self.model, self.opt.source_only, domain="src")
            tar_y_hat, tar_ln_1, tar_ln_2 = self.forward(tar_x, self.model, self.opt.source_only, domain="tar")
        else:
            src_y_hat, src_ln_1, src_ln_2 = self.forward(src_x, self.model_src, self.opt.source_only, domain=None)
            tar_y_hat, tar_ln_1, tar_ln_2 = self.forward(tar_x, self.model_tar, self.opt.source_only, domain=None)

        src_loss = self.criterion(src_y_hat, torch.log(src_y))
        tar_loss = self.criterion(tar_y_hat, torch.log(tar_y))

        dst_loss1 = torch.mean(self.pdist(src_y_hat, tar_y_hat))
        dst_loss2 = torch.mean(self.pdist(src_ln_1, tar_ln_1))
        dst_loss3 = torch.mean(self.pdist(src_ln_2, tar_ln_2))

        # mmd_loss3 = MMD(src_ln_2, tar_ln_2, kernel="multiscale")

        # reg_loss = self.RegLoss(self.model1, self.model2, self.alpha, self.beta)

        avg_diff_src_src = torch.mean(torch.abs(src_y_hat - torch.log(src_y)))
        avg_diff_tar_tar = torch.mean(torch.abs(tar_y_hat - torch.log(tar_y)))

        self.log("src_loss", src_loss.item(), on_step=False, on_epoch=True)
        self.log("tar_loss", tar_loss.item(), on_step=False, on_epoch=True)
        # self.log("mmd_loss1", mmd_loss1.item(), on_step=True, on_epoch=True)
        # self.log("mmd_loss2", mmd_loss2.item(), on_step=True, on_epoch=True)
        # self.log("mmd_loss3", mmd_loss3.item(), on_step=True, on_epoch=True)
        self.log("dst_loss1", dst_loss1, on_step=False, on_epoch=True)
        self.log("dst_loss2", dst_loss2, on_step=False, on_epoch=True)

        # self.log("alpha", self.alpha[0], on_step=False, on_epoch=True)
        # self.log("beta", self.beta[0], on_step=False, on_epoch=True)

        self.log("avg_diff_src_src", avg_diff_src_src.item(), on_step=False, on_epoch=True)

        self.log("avg_diff_tar_tar", avg_diff_tar_tar.item(), on_step=False, on_epoch=True)

        if self.opt.source_only:
            loss = src_loss
            return loss

        else:
            loss = src_loss + 0.01 * tar_loss + dst_loss1 + dst_loss2 + dst_loss3

            return {"loss": loss, "src_y_hat": src_y_hat, "tar_y_hat": tar_y_hat, "src_ln_1": src_ln_1, "tar_ln_1": tar_ln_1, "src_ln_2": src_ln_2, "tar_ln_2": tar_ln_2}

    def training_epoch_end(self, training_step_outputs):

        if not self.opt.source_only:

            # src_ln_1 = torch.cat([i["src_ln_1"] for i in training_step_outputs], axis=0)
            # src_ln_2 = torch.cat([i["src_ln_2"] for i in training_step_outputs], axis=0)
            # tar_ln_1 = torch.cat([i["tar_ln_1"] for i in training_step_outputs], axis=0)
            # tar_ln_2 = torch.cat([i["tar_ln_2"] for i in training_step_outputs], axis=0)
            # loss = torch.cat([i["loss"].view(1, -1) for i in training_step_outputs], axis=0)
            # torch.save(src_ln_1, f"feature/src_ln_1_epoch{self.epoch}.pt")
            # torch.save(src_ln_2, f"feature/src_ln_2_epoch{self.epoch}.pt")
            # torch.save(tar_ln_1, f"feature/tar_ln_1_epoch{self.epoch}.pt")
            # torch.save(tar_ln_2, f"feature/tar_ln_2_epoch{self.epoch}.pt")
            # torch.save(loss, f"feature/loss_epoch{self.epoch}.pt")

            self.epoch += 1

def test_trainer():
    model =OvenLightningModule(opt).cuda()
    oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.tar_input_file, opt.tar_target_file, batch_size=1)
    oven_data.setup()
    source_loader, target_loader = oven_data.test_dataloader()
    model.load_model()
    model.eval()
    for i, batch in enumerate(target_loader):
        inp, target = batch
        with torch.no_grad():
            if opt.single_f:
                if opt.source_only:
                    predictions, _, _ = model(inp, model.model, opt.source_only, "src")
                else:
                    predictions, _, _ = model(inp, model.model, opt.source_only, "tar")
            else:
                if opt.source_only:
                    predictions, _, _ = model(inp, model.model1, opt.source_only, None)
                elif opt.single_f:
                    predictions, _, _ = model(inp, model.model2, opt.source_only, None)

        predictions = torch.exp(predictions)
        plt.figure(figsize=(4, 3))
        plt.plot([i+1 for i in range(target.shape[1])], target.view(-1).cpu(), ".-", label="Target")
        plt.plot([i+1 for i in range(target.shape[1])], predictions.view(-1).cpu(), ".-", label="Prediction")
        plt.ylabel("Temperature")
        plt.xlabel("Area")
        plt.legend()
        model_name = opt.out_model_path.split("/")[-1]
        model_name = model_name.split(".")[0]
        plt.savefig(f"Figure/sample{i+1}_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close("all")
        err = torch.norm(target-predictions)
        print(f"{model_name} predictions:", predictions.view(-1))
        print(f"{model_name} target:", target.view(-1))
        print(f"{model_name} error:", err)

def run_trainer():
    model =OvenLightningModule(opt).cuda()
    if not opt.source_only:
        oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.tar_input_file, opt.tar_target_file, opt.batch_size)
    else:
        oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.src_input_file, opt.src_target_file, opt.batch_size)
    if opt.neptune_logger:
        logger = NeptuneLogger(
                api_key=opt.api_key,
                    project='junkataoka/heatmap'
                                )
    else:
        logger = None


    trainer = Trainer(max_epochs=opt.epochs,
                        gpus=opt.n_gpus,
                        logger=logger,
                        accelerator='cuda',
                        num_nodes=opt.num_nodes,
                        # gradient_clip_val=0.5,
                        # multiple_trainloader_mode="min_size"
                      )

    if opt.retrain:
        model.load_model()

    trainer.fit(model, datamodule=oven_data)
    if opt.single_f:
        torch.save({"model":model.model.state_dict()
                    }, opt.out_model_path)

    else:
        torch.save({"model1":model.model1.state_dict(),
                    "model2":model.model2.state_dict()
                    }, opt.out_model_path)

if __name__ == '__main__':
    if opt.run_type=="test":
        test_trainer()
    else:
        run_trainer()
