
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from torch_dataset import BreastCancerDataSet_Basic, BreastCancerDataSet_Mixup
from transforms import DualImageTransformation, get_transforms
import wandb
from training import train_model, train_model_simclr
import argparse
from models import SimClr, SimSiam
import gc

N_FOLDS = 5
IMG_SIZE = (256, 512)
TARGET = 'cancer'
WANDB_PROJECT_PREFIX = 'self_supervised_mammography_'

# Increaes NCCL_BUFFSIZE to prevent hangs
os.environ["NCCL_BUFFSIZE"] = f"{64 * 1024 * 1024}"

torch.set_float32_matmul_precision('high')


def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.set_device(rank)
  init_process_group(backend="nccl", rank=rank, world_size=world_size)


def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()

def main(rank, world_size, train_path, images_path, batch_size, 
         model_type, mixup, epochs, uniformity_loss, sim_clr):

    ddp_setup(rank, world_size)

    device = rank

    # Load dataset
    df_train = pd.read_csv(train_path)

    # Set up the (stratified and grouped) folds
    split = StratifiedGroupKFold(N_FOLDS, random_state=42, shuffle=True)
    for k, (_, test_idx) in enumerate(split.split(df_train, df_train.cancer, groups=df_train.patient_id)):
        df_train.loc[test_idx, 'split'] = k
    df_train.split = df_train.split.astype(int)

    # Use fold 0 for evaluation
    fold = 0

    # Set up transforms
    transforms = DualImageTransformation(get_transforms(aug = True, img_size = IMG_SIZE))

    # Load correct dataset class depending on the task, e.g. mixup or not
    if mixup:
        ds_train = BreastCancerDataSet_Mixup(df_train.query('split != @fold'), 
                                            images_path,
                                            TARGET = TARGET,
                                            transforms = transforms)
    else:
        ds_train = BreastCancerDataSet_Basic(df_train.query('split != @fold'), 
                                            images_path,
                                            TARGET = TARGET,
                                            transforms = transforms)

    
    eval_train_img_data = BreastCancerDataSet_Basic(df_train.query('split != @fold'), 
                                            images_path,
                                            TARGET = TARGET,
                                            transforms = get_transforms(aug = False, img_size = IMG_SIZE))
    eval_test_img_data = BreastCancerDataSet_Basic(df_train.query('split == @fold'), 
                                            images_path,
                                            TARGET = TARGET,
                                            transforms = get_transforms(aug = False, img_size = IMG_SIZE))

    out_dim = 256 if model_type == "resnet18" else 1024

    # Setup model
    if sim_clr:
        model = SimClr(model_type = model_type, img_size = IMG_SIZE, hidden_dim = 64 if model_type == "resnet18" else 128)
    else:
        model = SimSiam(model_type = model_type, img_size = IMG_SIZE, out_dim = out_dim)

    print("model", type(model).__name__)
    print("backbone dim", model.backbone_dim)

    print("Setting up model ... 1")
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print("Setting up model ... 2")
    model = DDP(model, device_ids=[device])

    print("Setting up model ... DONE")
    
    run = None

    if rank == 0:
        run = wandb.init(project=WANDB_PROJECT_PREFIX + model_type, config={
            "model": model_type,
            "batch_size": batch_size,
            "mixup": mixup,
            "img_size" : str(IMG_SIZE)
        })
        
    print("Starting training")
    cuda_device = f"cuda:{device}"
    model_checkpoint_filename = f"checkpoints/model_{'simclr' if sim_clr else ''}_{model_type}{'_mixup' if mixup else ''}{'_uniformity' if uniformity_loss else ''}_bs_{batch_size}"
    print("Saving best model to", model_checkpoint_filename)

    if sim_clr:
        train_model_simclr(run, model, ds_train, eval_train_img_data, 
                    eval_test_img_data, batch_size, cuda_device, epochs, 
                    shuffle = False, sampler = DistributedSampler(ds_train),
                    mixup = mixup, model_checkpoint_filename = model_checkpoint_filename,
                    uniformity = uniformity_loss)
    else:
        train_model(run, model, ds_train, eval_train_img_data, 
                    eval_test_img_data, batch_size, cuda_device, epochs, 
                    shuffle = False, sampler = DistributedSampler(ds_train),
                    mixup = mixup, model_checkpoint_filename = model_checkpoint_filename,
                    uniformity = uniformity_loss)
        
    destroy_process_group()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train_path', help="train.csv path", required=True)
    parser.add_argument('-i', '--images_path', help="path to images", required=True)
    parser.add_argument('-b', '--batch_size', help="batch size", required=True, type=int)
    parser.add_argument('-s', '--simclr', default=False, action='store_true')
    parser.add_argument('-m', '--mixup', default=False, action='store_true')
    parser.add_argument('-u', '--uniformity_loss', default=False, action='store_true')
    parser.add_argument('-mt', '--model_type', default="resnet50")
    parser.add_argument('-e', '--epochs', default="92", type=int)

    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size, args.train_path, args.images_path, args.batch_size, args.model_type, args.mixup, args.epochs, args.uniformity_loss, args.simclr), nprocs=world_size)
