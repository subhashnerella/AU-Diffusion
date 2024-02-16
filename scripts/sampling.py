import argparse, os, sys, glob
from datetime import datetime
sys.path.append(os.getcwd())
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from imagedata import DiscreteImages
from ldm.data.dataset import MultiDataset
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()

    return model

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0 ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates

@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(None, shape, verbose=True)
    
@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--config",
                        type=str,
                        default="configs/latent-diffusion/AU-ldm-vq.yaml",
                        help="path to config which constructs model",
                        )
    parser.add_argument(
                        "--outdir",
                        type=str,
                        nargs="?",
                        help="dir to write results to",
                        )
    parser.add_argument(
                        "--n_samples",
                        type=int,
                        default=3,
                        help="how many samples to produce for each given prompt. A.k.a batch size",
                        )
    parser.add_argument(
                        "--batch_size",
                        type=int,
                        default=4,
                        help="batch size for dataloader",
                        )
    parser.add_argument(
                        "--ckpt",
                        type=str,
                        default="logs/2023-10-29T00-59-03_AU-ldm-vq",
                        help="path to checkpoint of model",
                        )
    parser.add_argument(
                        "--seed",
                        type=int,
                        default=42,
                        help="the seed (for reproducible sampling)",
                        )
    parser.add_argument(
                        "--ddim_eta",
                        type=float,
                        default=0.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
                        )
    parser.add_argument(
                        "--ddim_steps",
                        type=int,
                        default=200,
                        help="number of ddim sampling steps",
                        )
    parser.add_argument(
                        "--n_iter",
                        type=int,
                        default=1,
                        help="sample this often",
                        )
    parser.add_argument(
                        "--precision",
                        type=str,
                        help="evaluate at this precision",
                        choices=["full", "autocast"],
                        default="autocast"
                        )
    return parser



def main():
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = get_parser()
    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    #check if opt.ckpt is a directory
    if os.path.isdir(opt.ckpt):
        ckpt = os.path.join(opt.ckpt, "checkpoints", "last.ckpt")
    model = load_model_from_config(config, ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    if opt.outdir is not None:
        os.makedirs(opt.outdir, exist_ok=True)
        outdir = opt.outdir
    else:
        outdir = opt.ckpt
        outdir = os.path.join(outdir, f"sampling_{now}")


    datasets = ['BP4D', 'DISFA', 'UNBC','BP4DPlus']
    split = 'val'
    aus =  ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU9', 'AU10','AU12', 'AU14', 'AU15', 'AU17','AU20', 'AU23', 'AU24', 'AU25','AU26','AU27','AU43']
    size = 256
    dataset = MultiDataset(datasets,aus, split, size)
    batch_size = opt.batch_size
    dataloader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True,
                                            )
    

    with model.ema_scope():
        sampler = DDIMSampler(model)
        for batch in tqdm(dataloader):
            aus = batch['aus']
            aus[aus == -1] = 0
            aus = aus.to(device)
            aus = aus.repeat_interleave(opt.n_samples,dim=0)
            aus_map = {'aus': aus}
            conditioning = model.get_learned_conditioning(aus_map)
            shape = (model.channels, model.image_size, model.image_size)
            samples,_ = sampler.sample(200,batch_size=opt.n_samples*batch_size,shape=shape, conditioning=conditioning,verbose=False,eta=1.0)
            x_samples = model.decode_first_stage(samples)
            paths = batch['file_path_']
            save_samples(x_samples, paths, outdir)

def save_samples(samples, paths, outdir):

    for i in range(len(paths)):
        sample = samples[3*i:3*i+3]
        
        path = paths[i].replace('data/','')
        path = path.split('/')[-4:]
        path ='/'.join(path)
        path = os.path.join(outdir, path).split('.')[0]
        os.makedirs(path, exist_ok=True)
        for i, s in enumerate(sample):
            s = custom_to_pil(s)
            s.save(os.path.join(outdir, f"{i}.png"))


def save_sample(sample, path, outdir):
    for i, s in enumerate(sample):
        s = custom_to_pil(s)
        path = path.replace('data/','')
        path = path.split('/')[-4:]
        path ='/'.join(path)
        path = os.path.join(outdir, path).split('.')[0]
        os.makedirs(path, exist_ok=True)
        s.save(os.path.join(outdir, f"{i}.png"))

if __name__ == "__main__":
    main()