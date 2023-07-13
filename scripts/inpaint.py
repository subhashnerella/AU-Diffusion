import argparse, os, sys, glob
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
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import torchvision



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--config",
                        type=str,
                        default="scripts/infconfig/AU-ldm-vq.yaml",
                        help="path to config which constructs model",
                        )
    parser.add_argument(
                        "--outdir",
                        type=str,
                        nargs="?",
                        help="dir to write results to",
                        default="outputs"
                        )
    parser.add_argument(
                        "--n_samples",
                        type=int,
                        default=2,
                        help="how many samples to produce for each given prompt. A.k.a batch size",
                        )
    parser.add_argument(
                        "--n_rows",
                        type=int,
                        default=0,
                        help="rows in the grid (default: n_samples)",
                        )
    parser.add_argument(
                        "--ckpt",
                        type=str,
                        default="logs/2023-06-23T23-55-11_AU-ldm-vq/checkpoints/last.ckpt",
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
                        "--strength",
                        type=float,
                        default=0.75,
                        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
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


    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()


    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = 8
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
 
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
       
    dataset = DiscreteImages('testimages', 18, size=256)
    length = dataset.__len__()
    data = dataset.__getitem__(0)
    #assert os.path.isfile(opt.init_img)
    init_image = torch.tensor(data['image'],device=device).unsqueeze(0)
    
    aus = np.zeros((18))
    aus[-4] = 1
    aus = np.repeat(aus[None,:], batch_size, axis=0)
    aus = torch.tensor(aus).to(device)
    data['aus'] = aus
    data['aus_humanlabel']=['AU25']

    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    data['image'] = init_image 
    log = model.log_images(data,plot_progressive_rows=False,quantize_denoised=False,plot_denoise_rows=False,img2img=False)
    inpaint = log['samples_inpainting']
    grid = torchvision.utils.make_grid(inpaint, nrow=4)
    grid = (grid + 1.0) / 2.0
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.cpu().numpy()
    grid = (grid * 255).astype(np.uint8)
    path = os.path.join(outpath, "inpaint.png")
    Image.fromarray(grid).save(path)

if __name__ == "__main__":
    main()