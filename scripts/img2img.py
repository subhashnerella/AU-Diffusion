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
                        default="configs/latent-diffusion/AU-ldm-vq.yaml",
                        help="path to config which constructs model",
                        )
    parser.add_argument(
                        "--outdir",
                        type=str,
                        nargs="?",
                        help="dir to write results to",
                        default="outputs/img2img-samples"
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
                        default="logs/2023-10-29T00-59-03_AU-ldm-vq/checkpoints/last.ckpt",
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
    init_image = torch.tensor(data['image'],device=device).unsqueeze(0).permute(0,3,1,2)

    aus = np.zeros((18))
    aus[-4] = 1
    aus = np.repeat(aus[None,:], batch_size, axis=0)
    aus = torch.tensor(aus).to(device)
    data['aus'] = aus

    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image)) 

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    # assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    # t_enc = int(opt.strength * opt.ddim_steps)
    # print(f"target t_enc is {t_enc} steps")

    strengths = np.asarray(1+np.linspace(0, 1, 9)[1:]*200,dtype=int)
    strengths = np.array([25]).repeat(batch_size, axis=0)
    

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()

                #for au in tqdm(data['aus'], desc="data"):
                    
                c = model.get_learned_conditioning(data)

                # encode (scaled latent)
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor(strengths).to(device))
                # decode it
                samples = sampler.decode(z_enc, c, strengths[0])

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1
                all_samples.append(x_samples)
                #if not opt.skip_grid:
                    # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

if __name__ == "__main__":
    main()