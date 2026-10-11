import argparse
from pathlib import Path
from PIL import Image

import torch
from diffsynth.pipelines.z_image_L2P import ZImagePipeline, ModelConfig, ZImageUnit_PromptEmbedder
import torchvision.transforms.functional as TF

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_l2p
from visual_anagrams.utils import add_args, save_illusion, save_metadata

from tqdm import tqdm

# Parse args
parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()
if args.cfg_scale_view is None:
    args.cfg_scale_view = args.cfg_scale

if len(args.views) != len(args.prompts):
    parser.error('--views must contain one view per prompt')

if args.negative_prompts is None:
    args.negative_prompts = [''] * len(args.prompts)
elif len(args.negative_prompts) == 1:
    args.negative_prompts = args.negative_prompts * len(args.prompts)
elif len(args.negative_prompts) != len(args.prompts):
    parser.error('--negative_prompts must contain one prompt or one per positive prompt')

# Do admin stuff
save_dir = Path(args.save_dir) / args.name
save_dir.mkdir(exist_ok=True, parents=True)

# Load reference image (for inverse problems)
if args.ref_im_path is not None:
    ref_im = Image.open(args.ref_im_path)
    ref_im = TF.to_tensor(ref_im) * 2 - 1
else:
    ref_im = None

torch.manual_seed(args.seed)

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=getattr(torch, args.dtype),
    device=args.device,
    model_configs=[
        ModelConfig(path=[args.main_model_path]),
        ModelConfig(path=args.text_encoder_paths),
    ],
    tokenizer_config=ModelConfig(path=args.tokenizer_path),
)

# Get prompt embeddings
prompts = [f'{args.style} {p}'.strip() for p in args.prompts]
embedder = ZImageUnit_PromptEmbedder()
prompt_embeds = embedder.process(pipe, prompts.copy())['prompt_embeds']
negative_prompt_embeds = embedder.process(pipe, args.negative_prompts.copy())['prompt_embeds']

# Get views
views = get_views(args.views, view_args=args.view_args)

# Save metadata
if args.save_metadata:
    save_metadata(views, args, save_dir)

# Sample illusions
for i in tqdm(range(args.num_samples)):
    # Admin stuff
    seed = args.seed + i
    sample_dir = save_dir / f'{seed:04}'
    sample_dir.mkdir(exist_ok=True, parents=True)

    image = sample_l2p(
                    pipe, 
                    prompt_embeds,
                    negative_prompt_embeds,
                    views,
                    ref_im=ref_im,
                    num_inference_steps=args.num_inference_steps,
                    cfg_scale=args.cfg_scale,
                    cfg_scale_view=args.cfg_scale_view,
                    reduction=args.reduction,
                    height=args.height, 
                    width=args.width,
                    seed=seed,
                    rand_device=args.rand_device,)
    save_illusion(image, views, sample_dir)
