import argparse
from pathlib import Path
from PIL import Image

import torch
from diffsynth.pipelines.z_image_L2P import ZImagePipeline, ModelConfig
import torchvision.transforms.functional as TF

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample
from visual_anagrams.utils import add_args, save_illusion, save_metadata



# Parse args
parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

# Do admin stuff
save_dir = Path(args.save_dir) / args.name
save_dir.mkdir(exist_ok=True, parents=True)

# Load reference image (for inverse problems)
if args.ref_im_path is not None:
    ref_im = Image.open(args.ref_im_path)
    ref_im = TF.to_tensor(ref_im) * 2 - 1
else:
    ref_im = None

main_model_path = "/path/model-1k-merge.safetensors"

text_encoder_paths = [
    "/path/Z-Image-Turbo/text_encoder/model-00001-of-00003.safetensors",
    "/path/Z-Image-Turbo/text_encoder/model-00002-of-00003.safetensors",
    "/path/Z-Image-Turbo/text_encoder/model-00003-of-00003.safetensors",
]

tokenizer_path = "/path/Z-Image-Turbo/tokenizer"

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[main_model_path]),
        ModelConfig(path=text_encoder_paths),
    ],
    tokenizer_config=ModelConfig(path=tokenizer_path),
)
pipe.enable_model_cpu_offload()
pipe = pipe.to(args.device)

# Get prompt embeddings
prompts = [f'{args.style} {p}'.strip() for p in args.prompts]
prompt_embeds = [pipe.encode_prompt(p) for p in prompts]
prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
prompt_embeds = torch.cat(prompt_embeds)
negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds

# Get views
views = get_views(args.views, view_args=args.view_args)

# Save metadata
save_metadata(views, args, save_dir)

# Sample illusions
for i in range(args.num_samples):
    # Admin stuff
    generator = torch.manual_seed(args.seed + i)
    sample_dir = save_dir / f'{args.seed + i:04}'
    sample_dir.mkdir(exist_ok=True, parents=True)

    image = sample(pipe, 
                    prompt_embeds,
                    negative_prompt_embeds,
                    views,
                    ref_im=ref_im,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    reduction=args.reduction,
                    generator=generator)
    save_illusion(image, views, sample_dir)

