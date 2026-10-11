import pickle
from pathlib import Path

import torch

from PIL import Image


def add_args(parser):
    '''
    Add arguments for sampling to a parser
    '''

    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--save_dir", type=str, default='results', help='Location to samples and metadata')
    parser.add_argument("--prompts", required=True, type=str, nargs='+', help='Prompts to use, corresponding to each view.')
    parser.add_argument("--negative_prompts", type=str, nargs='+', default=None,
                        help='One negative prompt for all views, or one per view (default: empty).')
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--main_model_path", type=str, required=True,
                        help='Full merged L2P pixel-space inference checkpoint (.safetensors).')
    parser.add_argument("--text_encoder_paths", type=str, nargs='+', required=True,
                        help='Z-Image-Turbo text encoder checkpoint shards.')
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help='Z-Image-Turbo tokenizer directory.')
    parser.add_argument("--dtype", choices=['bfloat16', 'float16', 'float32'], default='bfloat16',
                        help='Model precision')
    parser.add_argument("--views", required=True, type=str, nargs='+', help='Name of views to use. See `get_views` in `views.py`.')
    parser.add_argument("--style", default='', type=str, help='Optional string to prepend prompt with')
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--reduction", type=str, default='mean')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--cfg_scale_view", type=float, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--rand_device", type=str, default='cpu', help='Device for seeded initial noise.')
    parser.add_argument("--save_metadata", action='store_true', help='If true, save metadata about the views. May use lots of disk space, particularly for permutation views.')
    parser.add_argument("--view_args", default=None, type=str, nargs='+', help='Args to pass to views')
    parser.add_argument("--ref_im_path", type=str, default=None, help='Image from which to extract a component, when solving inverse problems')

    return parser

# adapted from diffsynth/pipelines/z_image_L2P.py
def pixel_output_to_image(pixel_tensor, min_value=-1, max_value=1):
    """
    将 Pixel Space 的 Tensor (B, C, H, W) 转换为 PIL Image
    适配输入范围 [min_value, max_value]
    """
    range_val = max_value - min_value
    pixel_tensor = (pixel_tensor - min_value) / range_val
    
    pixel_tensor = pixel_tensor.clamp(0, 1)
    pixel_tensor = pixel_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    
    pixel_tensor = (pixel_tensor * 255).round().astype("uint8")
    
    images = [Image.fromarray(image) for image in pixel_tensor]
    return images[0] if len(images) == 1 else images


def save_illusion(image, views, sample_dir):
    '''
    Saves the illusion (`image`), as well as all views of the illusion

    image (torch.tensor) :
        Tensor of shape (1,3,H,W) representing the image

    views (views.BaseView) :
        Represents the view, inherits from BaseView

    sample_dir (pathlib.Path) :
        pathlib Path object, representing the directory to save to
    '''

    size = image.shape[-1]

    # Save illusion
    save_name = sample_dir / f'sample_{size}.png'
    PIL_image = pixel_output_to_image(image)
    PIL_image.save(save_name)

    # Save views of the illusion
    # TODO: This is kind of ugly... can we fix it?
    im_views = torch.cat([
        view.save_view(image[0]) if hasattr(view, 'save_view') else view.view(image[0])
        for view in views
    ], dim=-1).unsqueeze(0)
    save_name_views = sample_dir / f'sample_{size}.views.png'
    PIL_im_views = pixel_output_to_image(im_views)
    PIL_im_views.save(save_name_views)

    for view in views:
        im_view = view.view(image[0])
        save_name_view = sample_dir / f'sample_{size}.{view.__class__.__name__}.png'
        PIL_im_view = pixel_output_to_image(im_view.unsqueeze(0))
        PIL_im_view.save(save_name_view)

def save_metadata(views, args, save_dir):
    '''
    Saves the following the sample_dir
        1) pickled view object
        2) args for the illusion
    '''

    metadata = {
        'views': views,
        'args': args
    }
    with open(save_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def get_courier_font_path():
    font_path = Path(__file__).parent / 'assets' / 'CourierPrime-Regular.ttf'
    return str(font_path)
