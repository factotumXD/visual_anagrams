from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

@torch.no_grad()
def sample_l2p(pipe,
            prompt_embeds,
            negative_prompt_embeds, 
            views,
            ref_im=None,
            num_inference_steps=100,
            cfg_scale=7.0,
            cfg_scale_view=7.0,
            reduction='mean',
            height=1024,
            width=1024,
            seed=None,
            rand_device='cpu',
            progress_bar_cmd=tqdm
            ):

    # Setup timesteps
    pipe.scheduler.set_timesteps(num_inference_steps)

    image = pipe.generate_noise(
            (1, 3, height, width), 
            seed=seed, 
            rand_device=rand_device, 
            rand_torch_dtype=torch.float32,
            torch_dtype=torch.float32
        )

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    # Resize ref image to correct size
    if ref_im is not None:
        ref_im = TF.resize(ref_im, height)
        ref_im = ref_im.to(image.device).to(image.dtype)

    for i, t in enumerate(progress_bar_cmd(pipe.scheduler.timesteps)):
        # If solving an inverse problem, then project x_t so
        # that first component matches reference image's first component
        if ref_im is not None:
            # Inject noise to reference image
            alpha_cumprod = pipe.scheduler.alphas_cumprod[t]
            ref_noisy = torch.sqrt(alpha_cumprod) * ref_im + \
                        torch.sqrt(1 - alpha_cumprod) * torch.randn_like(ref_im)

            # Replace 1st component of x_t with 1st component of noisy ref image
            ref_noisy_component = views[0].inverse_view(ref_noisy)
            noisy_images_component = views[1].inverse_view(noisy_images[0])
            noisy_images = ref_noisy_component + noisy_images_component

            # Add back batch dim
            noisy_images = noisy_images[None]

        timestep = t.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        velocities = []
        for j, view in enumerate(views):
            viewed_image = view.view(image[0])
            if viewed_image.shape != image.shape[1:]:
                raise ValueError('Each view must preserve the RGB image shape.')
            # The upstream wrapper treats BCHW's B as video frames in a
            # single sample. Separate calls keep views and prompts paired.
            velocity = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale if view.__class__.__name__== "IdentityView" else cfg_scale_view,
                {'latents': viewed_image[None]},
                {'prompt_embeds': [prompt_embeds[j]]},
                {'prompt_embeds': [negative_prompt_embeds[j] if negative_prompt_embeds is not None else None]},
                **models, timestep=timestep, progress_id=i,
            )

            aligned = view.inverse_view(velocity[0].float()).float()
            if aligned.shape != image.shape[1:]:
                raise ValueError('Each inverse view must preserve the RGB image shape.')
            velocities.append(aligned)

        velocities = torch.stack(velocities)
        if reduction == 'mean':
            velocity = velocities.mean(0)
        elif reduction == 'sum':
            velocity = velocities.sum(0)
        else:
            velocity = velocities[i % len(views)]
        # Use the original scheduler timestep for sigma lookup, keeping
        # the pixel trajectory float32 as in L2P's native pipeline.
        image = pipe.scheduler.step(velocity[None], t, image)

    return image
