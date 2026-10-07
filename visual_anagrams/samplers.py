from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from diffusers.utils.torch_utils import randn_tensor

@torch.no_grad()
def sample(pipe,
            prompt_embeds,
            negative_prompt_embeds, 
            views,
            ref_im=None,
            num_inference_steps=100,
            guidance_scale=7.0,
            reduction='mean',
            generator=None):

    # Params
    num_images_per_prompt = 1
    #device = model.device
    device = torch.device('cuda')   # Sometimes model device is cpu???
    height = 1024
    width = 1024
    batch_size = 1      # TODO: Support larger batch sizes, maybe
    num_prompts = prompt_embeds.shape[0]
    assert num_prompts == len(views), \
        "Number of prompts must match number of views!"

    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Setup timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Make noise
    noise = pipe.generate_noise(
            (1, 3, height, width), 
            rand_device=device, 
            rand_torch_dtype=torch.float32,
            torch_dtype=torch.float32
        )

    # Parameters
    inputs_posi = {
        "prompt": prompt,
    }
    inputs_nega = {
        "negative_prompt": negative_prompt,
    }
    inputs_shared = {
        "cfg_scale": cfg_scale,
        "input_image": input_image, "denoising_strength": denoising_strength,
        "height": height, "width": width,
        "seed": seed, "rand_device": rand_device,
        "num_inference_steps": num_inference_steps,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    # Resize ref image to correct size
    if ref_im is not None:
        ref_im = TF.resize(ref_im, height)
        ref_im = ref_im.to(noise.device).to(noise.dtype)

    for i, t in enumerate(tqdm(timesteps)):
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

        # Apply views to noisy_image
        viewed_noisy_images = []
        for view_fn in views:
            viewed_noisy_images.append(view_fn.view(noisy_images[0]))
        viewed_noisy_images = torch.stack(viewed_noisy_images)

        # Duplicate inputs for CFG
        # Model input is: [ neg_0, neg_1, ..., pos_0, pos_1, ... ]
        model_input = torch.cat([viewed_noisy_images] * 2)
        model_input = pipe.scheduler.scale_model_input(model_input, t)

        # Predict noise estimate
        noise_pred = pipe.unet(
            model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Extract uncond (neg) and cond noise estimates
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # Invert the unconditional (negative) estimates
        inverted_preds = []
        for pred, view in zip(noise_pred_uncond, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_uncond = torch.stack(inverted_preds)

        # Invert the conditional estimates
        inverted_preds = []
        for pred, view in zip(noise_pred_text, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_text = torch.stack(inverted_preds)

        # Split into noise estimate and variance estimates
        noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Reduce predicted noise and variances
        noise_pred = noise_pred.view(-1,num_prompts,3,64,64)
        predicted_variance = predicted_variance.view(-1,num_prompts,3,64,64)
        if reduction == 'mean':
            noise_pred = noise_pred.mean(1)
            predicted_variance = predicted_variance.mean(1)
        elif reduction == 'sum':
            # For factorized diffusion
            noise_pred = noise_pred.sum(1)
            predicted_variance = predicted_variance.mean(1)
        elif reduction == 'alternate':
            noise_pred = noise_pred[:,i%num_prompts]
            predicted_variance = predicted_variance[:,i%num_prompts]
        else:
            raise ValueError('Reduction must be either `mean` or `alternate`')
        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        # compute the previous noisy sample x_t -> x_t-1
        noisy_images = pipe.scheduler.step(
            noise_pred, t, noisy_images, generator=generator, return_dict=False
        )[0]

    # Return denoised images
    return noisy_images
