from typing import Any
import gradio as gr

from backend.models.lcmdiffusion_setting import LCMDiffusionSetting
from context import Context
from models.interface_types import InterfaceType
from app_settings import Settings
from constants import LCM_DEFAULT_MODEL, LCM_DEFAULT_MODEL_OPENVINO
from frontend.utils import is_reshape_required
from app_settings import AppSettings
from constants import DEVICE
from frontend.utils import enable_openvino_controls
from scipy.ndimage import zoom
import numpy as np
from PIL import Image

random_enabled = True

context = Context(InterfaceType.WEBUI)
previous_width = 0
previous_height = 0
previous_model_id = ""
previous_num_of_images = 0


def generate_text_to_image(
    prompt,
    inference_steps,
    guidance_scale,
    seed,
    use_openvino,
    use_safety_checker,
) -> Any:
    global previous_height, previous_width, previous_model_id, previous_num_of_images
    model_id = LCM_DEFAULT_MODEL
    if use_openvino:
        model_id = LCM_DEFAULT_MODEL_OPENVINO

    use_seed = True if seed != -1 else False

    lcm_diffusion_settings = LCMDiffusionSetting(
        lcm_model_id=model_id,
        prompt=prompt,
        image_height=384,
        image_width=384,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        number_of_images=1,
        seed=seed,
        use_openvino=use_openvino,
        use_safety_checker=use_safety_checker,
        use_seed=use_seed,
    )
    settings = Settings(
        lcm_diffusion_setting=lcm_diffusion_settings,
    )
    reshape = False
    if use_openvino:
        reshape = is_reshape_required(
            previous_width,
            384,
            previous_height,
            384,
            previous_model_id,
            model_id,
            previous_num_of_images,
            1,
        )
    images = context.generate_text_to_image(
        settings,
        reshape,
        DEVICE,
    )
    previous_width = 384
    previous_height = 384
    previous_model_id = model_id
    previous_num_of_images = 1
    out_images = []
    for image in images:
        img_arr = np.array(image)
        upscaled_image = zoom(img_arr, (2, 2, 1), order=3)
        out_images.append(Image.fromarray(upscaled_image.astype(np.uint8)))

    return out_images


def get_text_to_image_ui(app_settings: AppSettings) -> None:
    with gr.Blocks():
        with gr.Row():
            with gr.Column():

                def random_seed():
                    global random_enabled
                    random_enabled = not random_enabled
                    seed_val = -1
                    if not random_enabled:
                        seed_val = 42
                    return gr.Number.update(
                        interactive=not random_enabled, value=seed_val
                    )

                with gr.Row():
                    prompt = gr.Textbox(
                        label="Describe the image you'd like to see",
                        lines=3,
                        placeholder="A fantasy landscape",
                    )

                    generate_btn = gr.Button(
                        "Generate",
                        elem_id="generate_button",
                        scale=0,
                    )

                with gr.Accordion("Advanced options", open=False):
                    guidance_scale = gr.Slider(
                        1.0, 30.0, value=8, step=0.5, label="Guidance Scale"
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        interactive=False,
                    )
                    seed_checkbox = gr.Checkbox(
                        label="Use random seed",
                        value=True,
                        interactive=True,
                    )

                    openvino_checkbox = gr.Checkbox(
                        label="Use OpenVINO",
                        value=True,
                        interactive=False,
                    )

                    safety_checker_checkbox = gr.Checkbox(
                        label="Use Safety Checker",
                        value=True,
                        interactive=True,
                    )
                    num_inference_steps = gr.Slider(
                        1, 8, value=4, step=1, label="Inference Steps"
                    )
                    # image_height = gr.Slider(
                    #     256, 768, value=384, step=64, label="Image Height",interactive=Fa
                    # )
                    # image_width = gr.Slider(
                    #     256, 768, value=384, step=64, label="Image Width"
                    # )
                    # num_images = gr.Slider(
                    #     1,
                    #     50,
                    #     value=1,
                    #     step=1,
                    #     label="Number of images to generate",
                    # )

                    input_params = [
                        prompt,
                        num_inference_steps,
                        guidance_scale,
                        seed,
                        openvino_checkbox,
                        safety_checker_checkbox,
                    ]

            with gr.Column():
                output = gr.Gallery(
                    label="Generated images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                )

    seed_checkbox.change(fn=random_seed, outputs=seed)
    generate_btn.click(
        fn=generate_text_to_image,
        inputs=input_params,
        outputs=output,
    )
