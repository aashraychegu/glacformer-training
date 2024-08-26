import gradio as gr
import os
import signal
import pickle
from PIL import Image
import numpy as np
import pathlib as pl


def blend_images(image, mask, offset=0):
    image = image.convert("RGB")
    mask = mask.convert("RGB")
    mask2 = mask.transform(mask.size, Image.AFFINE, (1, 0, 0, 0, 1, -1 * offset))
    result = Image.blend(image, mask2, alpha=0.08)
    return result.resize((image.size[0] * 6, image.size[1] * 3))


def set_offset(image, mask, parent_pid):

    def get_offset(editor_out, slider):
        difference = slider
        if editor_out["background"].size[1] != image.size[1] * 3:
            difference = int(editor_out["background"].size[1] / 3)
        with open(pl.Path(__file__).parent / "offset.pickle", "wb") as f:
            pickle.dump(difference, f)
        print(f"Offset saved as {difference}")
        gr.Warning(
            f"<h1><span style='font-size: 48px; color:green'>Offset saved as {difference} - Press close to close the Editor</span></h1>",
            duration=2,
        )

    def adjust_image(slider_val):
        gr.Info("Adjusting mask", duration=0.5)
        return blend_images(image, mask, slider_val)

    def close_editor():
        gr.Warning(
            """
            <script>
                function closeBrowser(){
                    console.log("Closing Browser");
                    window.close();
                }
            </script>
            <h1><span style='font-size: 48px; color:red' onload="closeBrowser();">Closing Editor - Close the tab</span></h1>
            """,
            duration=2,
        )
        os.kill(signal.CTRL_C_EVENT, 0)

    with gr.Blocks(
        fill_width=True,
        css="""#slider1 { transform: rotate(90deg); width: 300px; height: 300px;}""",
    ) as blocks:
        with gr.Row():
            with gr.Column(scale=3):
                editor = gr.ImageEditor(
                    value=blend_images(image, mask, 0),
                    type="pil",
                    # scale=3,
                )
            with gr.Column(scale=0, min_width=100):
                slider = gr.Slider(
                    0,
                    100,
                    value=0,
                    label="Offset",
                    step=1,
                    elem_id="slider1",
                    interactive=True,
                )
                slider.release(adjust_image, inputs=slider, outputs=editor)
                button = gr.Button(value="Get Offset")
                button.click(get_offset, inputs=[editor, slider])
                close = gr.Button(value="Close")
                close.click(close_editor)

    blocks.launch(
        inbrowser=True,
    )

    with open(pl.Path(__file__).parent / "offset.pickle", "rb") as f:
        offset = pickle.load(f)
    return int(offset)


# if __name__ == "__main__":
#     print(set_offset(
#         Image.open(
#             r"C:\Users\aashr\Desktop\research\glaciers\secondleg\30_0003576_0003600-reel_begin_end\cropped_img_30_0003576_0003600-reel_begin_end.png"
#         ).crop((0, 0, 500, 1360)),
#         visualize_pixels(
#             Image.open(
#                 r"C:\Users\aashr\Desktop\research\glaciers\secondleg\30_0003576_0003600-reel_begin_end\img_mask_30_0003576_0003600-reel_begin_end.png"
#             )
#         ).crop((0, 0, 500, 1360)),
#         0,
#     ))
