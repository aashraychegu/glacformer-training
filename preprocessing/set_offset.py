import gradio as gr
import os
import signal
import pickle
from PIL import Image
import numpy as np
import pathlib as pl

# Explaining this is a real trip
# this is a script with a function that opens a web based GUI for selecting the offset of the mask


# blends 2 images together with an alpha value for the mask 'specially calibrated' for glacial scopes
def blend_images(image, mask, offset=0):
    image = image.convert("RGB")
    mask = mask.convert("RGB")
    mask2 = mask.transform(mask.size, Image.AFFINE, (1, 0, 0, 0, 1, -1 * offset))
    result = Image.blend(image, mask2, alpha=0.08)
    return result.resize((image.size[0] * 6, image.size[1] * 3))


# The primary function that is exported
def set_offset(image, mask, parent_pid):

    # The function that is called when you click the get offset button
    def get_offset(editor_out, slider):
        difference = slider
        if editor_out["background"].size[1] != image.size[1] * 3:
            difference = int(editor_out["background"].size[1] / 3)

        # due to the silliness of this library, i have to save the offset to a file and read from the file later
        with open(pl.Path(__file__).parent / "offset.pickle", "wb") as f:
            pickle.dump(difference, f)
        print(f"Offset saved as {difference}")
        gr.Warning(
            f"<h1><span style='font-size: 48px; color:green'>Offset saved as {difference} - Press close to close the Editor</span></h1>",
            duration=2,
        )

    # The function that is called when you move the slider
    def adjust_image(slider_val):
        gr.Info("Adjusting mask", duration=0.5)
        return blend_images(image, mask, slider_val)

    # The function that is called when you click the close button
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
        # tries to kill the parent process to close the editor
        os.kill(signal.CTRL_C_EVENT, 0)

    # main GUI
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

    # launches the GUI
    blocks.launch(
        inbrowser=True,
    )

    # returns the offset from the file
    with open(pl.Path(__file__).parent / "offset.pickle", "rb") as f:
        offset = pickle.load(f)

    return int(offset)
