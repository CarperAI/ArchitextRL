import functools

import gradio as gr
from PIL import Image
import time
import threading
import random


def init_image():
    imgs = [Image.new('RGB', (100, 100), color=(0, 0, 0)) for _ in range(12*12)]
    return imgs


def random_image():
    imgs = [Image.new('RGB', (100, 100), color=tuple(random.sample(range(256), 3))) for _ in range(12*12)]
    return imgs


images = init_image()
on = False

"""
# Run a daemon thread to execute `generate_image` every 5 seconds if `on` is True
def run_thread():
    global on, images
    while True:
        if on:
            images = generate_image()
        time.sleep(2)


# Start the thread
thread = threading.Thread(target=run_thread, daemon=True)
thread.start()
"""


def get_image(id=None):
    global images
    if id is None:
        return images
    return images[id]


def on_click():
    global on
    on = not on

    return str(on)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                label="Status",
                interactive=False,
                show_label=False,
                max_lines=1,
                placeholder=str(on),
            ).style(
                container=True,
            )
            btn = gr.Button("Run")
        with gr.Column(scale=12):
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery",
            ).style(columns=12, rows=12, container=False, object_fit="contain", height="100%")

    demo.load(get_image, inputs=[], outputs=[gallery], every=2,)

    """
    evt = demo.set_event_trigger(
                event_name="load",
                fn=get_image,
                inputs=[],
                outputs=[gallery],
                api_name=None,
                preprocess=True,
                postprocess=True,
                scroll_to_output=False,
                show_progress=False,
                js=None,
                queue=None,
                batch=False,
                max_batch_size=4,
                every=2,
                no_target=True,
            )
    """

    btn.click(on_click, None, text, show_progress=False)



#if __name__ == "__main__":
demo.queue(concurrency_count=1, max_size=20).launch()
