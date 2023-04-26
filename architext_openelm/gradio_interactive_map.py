import os

import gradio as gr
import pickle

from PIL import Image
from omegaconf import OmegaConf

from run_elm import ArchitextELM


def get_imgs(elm_obj):
    dims = elm_obj.dims
    result = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            if elm_obj[i, j] == 0.0:
                img = Image.new('RGB', (256, 256), color=(255, 255, 255))
            else:
                img = elm_obj[i, j].get_image()
            result.append(img)

    return result

typologies = ["1b1b", "2b1b", "2b2b", "3b1b", "3b2b", "3b3b", "4b1b", "4b2b", "4b3b", "4b4b"]
# The html for a table of typologies dividing the column into 10 equal parts, with width 100%
label_html = f"<table style=\"width: 100%;\">" + \
             "".join([f"<td style=\"width: 10%; text-align: center;\">{t}</td>" for t in typologies]) + "</table>"

cfg = OmegaConf.load("config/architext_gpt3.5_cfg.yaml")
elm_obj: ArchitextELM | None = None
elm_imgs = []


def run_elm(api_key: str, init_step: float, mutate_step: float, batch_size: float):
    init_step = int(init_step)
    mutate_step = int(mutate_step)
    batch_size = int(batch_size)

    os.environ["OPENAI_API_KEY"] = api_key
    global elm_obj
    if elm_obj is None:
        elm_obj = ArchitextELM(cfg)
    elm_obj.cfg.evo_init_steps = init_step
    elm_obj.cfg.evo_n_steps = init_step + mutate_step
    elm_obj.environment.batch_size = batch_size
    elm_obj.map_elites.env.batch_size = batch_size
    elm_obj.run()

    elm_imgs.append(get_imgs(elm_obj.map_elites.genomes))
    return elm_imgs[-1], len(elm_imgs) - 1


def slider_change(num: int):
    num = int(num)
    return elm_imgs[min(num, len(elm_imgs) - 1)]


def save():
    if elm_obj is None:
        return
    with open(f'recycled.pkl', 'wb') as f:
        pickle.dump(elm_obj.map_elites.recycled, f)
    with open(f'map.pkl', 'wb') as f:
        pickle.dump(elm_obj.map_elites.genomes, f)
    with open(f'history.pkl', 'wb') as f:
        pickle.dump(elm_obj.map_elites.history, f)


def on_select(evt: gr.SelectData):
    x, y = evt.index
    print("index: ", evt.index)

    genome = elm_obj.map_elites.genomes[x, y]
    if genome is None:
        return {}
    else:
        return genome.get_json()


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            api_key = gr.Textbox(
                label="OpenAI api key",
                max_lines=1,
            ).style(container=True)
            init_step = gr.Number(
                label="Initial random steps",
                show_label=True,
                value=1,
            ).style(container=True)
            mutate_step = gr.Number(
                label="mutation steps",
                show_label=True,
                value=1,
            ).style(container=True)
            batch_size = gr.Number(
                label="batch size",
                show_label=True,
                value=2,
            )
            btn = gr.Button("Run")
            design_json = gr.JSON(
                label="Design JSON",
                show_label=False,
                value="{}",
            )
            slider = gr.Slider(
                    minimum=0,
                    maximum=20,
            )
            save_btn = gr.Button("Save")
        with gr.Column(scale=10):
            with gr.Row():
                label = gr.HTML(
                    label="Typology",
                    show_label=False,
                    value=label_html,
                )

            with gr.Row():
                gallery = gr.Gallery(
                    label="Generated images", show_label=False, elem_id="gallery",
                ).style(columns=10, rows=10, container=False, object_fit="contain", height="100%")

    btn.click(run_elm, [api_key, init_step, mutate_step, batch_size], [gallery, slider], show_progress=True)
    gallery.select(on_select, None, design_json)
    save_btn.click(save, None, None)
    slider.change(slider_change, slider, gallery)



#if __name__ == "__main__":
demo.queue(concurrency_count=1, max_size=20).launch()
