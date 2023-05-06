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
WIDTH, HEIGHT = 10, 10
elm_imgs = [[Image.new('RGB', (256, 256), color=(255, 255, 255)) for _ in range(WIDTH * HEIGHT)]]


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


def load():
    global elm_obj
    with open(f'recycled.pkl', 'rb') as f:
        recycled = pickle.load(f)
    with open(f'map.pkl', 'rb') as f:
        genomes = pickle.load(f)
    with open(f'history.pkl', 'rb') as f:
        history = pickle.load(f)
    elm_obj = ArchitextELM(cfg)
    elm_obj.map_elites.recycled = recycled
    elm_obj.map_elites.genomes = genomes
    elm_obj.map_elites.history = history
    elm_imgs.append(get_imgs(elm_obj.map_elites.genomes))
    return elm_imgs[-1], len(elm_imgs) - 1


def on_select(evt: gr.SelectData):
    i = evt.index
    x = i // WIDTH
    y = i % WIDTH

    print("index: ", evt.index)
    if elm_obj is None:
        return {}

    genome = elm_obj.map_elites.genomes[x, y]
    if genome == 0.0:
        return {}
    else:
        return genome.to_dict(prompt_layout_only=False)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            api_key = gr.Textbox(
                label="OpenAI api key",
                max_lines=1,
            ).style(container=True)
            btn_load = gr.Button("Load")
            save_btn = gr.Button("Save")
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
            slider = gr.Slider(
                    minimum=0,
                    maximum=20,
                    step=1
            )
            btn = gr.Button("Run")
            btn_recenter = gr.Button("Recenter")
            design_json = gr.JSON(
                label="Design JSON",
                show_label=False,
                value="{}",
            )
        with gr.Column(scale=10):
            with gr.Row():
                label = gr.HTML(
                    label="Typology",
                    show_label=False,
                    value=label_html,
                )

            with gr.Row():
                gallery = gr.Gallery(
                    value=elm_imgs[0], label="Generated designs", show_label=False, elem_id="gallery",
                ).style(columns=WIDTH, rows=HEIGHT, container=False, object_fit="fill", height="100%")

    btn.click(run_elm, [api_key, init_step, mutate_step, batch_size], [gallery, slider], show_progress=True)
    btn_load.click(load, None, [gallery, slider], show_progress=True)
    gallery.select(on_select, None, design_json)
    save_btn.click(save, None, None)
    slider.change(slider_change, slider, gallery)



#if __name__ == "__main__":
demo.queue(concurrency_count=1, max_size=20).launch()
