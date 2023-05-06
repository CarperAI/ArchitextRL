import json
import pathlib
import random
import subprocess
import numpy as np

import streamlit as st
from PIL import Image
from grid import st_grid
import io
import base64
import os
import pickle

from omegaconf import OmegaConf

from run_elm import ArchitextELM
from util import save_folder

def img_process(img_bytes):
    encoded_img = base64.b64encode(img_bytes).decode()
    return f"data:image/jpeg;base64,{encoded_img}"


def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format="jpeg")
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


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


def get_blank_grid():
    return [Image.new('RGB', (256, 256), color=(255, 255, 255)) for _ in range(WIDTH * HEIGHT)]


typologies = ["1b1b", "2b1b", "2b2b", "3b1b", "3b2b", "3b3b", "4b1b", "4b2b", "4b3b", "4b4b"]

# Initialize variables and state variables
try:
    cfg = OmegaConf.load("config/architext_gpt3.5_cfg.yaml")
except:
    cfg = OmegaConf.load("architext_openelm/config/architext_gpt3.5_cfg.yaml")


WIDTH, HEIGHT, Y_STEP = 5, 5, 0.1
# todo: multiple dims for map?
cfg.behavior_n_bins = WIDTH

st.session_state.setdefault("x_start", 0)
st.session_state.setdefault("y_start", 1.0)
st.session_state.setdefault("session_id",
                            "".join([random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(5)]))
st.session_state.setdefault("elm_imgs",
                            [get_blank_grid()]
                            )
st.session_state.setdefault("elm_obj", None)
st.session_state.setdefault("last_msg", "")

# create folder sessions/ if not exist
if not os.path.exists("sessions"):
    os.makedirs("sessions")
session_loc = "sessions/" + st.session_state["session_id"] + ".pkl"
if os.path.exists(session_loc):
    with open(session_loc, "rb") as f:
        loaded_state = pickle.load(f)
        st.session_state.update(loaded_state)


def update_elm_obj(elm_obj, init_step, mutate_step, batch_size):
    init_step = int(init_step)
    mutate_step = int(mutate_step)
    batch_size = int(batch_size)

    elm_obj.cfg.evo_init_steps = init_step
    elm_obj.cfg.evo_n_steps = init_step + mutate_step
    elm_obj.environment.batch_size = batch_size
    elm_obj.map_elites.env.batch_size = batch_size


def get_elm_obj(old_elm_obj=None):
    x_start = st.session_state["x_start"]
    y_start = st.session_state["y_start"]
    behavior_mode = {'genotype_ndim': 2,
                     'genotype_space': np.array([[y_start, y_start + HEIGHT * Y_STEP], [x_start, x_start + WIDTH]]).T
                     }

    elm_obj = ArchitextELM(cfg, behavior_mode=behavior_mode)
    if old_elm_obj is not None:
        elm_obj.map_elites.import_genomes(old_elm_obj.map_elites.export_genomes())

    return elm_obj


def run_elm(api_key: str, init_step: float, mutate_step: float, batch_size: float):
    os.environ["OPENAI_API_KEY"] = api_key

    if st.session_state["elm_obj"] is None:
        st.session_state["elm_obj"] = get_elm_obj()

    elm_obj = st.session_state["elm_obj"]
    update_elm_obj(elm_obj, init_step=init_step, mutate_step=mutate_step, batch_size=batch_size)
    elm_obj.run()

    #if "elm_imgs" in st.session_state:
    #    st.session_state["elm_imgs"] = [get_imgs(elm_obj.map_elites.genomes)]
    #else:
    #    st.session_state["elm_imgs"] = [get_imgs(elm_obj.map_elites.genomes)]

    st.session_state["elm_imgs"] = [get_imgs(elm_obj.map_elites.genomes)]


def save():
    if st.session_state["elm_obj"] is None:
        return
    elm_obj = st.session_state["elm_obj"]

    with open(str(save_folder / f'recycled.pkl'), 'wb') as f:
        pickle.dump(elm_obj.map_elites.recycled, f)
    with open(str(save_folder / f'map.pkl'), 'wb') as f:
        pickle.dump(elm_obj.map_elites.genomes, f)
    with open(str(save_folder / f'history.pkl'), 'wb') as f:
        pickle.dump(elm_obj.map_elites.history, f)


def load(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        with open(str(save_folder / f'recycled.pkl'), 'rb') as f:
            recycled = pickle.load(f)
        with open(str(save_folder / f'map.pkl'), 'rb') as f:
            genomes = pickle.load(f)
        with open(str(save_folder / f'history.pkl'), 'rb') as f:
            history = pickle.load(f)
    except Exception as e:
        st.session_state["last_msg"] = f"Error reading the files. Error message: {str(e)}"
        return


    if genomes.dims != (WIDTH, HEIGHT):
        print(f"Map size mismatch. Got {genomes.dims} != {(WIDTH, HEIGHT)}")
        return

    st.session_state["elm_obj"] = ArchitextELM(cfg)

    elm_obj = st.session_state["elm_obj"]

    elm_obj.map_elites.load_maps(genomes=genomes, recycled=recycled)
    elm_obj.map_elites.history = history

    st.session_state["elm_imgs"] = [get_imgs(elm_obj.map_elites.genomes)]


def recenter():
    last_clicked = st.session_state.get("last_clicked", -1)
    if last_clicked < 0 or last_clicked >= WIDTH * HEIGHT:
        return

    last_x = last_clicked % WIDTH
    last_y = last_clicked // WIDTH

    new_x_start = min(len(typologies) - WIDTH, max(0, last_x + st.session_state["x_start"] - WIDTH // 2))
    new_y_start = st.session_state["y_start"] + Y_STEP * (last_y - HEIGHT // 2)

    new_x = last_x - new_x_start + st.session_state["x_start"]
    new_y = HEIGHT // 2

    st.session_state["x_start"] = new_x_start
    st.session_state["y_start"] = new_y_start

    st.session_state["last_clicked"] = new_y * WIDTH + new_x

    st.session_state["elm_obj"] = get_elm_obj(st.session_state["elm_obj"])
    st.session_state["elm_imgs"] = [get_imgs(st.session_state["elm_obj"].map_elites.genomes)]


st.set_page_config(layout="wide")
col1, col2 = st.columns([3, 12])

with col1:
    api_key = st.text_input("OpenAI API")
    init_step = st.number_input("Init Step", value=1)
    mutate_step = st.number_input("Mutate Step", value=1)
    batch_size = st.number_input("Batch Size", value=2)

    #if len(st.session_state["elm_imgs"]) > 1:
    #    slider_index = st.slider("Step", 0,
    #                             len(st.session_state["elm_imgs"]) - 1,
    #                             len(st.session_state["elm_imgs"]) - 1)
    #else:
    #    slider_index = 0

    run = st.button("Run")
    do_load = st.button("Load")
    do_save = st.button("Save")
    do_recenter = st.button("Re-center")

if run:
    run_elm(api_key, init_step, mutate_step, batch_size)

if do_load:
    load(api_key)

if do_save:
    save()

if do_recenter:
    recenter()

with col2:
    assert st.session_state["x_start"] + WIDTH <= len(typologies)
    clicked = st_grid(
        [img_process(image_to_byte_array(img.convert('RGB'))) for img in st.session_state["elm_imgs"][0]],
        titles=[f"Image #{str(i)}" for i in range(len(st.session_state["elm_imgs"][0]))],
        div_style={"justify-content": "center", "width": "650px", "overflow": "auto"},
        table_style={"justify-content": "center", "width": "100%"},
        img_style={"cursor": "pointer"},
        num_cols=WIDTH,
        col_labels=typologies[st.session_state["x_start"] : st.session_state["x_start"] + WIDTH],
        row_labels=["{:.2f}".format(i * Y_STEP + st.session_state["y_start"]) for i in range(HEIGHT)],
        selected=int(st.session_state.get("last_clicked", -1)),
    )

if "last_msg" in st.session_state:
    st.write(st.session_state["last_msg"])

st.write("session id: " + st.session_state["session_id"])

if "prompt_tokens" in st.session_state:
    st.write(f"Prompt Tokens: {st.session_state['prompt_tokens']}")
if "tokens" in st.session_state:
    st.write(f"Total Tokens: {st.session_state['tokens']}")

if "elm_obj" in st.session_state and st.session_state["elm_obj"] is not None:
    st.write(f"Niches filled: {st.session_state['elm_obj'].map_elites.fitnesses.niches_filled}")
    st.write(f"Objects in recycle queue: {sum(obj is not None for obj in st.session_state['elm_obj'].map_elites.recycled)}")
    st.write(f"Max fitness: {st.session_state['elm_obj'].map_elites.fitnesses.maximum}")

    if "last_clicked" in st.session_state and st.session_state["last_clicked"] != -1:
        last_x = st.session_state["last_clicked"] % WIDTH
        last_y = st.session_state["last_clicked"] // WIDTH
        genome = st.session_state["elm_obj"].map_elites.genomes[(last_y, last_x)]
        if genome != 0.0:
            st.json(genome.design_json)

if clicked != "" and clicked != -1:
    st.session_state["last_clicked"] = int(clicked)
    st.experimental_rerun()

with open(session_loc, "wb") as f:
    pickle.dump({k: st.session_state[k] for k in ["elm_obj", "elm_imgs"]}, f)
    print(f"Session {st.session_state['session_id']} for ELM pictures saved")
