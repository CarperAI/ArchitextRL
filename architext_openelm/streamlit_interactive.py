import functools
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
from threading import Lock
from run_elm import ArchitextELM
from util import save_folder

_lock = Lock()

def img_process(img_bytes):
    encoded_img = base64.b64encode(img_bytes).decode()
    return f"data:image/jpeg;base64,{encoded_img}"


def image_to_byte_array(image: Image):
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


def update_starts():
    if st.session_state.get("elm_obj", None) is not None:
        st.session_state["x_start"] = int(st.session_state["elm_obj"].environment.behavior_mode["genotype_space"][0, 1])
        st.session_state["y_start"] = st.session_state["elm_obj"].environment.behavior_mode["genotype_space"][0, 0]


def _post_run():
    with open(session_loc, "wb") as f:
        pickle.dump({k: st.session_state[k] for k in ["elm_obj", "elm_imgs"]}, f)


typologies = ["1b1b", "2b1b", "3b1b", "4b1b", "2b2b", "3b2b", "4b2b", "3b3b", "4b3b", "4b4b"]

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

update_starts()

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


def run_elm(api_key: str, init_step: float, mutate_step: float, batch_size: float, placeholder=None):
    os.environ["OPENAI_API_KEY"] = api_key

    if st.session_state["elm_obj"] is None:
        st.session_state["elm_obj"] = get_elm_obj()

    elm_obj = st.session_state["elm_obj"]
    update_elm_obj(elm_obj, init_step=init_step, mutate_step=mutate_step, batch_size=batch_size)

    if placeholder is not None:
        pbar = placeholder.progress(1, text="Generation in progress. Please wait.")
    else:
        pbar = None
    elm_obj.run(progress_bar=pbar)

    st.session_state["elm_imgs"] = [get_imgs(elm_obj.map_elites.genomes)]
    _post_run()
    save()


def export(map_elites):
    return {
        "recycled": map_elites.recycled,
        "genomes": map_elites.genomes,
        "history": map_elites.history,
        "x_start": st.session_state["x_start"],
        "y_start": st.session_state["y_start"],
    }


def save():
    if st.session_state["elm_obj"] is None:
        return
    elm_obj = st.session_state["elm_obj"]

    with open(str(save_folder / f'saved_{st.session_state["session_id"]}.pkl'), 'wb') as f:
        pickle.dump(export(elm_obj.map_elites), f)
    st.experimental_rerun()


def load(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        with open(str(save_folder / f'saved_{st.session_state["session_id"]}.pkl'), 'rb') as f:
            loaded_state = pickle.load(f)
        recycled, genomes, history, x_start, y_start = \
            loaded_state["recycled"], loaded_state["genomes"], loaded_state["history"], \
            loaded_state["x_start"], loaded_state["y_start"]
        st.session_state["x_start"] = x_start
        st.session_state["y_start"] = y_start

    except Exception as e:
        st.session_state["last_msg"] = f"Error reading the files. Error message: {str(e)}"
        return

    if genomes.dims != (WIDTH, HEIGHT):
        last_msg = f"Map size mismatch. Got {genomes.dims} != {(WIDTH, HEIGHT)}"
        st.session_state["last_msg"] = last_msg
        return

    st.session_state["elm_obj"] = get_elm_obj(None)

    elm_obj = st.session_state["elm_obj"]
    elm_obj.map_elites.load_maps(genomes=genomes, recycled=recycled)
    elm_obj.map_elites.history = history

    st.session_state["elm_imgs"] = [get_imgs(elm_obj.map_elites.genomes)]

    _post_run()


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


# ----- Components and rendering -----


st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([2, 4, 2])

st.write("# Architext interactive map")
st.write("## How it works")
st.write("1. Paste your OAI key (we do not save it, as you can check from our source code:"
         "https://github.com/CarperAI/ArchitextRL/tree/main/architext_openelm).")
st.write("2. Choose the parameters for the map generation.")
st.write("3. Click `run`.")
st.write("*DO NOT REFRESH.* Every session has a unique id and will be lost upon refresh. Make sure to download "
         "the map if you want to keep it.")
st.write("## Other buttons and options")
st.write("- `Re-center` button is the biggest player here: if you select a grid on the map, clicking "
         " `Re-center` will recenter the map for you unless an axis could go out-of-bound. "
         "A new MAPElites object will be generated in this process, but all genomes will be copied over "
         "(and some invalid genomes might revive because the range of the diversity metrics changed).")
st.write("- `Save` button will save the state of the map into a pickle file.")
st.write("- `Download map` button will show if you have a saved map. "
         "It allows you to download the pkl file and keep it.")
st.write("- `Load the map` button will show if you have a saved map. It loads from the pkl file on server.")
st.write("- Or you can upload your local map through the file uploader. ")

with col1:
    api_key = st.text_input("OpenAI API")
    init_step = st.number_input("Init Step", value=1)
    mutate_step = st.number_input("Mutate Step", value=1)
    batch_size = st.number_input("Batch Size", value=2)

    save_path = save_folder / f'saved_{st.session_state["session_id"]}.pkl'

    run = st.button("Run")
    do_recenter = st.button("Re-center")
    with st.form("file uploader", clear_on_submit=True):
        uploaded_file = st.file_uploader("Upload your map", type=["pkl"])
        submitted = st.form_submit_button("Upload")

        if submitted and uploaded_file is not None:
            with _lock:
                bytes_data = uploaded_file.getvalue()
                with open(save_path, "wb") as f:
                    f.write(bytes_data)
                load(api_key)
                st.experimental_rerun()

    if os.path.exists(save_path):
        do_load = st.button("Load the map")
    else:
        do_load = False
    do_save = st.button("Save the map")
    if os.path.exists(save_path):
        with open(save_path, "rb") as file:
            st.download_button(
                label="Download the map",
                data=file,
                file_name=f'saved_{st.session_state["session_id"]}.pkl',
            )


if do_load:
    load(api_key)

if do_save:
    save()

if do_recenter:
    recenter()

with col2:
    assert st.session_state["x_start"] + WIDTH <= len(typologies)
    pbar_placeholder = st.empty()
    with pbar_placeholder:
        clicked = st_grid(
            [img_process(image_to_byte_array(img.convert('RGB'))) for img in st.session_state["elm_imgs"][0]],
            titles=[f"Image #{str(i)}" for i in range(len(st.session_state["elm_imgs"][0]))],
            div_style={"justify-content": "center", "width": "650px", "overflow": "auto"},
            table_style={"justify-content": "center", "width": "100%"},
            img_style={"cursor": "pointer"},
            num_cols=WIDTH,
            col_labels=typologies[st.session_state["x_start"]: st.session_state["x_start"] + WIDTH],
            row_labels=["{:.2f}".format(i * Y_STEP + st.session_state["y_start"]) for i in range(HEIGHT)],
            selected=int(st.session_state.get("last_clicked", -1)),
        )

with col3:
    if "last_msg" in st.session_state:
        st.write(st.session_state["last_msg"])

    st.write("session id: " + st.session_state["session_id"])

    if "prompt_tokens" in st.session_state:
        st.write(f"Prompt Tokens: {st.session_state['prompt_tokens']}")
    if "tokens" in st.session_state:
        st.write(f"Total Tokens: {st.session_state['tokens']}")

    if st.session_state.get("elm_obj", None) is not None:
        st.write(f"Niches filled: {st.session_state['elm_obj'].map_elites.fitnesses.niches_filled}")
        st.write(
            f"Objects in recycle queue: {sum(obj is not None for obj in st.session_state['elm_obj'].map_elites.recycled)}")
        st.write(f"Max fitness: {st.session_state['elm_obj'].map_elites.fitnesses.maximum}")

        if "last_clicked" in st.session_state and st.session_state["last_clicked"] != -1:
            last_x = st.session_state["last_clicked"] % WIDTH
            last_y = st.session_state["last_clicked"] // WIDTH
            genome = st.session_state["elm_obj"].map_elites.genomes[(last_y, last_x)]
            if genome != 0.0:
                st.write(genome.typologies_from[genome.typology()])
                st.json(genome.design_json)

if run:
    with _lock:
        run_elm(api_key, init_step, mutate_step, batch_size, placeholder=pbar_placeholder)


if clicked != "" and clicked != -1:
    st.session_state["last_clicked"] = int(clicked)
    st.experimental_rerun()

_post_run()
st.session_state["last_msg"] = ""

