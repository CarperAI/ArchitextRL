import random
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
_AVAILABLE_MODELS = ["finetuned", "Architext GPT-J", "GPT-3.5"]


def img_process(img_bytes):
    encoded_img = base64.b64encode(img_bytes).decode()
    return f"data:image/jpeg;base64,{encoded_img}"


def image_to_byte_array(image: Image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="jpeg")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def get_imgs(genomes, backgrounds=None):
    dims = genomes.dims
    result = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            if genomes[i, j] == 0.0:
                img = Image.new('RGB', (256, 256), color=(255, 255, 255))
            else:
                bg_img = backgrounds[i * dims[1] + j] if backgrounds is not None else None
                img = genomes[i, j].get_image(bg_img=bg_img)

            result.append(img)

    return result


def get_heat_imgs(genomes):
    dims = genomes.dims
    hist_len = genomes.history_length
    max_hist = max(1, np.max(np.where(genomes.top == hist_len - 1, -1, genomes.top)) + 1)
    result = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            hist = genomes.top[i, j] + 1 if genomes.top[i, j] < hist_len - 1 else 0
            intensity = int(hist / max_hist * 255)
            img = Image.new('RGB', (256, 256), color=(255, 255 - intensity, 255 - intensity))
            result.append(img)

    return result


def update_starts():
    if st.session_state.get("elm_obj", None) is not None:
        st.session_state["x_start"] = int(st.session_state["elm_obj"].environment.behavior_mode["genotype_space"][0, 1])
        st.session_state["y_start"] = st.session_state["elm_obj"].environment.behavior_mode["genotype_space"][0, 0]


def _update_images(elm_obj):
    st.session_state["elm_imgs"] = get_imgs(elm_obj.map_elites.genomes)
    st.session_state["heat_imgs"] = get_heat_imgs(elm_obj.map_elites.genomes)


def _discard_recycled(elm_obj):
    if elm_obj is not None and st.session_state.get("discard_recycled", False):
        elm_obj.map_elites.recycled = [None] * len(elm_obj.map_elites.recycled)
        elm_obj.map_elites.recycled_count = 0


def _collect_genomes(elm_obj):
    if elm_obj is not None:
        st.session_state["available_genomes"].extend(elm_obj.map_elites.export_genomes(include_recycled=False))


def _post_run():
    ...
    # with open(session_loc, "wb") as f:
    #    pickle.dump({k: st.session_state[k] for k in ["elm_obj", "elm_imgs"]}, f)


typologies = ["1b1b", "2b1b", "3b1b", "4b1b", "2b2b", "3b2b", "4b2b", "3b3b", "4b3b", "4b4b"]

# Initialize variables and state variables

st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([2, 4, 2])

# st.session_state.setdefault("width", 5)
# st.session_state.setdefault("height", 5)
st.session_state.setdefault("map_size", 5)
st.session_state.setdefault("y_step", 0.1)


def get_blank_grid():
    WIDTH, HEIGHT = st.session_state.get("map_size", 5), st.session_state.get("map_size", 5)
    return [Image.new('RGB', (256, 256), color=(255, 255, 255)) for _ in range(WIDTH * HEIGHT)]


st.session_state.setdefault("model", "Architext GPT-J")
st.session_state.setdefault("x_start", 0)
st.session_state.setdefault("y_start", 1.0)
st.session_state.setdefault("session_id",
                            "".join([random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(5)]))
st.session_state.setdefault("elm_imgs",
                            get_blank_grid()
                            )
st.session_state.setdefault("heat_imgs",
                            get_blank_grid())
st.session_state.setdefault("elm_obj", None)
st.session_state.setdefault("last_msg", "")
st.session_state.setdefault("last_clicked", (st.session_state["map_size"] + 1) * (st.session_state["map_size"] // 2))
st.session_state.setdefault("api_key", "")
st.session_state.setdefault("available_genomes", [])  # save all genomes that show up in the map at least once

update_starts()


def get_cfg():
    model = st.session_state["model"]
    if model == "Architext GPT-J":
        try:
            cfg = OmegaConf.load("config/architext_cfg.yaml")
        except:
            cfg = OmegaConf.load("architext_openelm/config/architext_cfg.yaml")
    elif model == "GPT-3.5":
        try:
            cfg = OmegaConf.load("config/architext_gpt3.5_cfg.yaml")
        except:
            cfg = OmegaConf.load("architext_openelm/config/architext_gpt3.5_cfg.yaml")
    elif model == "finetuned":
        try:
            cfg = OmegaConf.load("config/architext_finetuned.yaml")
        except:
            cfg = OmegaConf.load("architext_openelm/config/architext_finetuned.yaml")

    else:
        raise ValueError("Model not supported")
    cfg.behavior_n_bins = st.session_state["map_size"]

    return cfg


# create folder sessions/ if not exist
if not os.path.exists("sessions"):
    os.makedirs("sessions")


def update_elm_obj(elm_obj, init_step, mutate_step, batch_size):
    init_step = int(init_step)
    mutate_step = int(mutate_step)
    batch_size = int(batch_size)

    elm_obj.cfg.evo_init_steps = init_step
    elm_obj.cfg.evo_n_steps = init_step + mutate_step
    elm_obj.environment.batch_size = batch_size
    elm_obj.map_elites.env.batch_size = batch_size


def _need_reload(elm_obj, x_start, y_start, width, height, y_step) -> bool:
    if elm_obj is None:
        return True
    if elm_obj.environment.behavior_mode["genotype_space"][0, 1] != y_start + height * y_step:
        return True
    if elm_obj.environment.behavior_mode["genotype_space"][0, 0] != y_start:
        return True
    if elm_obj.environment.behavior_mode["genotype_space"][1, 1] != x_start + width:
        return True
    if elm_obj.environment.behavior_mode["genotype_space"][1, 0] != x_start:
        return True
    return False


def get_elm_obj(old_elm_obj=None):
    x_start = st.session_state["x_start"]
    y_start = st.session_state["y_start"]
    width, height, y_step = st.session_state["map_size"], st.session_state["map_size"], st.session_state["y_step"]
    behavior_mode = {'genotype_ndim': 2,
                     'genotype_space': np.array([[y_start, y_start + height * y_step], [x_start, x_start + width]]).T
                     }

    if not _need_reload(st.session_state.get("elm_obj", None), x_start, y_start, width, height, y_step):
        return st.session_state["elm_obj"]

    elm_obj = ArchitextELM(get_cfg(), behavior_mode=behavior_mode)
    if old_elm_obj is not None:
        elm_obj.map_elites.import_genomes(old_elm_obj.map_elites.export_genomes())
    if st.session_state.get("available_genomes", []):
        elm_obj.map_elites.import_genomes(st.session_state["available_genomes"])

    _update_images(elm_obj)
    _discard_recycled(elm_obj)
    save()
    return elm_obj


def run_elm(api_key: str, init_step: float, mutate_step: float, batch_size: float, placeholder=None):
    os.environ["OPENAI_API_KEY"] = api_key
    print(get_cfg())

    if st.session_state.get("elm_obj", None) is None:
        st.session_state["elm_obj"] = get_elm_obj()

    elm_obj = st.session_state["elm_obj"]
    update_elm_obj(elm_obj, init_step=init_step, mutate_step=mutate_step, batch_size=batch_size)

    if placeholder is not None:
        pbar = placeholder.progress(1, text="Generation in progress. Please wait.")
    else:
        pbar = None
    elm_obj.run(progress_bar=pbar)

    _update_images(elm_obj)
    _discard_recycled(elm_obj)
    _collect_genomes(elm_obj)
    _post_run()
    save()


def run():
    # todo: Use Semaphore for models separately? GPT-J depends on GPU/RAM capacity and GPT-3.5 depends on rate limit
    with _lock:
        try:
            run_elm(st.session_state["api_key"],
                    st.session_state["init_step"],
                    st.session_state["mutate_step"],
                    st.session_state["batch_size"],
                    placeholder=pbar_placeholder)
        except Exception as e:
            st.session_state["last_msg"] = str(e)


def export(map_elites):
    return {
        "recycled": map_elites.recycled,
        "genomes": map_elites.genomes,
        "history": map_elites.history,
        "available_genomes": st.session_state["available_genomes"],
        "x_start": st.session_state["x_start"],
        "y_start": st.session_state["y_start"],
        "y_step": st.session_state["y_step"]
    }


def save():
    if st.session_state.get("elm_obj", None) is None:
        return
    elm_obj = st.session_state["elm_obj"]

    with open(str(save_folder / f'saved_{st.session_state["session_id"]}.pkl'), 'wb') as f:
        pickle.dump(export(elm_obj.map_elites), f)


def load(api_key):
    os.environ["OPENAI_API_KEY"] = api_key

    try:
        with open(str(save_folder / f'saved_{st.session_state["session_id"]}.pkl'), 'rb') as f:
            loaded_state = pickle.load(f)
        recycled, genomes, history, available_genomes, x_start, y_start, map_y_step = \
            loaded_state["recycled"], loaded_state["genomes"], loaded_state["history"], \
            loaded_state.get("available_genomes", []), \
            loaded_state["x_start"], loaded_state["y_start"], loaded_state.get('y_step', 0.1)
        st.session_state["x_start"] = x_start
        st.session_state["y_start"] = y_start

    except Exception as e:
        st.session_state["last_msg"] = f"Error reading the files. Error message: {str(e)}"
        return

    assert genomes.dims[0] == genomes.dims[1], "Map size must be square"
    st.session_state["last_msg"] = f"Map size:  {genomes.dims}"
    st.session_state["last_clicked"] = (genomes.dims[0] * (genomes.dims[1]) // 2) + genomes.dims[0] // 2

    st.session_state["available_genomes"] = available_genomes
    st.session_state["elm_obj"] = get_elm_obj(None)

    elm_obj = st.session_state["elm_obj"]
    elm_obj.map_elites.load_maps(genomes=genomes, recycled=recycled)
    elm_obj.map_elites.history = history

    st.session_state["elm_imgs"] = get_imgs(elm_obj.map_elites.genomes)

    _post_run()
    return genomes.dims[0], map_y_step


def recenter():
    if "map_size" not in st.session_state:
        # not initialized yet... Not sure how Streamlit works on these functions but such case does happen
        return

    WIDTH, HEIGHT, Y_STEP = st.session_state["map_size"], st.session_state["map_size"], st.session_state["y_step"]

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


def upload_submit():
    with _lock:
        bytes_data = st.session_state.file_uploader.getvalue()
        with open(save_path, "wb") as f:
            f.write(bytes_data)
        new_size, new_y_step = load(st.session_state["api_key"])
        st.session_state.map_size = new_size
        st.session_state.y_step = new_y_step


# ----- Components and rendering -----

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
    size_slider_placeholder = st.empty()
    step_slider_placeholder = st.empty()

    index = 0
    for i, m in enumerate(_AVAILABLE_MODELS):
        if st.session_state["model"] == m:
            index = i
    model = st.radio("Model", _AVAILABLE_MODELS, index=index, key="model")
    st.checkbox("Discard out-of-bound genomes", value=True, key="discard_recycled")

    # Note that we don't even save the api key in the session state
    if model == "GPT-3.5":
        api_key = st.text_input("OpenAI API Key", key="api_key")

    with st.form("parameters", clear_on_submit=False):
        st.number_input("Init Step", value=1, key="init_step")
        st.number_input("Mutate Step", value=1, key="mutate_step")
        st.number_input("Batch Size", value=2, key="batch_size")

        run = st.form_submit_button("Run", on_click=run)


    save_path = save_folder / f'saved_{st.session_state["session_id"]}.pkl'

    do_recenter = st.button("Re-center", on_click=recenter)

    with st.form("file uploader", clear_on_submit=True):
        uploaded_file = st.file_uploader("Upload your map", type=["pkl"], key="file_uploader")
        submitted = st.form_submit_button("Upload", on_click=upload_submit)

    # `upload_submit` changes sliders. Therefore, they need to be instantiated after
    with size_slider_placeholder:
        map_size = st.slider("Map size", min_value=3, max_value=10, key="map_size",
                             value=5, step=1, on_change=recenter)
    with step_slider_placeholder:
        y_step = st.slider("y_step", min_value=0.05, max_value=0.5, key="y_step",
                           value=0.1, step=0.05, on_change=recenter)

    if os.path.exists(save_path):
        with open(save_path, "rb") as file:
            st.download_button(
                label="Download the map",
                data=file,
                file_name=f'saved_{st.session_state["session_id"]}.pkl',
            )

with col2:
    map_type = st.radio("Map type", ["Genomes", "Heat map"], index=0, horizontal=True)
    # Overlay mode is not supported yet... How do we make it look nice?
    #if map_type == "Overlay":
        #imgs = get_imgs(st.session_state["elm_obj"].map_elites.genomes, backgrounds=st.session_state["heat_imgs"])
        #imgs = [PIL.Image.blend(img1, img2, 0.5) for img1, img2 in zip(st.session_state["elm_imgs"], st.session_state["heat_imgs"])]
    if map_type == "Heat map":
        imgs = st.session_state["heat_imgs"]
    elif map_type == "Genomes":
        imgs = st.session_state["elm_imgs"]

    if st.session_state["x_start"] + st.session_state["map_size"] > len(typologies):
        # out-of-bound can happen the x_start is non-zero and the map_size slider is changed to a big number
        assert st.session_state["map_size"] <= len(typologies)
        st.session_state["x_start"] = len(typologies) - st.session_state["map_size"]

    pbar_placeholder = st.empty()
    with pbar_placeholder:
        clicked = st_grid(
            [img_process(image_to_byte_array(img.convert('RGB'))) for img in imgs],
            titles=[f"Image #{str(i)}" for i in range(len(imgs))],
            div_style={"justify-content": "center", "width": "650px", "overflow": "auto"},
            table_style={"justify-content": "center", "width": "100%"},
            img_style={"cursor": "pointer"},
            num_cols=st.session_state["map_size"],
            col_labels=typologies[
                       st.session_state["x_start"]: st.session_state["x_start"] + st.session_state["map_size"]],
            row_labels=["{:.2f}".format(i * st.session_state["y_step"] + st.session_state["y_start"]) for i in
                        range(st.session_state["map_size"])],
            selected=int(st.session_state.get("last_clicked", -1)),
            key="last_clicked",
            default=-1,
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
        st.write(f"Generated in this session: {len(st.session_state['available_genomes'])}")
        st.write(f"Niches filled: {st.session_state['elm_obj'].map_elites.fitnesses.niches_filled}")
        st.write(
            f"Objects in recycle queue: {st.session_state['elm_obj'].map_elites.recycled_count}")
        st.write(f"Max fitness: {st.session_state['elm_obj'].map_elites.fitnesses.maximum}")

        if "last_clicked" in st.session_state and st.session_state["last_clicked"] != -1:
            last_x = st.session_state["last_clicked"] % st.session_state["map_size"]
            last_y = st.session_state["last_clicked"] // st.session_state["map_size"]
            genome = st.session_state["elm_obj"].map_elites.genomes[(last_y, last_x)]
            if genome != 0.0:
                st.write(genome.typologies_from[genome.typology()])
                st.json(genome.design_json)


_post_run()
st.session_state["last_msg"] = ""
