import gradio as gr
import pickle

def on_click(path):
    m = pickle.load(open(path, "rb"))
    dims = m.dims
    result = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            result.append(map[i, j].get_random_genome())


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                label="File path",
                #interactive=False,
                show_label=False,
                max_lines=1,
                placeholder="map.pkl",
            ).style(
                container=True,
            )
            btn = gr.Button("Show")
        with gr.Column(scale=12):
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery",
            ).style(columns=10, rows=10, container=False, object_fit="contain", height="100%")

    btn.click(on_click, text, gallery, show_progress=False)



#if __name__ == "__main__":
demo.queue(concurrency_count=1, max_size=20).launch()
