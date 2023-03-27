from pathlib import Path
import numpy as np
import random
import re
import textwrap
from shapely.geometry.polygon import Polygon

from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageFont, ImageColor
import gradio as gr
from random import shuffle
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model_path = Path('path/to/architext/model')
finetuned = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')

def merge_images(im1, im2):
    images = [im1, im2]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    combined = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        combined.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return combined

room_labels = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, 
                         "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}

architext_colors = [[0, 0, 0], [249, 222, 182], [195, 209, 217], [250, 120, 128], [126, 202, 234], [190, 0, 198], [255, 255, 255], 
                   [6, 53, 17], [17, 33, 58], [132, 151, 246], [197, 203, 159], [6, 53, 17],]

regex = re.compile(".*?\((.*?)\)")

def draw_polygons(polygons, colors, im_size=(256, 256), b_color="white", fpath=None):

    image = Image.new("RGB", im_size, color="white")
    draw = ImageDraw.Draw(image)

    for poly, color, in zip(polygons, colors):
        xy = poly.exterior.xy
        coords = np.dstack((xy[1], xy[0])).flatten()
        draw.polygon(list(coords), fill=(0, 0, 0))       
        
        #get inner polygon coordinates
        small_poly = poly.buffer(-1, resolution=32, cap_style=2, join_style=2, mitre_limit=5.0)
        if small_poly.geom_type == 'MultiPolygon':
            mycoordslist = [list(x.exterior.coords) for x in small_poly]
            for coord in mycoordslist:
                coords = np.dstack((np.array(coord)[:,1], np.array(coord)[:, 0])).flatten()
                draw.polygon(list(coords), fill=tuple(color)) 
        elif poly.geom_type == 'Polygon':
            #get inner polygon coordinates
            xy2 = small_poly.exterior.xy
            coords2 = np.dstack((xy2[1], xy2[0])).flatten()
            # draw it on canvas, with the appropriate colors
            draw.polygon(list(coords2), fill=tuple(color)) 

    #image = image.transpose(Image.FLIP_TOP_BOTTOM)

    if(fpath):
        image.save(fpath, format='png', quality=100, subsampling=0)
        np.save(fpath, np.array(image))

    return draw, image

def prompt_to_layout(user_prompt, fpath=None):
    
    model_prompt = '[User prompt] {} [Layout]'.format(user_prompt)
    input_ids = tokenizer(model_prompt, return_tensors='pt')
    output = finetuned.generate(**input_ids, do_sample=True, top_p=0.94, top_k=100, max_length=300)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    layout = output[0].split('[User prompt]')[1].split('[Layout]')[1].split(', ')
    spaces = [txt.split(':')[0].lstrip() for txt in layout]
    spaces = [re.sub(r'\d+', '', s) for s in spaces]

    coordinates = [txt.split(':')[1] for txt in layout]
    coordinates = [re.findall(regex, coord) for coord in coordinates]
    
    polygons = []
    for coord in coordinates:
        polygons.append([point.split(',') for point in coord])
        
    geom = []
    for poly in polygons:
        geom.append(Polygon(np.array(poly, dtype=int)))
        
    colors = [architext_colors[room_labels[space]] for space in spaces]
    
    _, im = draw_polygons(geom, colors, fpath=fpath)
    
    legend = Image.open("legend.png")
    
    im_new = Image.new('RGB', (256, 296))
    im_new.paste(legend, (0, 0))
    im_new.paste(im, (0, 40))
    
    return im_new, layout, output

def mut_txt2layout(mut_output):
    layout = mut_output[0].split('[User prompt]')[1].split('[Layout]')[1].split(', ')
    spaces = [txt.split(':')[0].lstrip() for txt in layout]
    spaces = [re.sub(r'\d+', '', s) for s in spaces]
    coordinates = [txt.split(':')[1] for txt in layout]
    coordinates = [re.findall(regex, coord) for coord in coordinates]

    polygons = []
    for coord in coordinates:
        polygons.append([point.split(',') for point in coord])

    geom = []
    for poly in polygons:
        geom.append(Polygon(np.array(poly, dtype=int)))

    colors = [architext_colors[room_labels[space]] for space in spaces]
    _, im = draw_polygons(geom, colors, fpath=None)

    legend = Image.open("legend.png")

    im_new = Image.new('RGB', (256, 296))
    im_new.paste(legend, (0, 0))
    im_new.paste(im, (0, 40))
    
    return im_new

def prompt_with_mutation(user_prompt, mut_rate, fpath=None):
    
    #Create initial layout based on prompt
    im, layout, output = prompt_to_layout(user_prompt)
        
    #Create mutated layout based on initial
    mut_len = int((1-mut_rate)*len(layout))
    index1 = random.randrange(0,len(layout)-mut_len)
    rooms = layout[index1:index1+mut_len]
    rooms = [room.lstrip().rstrip() for room in rooms]
    shuffle(rooms)
    rooms = ', '.join(rooms).lstrip().rstrip() + ','
    new_prompt = '[User prompt] {} [Layout] {}'.format(user_prompt, rooms)
    input_ids = tokenizer(new_prompt, return_tensors='pt')
    mut_output = finetuned.generate(**input_ids, do_sample=True, top_p=0.94, temperature=0.1, max_length=300)
    mut_output = tokenizer.batch_decode(mut_output, skip_special_tokens=True)
    mut_im = mut_txt2layout(mut_output)
    
    return im, mut_im

def gen_and_mutate(user_prompt, mutate=False, mut_rate=0.2):    
    if(mutate):
        im, mut_im = None, None
        while (mut_im is None):
            try:
                im, mut_im = prompt_with_mutation(user_prompt, mut_rate, fpath=None)
            except:
                pass
    else:
        mut_im=Image.open(r"C:\\Users\\user\\Desktop\\empty.png")
        im, _, _ = prompt_to_layout(user_prompt)
        
    return im, mut_im

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            textbox = gr.components.Textbox(placeholder='house with two bedrooms and one bathroom', lines="1", 
                                        label="DESCRIBE YOUR DESIGN")
            checkbox =  gr.components.Checkbox(label='Mutate')
            slider = gr.components.Slider(0.2, 0.8, step=0.1, label='Mutation rate')
            generate = gr.components.Button(value="Generate layout")
        generated = gr.components.Image(label='Generated Layout')
        mutated = gr.components.Image(label='Mutated Layout')
    with gr.Row():
        generate.click(gen_and_mutate, inputs=[textbox, checkbox, slider], outputs=[generated, mutated])
    
demo.launch()