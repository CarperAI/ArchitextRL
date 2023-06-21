"""
This app is modified from:
https://github.com/vivien000/st-clickable-images/tree/master
"""

import os
import pathlib
import subprocess

import streamlit.components.v1 as components

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "clickable_images", url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")

    # Get the folder of the current file
    app_base_folder = pathlib.Path(__file__).parent
    # If frontend/build does not exist, run `npm run build`
    if not app_base_folder.joinpath("frontend/build").exists():
        subprocess.run(["curl", "https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh", "--output", "i.sh"],
                       cwd=str(app_base_folder / "frontend"))
        subprocess.run(["bash", "i.sh"], cwd=str(app_base_folder / "frontend"))
        subprocess.run(["bash", "build.sh"], cwd=str(app_base_folder))

    _component_func = components.declare_component(
        "st_grid", path=build_dir
    )


def st_grid(paths, titles, div_style, table_style, img_style, num_cols,
            col_labels=None, row_labels=None, key=None, selected=-1, default=-1):
    """Display a clickable map.
    """
    component_value = _component_func(
        paths=paths,
        titles=titles,
        div_style=div_style,
        table_style=table_style,
        img_style=img_style,
        num_cols=num_cols,
        col_labels=col_labels,
        row_labels=row_labels,
        key=key,
        selected=selected,
        default=default,
    )

    return component_value
