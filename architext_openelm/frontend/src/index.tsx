import { Streamlit, RenderData } from "streamlit-component-lib"

function create_td_class(text: string): HTMLTableDataCellElement {
    let td = document.createElement("td")
    td.className = text
    return td
}

function create_tr(): HTMLTableRowElement {
    let tr = document.createElement("tr")
    tr.style.width = "100%"
    return tr
}

function onRender(event: Event): void {
    const data = (event as CustomEvent<RenderData>).detail

    // Remove existing content
    let child = document.body.lastElementChild;
    if (child) {
        document.body.removeChild(child)
    }

    // Add style the image container
    let div = document.body.appendChild(document.createElement("div"))
    for (let key in data.args["div_style"]) {
        div.style[key as any] = data.args["div_style"][key]
    }

    let num_cols = data.args["num_cols"]

    // Create css style for row_label and cell
    let style = document.createElement("style")
    style.innerHTML = `
        .row_label {
            font-weight: bold;
            text-align: center;
            vertical-align: middle;
            width: ${100 / (num_cols + 1)}%;
        }
        .cell {
            text-align: center;
            vertical-align: middle;
            width: ${100 / (num_cols + 1)}%;
        }
        img {
            max-width: 100%;
            max-height: 100%;
            display: block;
            margin: 0 auto;
        }
        img.selected{
            border: 2px solid blue;
        }
        img.unselected{
            border: 1px solid black;
        }
}
    `
    document.head.appendChild(style)

    // Add and style all images
    let imagesLoaded = 0

    // Create a table of images loaded from data.args["paths"] with 5 columns
    div.appendChild(document.createElement("table"))
    let table = div.lastElementChild as HTMLTableElement
    for (let key in data.args["table_style"]) {
        table.style[key as any] = data.args["table_style"][key]
    }


    // Add column names
    table.appendChild(create_tr())
    let row = table.lastElementChild as HTMLTableRowElement
    row.appendChild(create_td_class("row_label"))
    for (let i = 0; i < num_cols; i++) {
        row.appendChild(create_td_class("row_label"))
        let cell = row.lastElementChild as HTMLTableCellElement
        cell.innerText = data.args["col_labels"][i]
    }

    // Form the table by adding images
    for (let i = 0; i < data.args["paths"].length; i++) {
        if (i % num_cols === 0) {
            let td = create_td_class("cell")
            td.innerText = data.args["row_labels"][Math.floor(i / num_cols)]

            table.appendChild(create_tr()).appendChild(td)
        }
        let row = table.lastElementChild as HTMLTableRowElement
        row.appendChild(create_td_class("cell"))
        let cell = row.lastElementChild as HTMLTableCellElement
        cell.appendChild(document.createElement("img"))
        let img = cell.lastElementChild as HTMLImageElement
        for (let key in data.args["img_style"]) {
            img.style[key as any] = data.args["img_style"][key]
        }

        // Highlight the selected image
        img.className = i == data.args["selected"] ? "selected" : "unselected"

        img.src = data.args["paths"][i]
        if (data.args["titles"].length > i) {
            img.title = data.args["titles"][i]
        }
        img.onclick = function (): void {
            Streamlit.setComponentValue(i)
        }
        // eslint-disable-next-line
        img.onload = function (): void {
            imagesLoaded++
            if (imagesLoaded === data.args["paths"].length) {
                Streamlit.setFrameHeight()
            }
        }
    }


}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
Streamlit.setComponentReady()