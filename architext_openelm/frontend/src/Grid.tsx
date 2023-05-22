import {Streamlit, StreamlitComponentBase, withStreamlitConnection} from "streamlit-component-lib";
import React, { ReactNode } from "react"

interface State {
    Selected: number
}

class Grid extends StreamlitComponentBase<State> {
    public state = {Selected: -1}

    public render = (): ReactNode => {
        const paths = this.props.args["paths"]
        const titles = this.props.args["titles"]
        const div_style = this.props.args["div_style"]
        const table_style = this.props.args["table_style"]
        const img_style = this.props.args["img_style"]
        const num_cols = this.props.args["num_cols"]
        const col_labels = this.props.args["col_labels"]
        const row_labels = this.props.args["row_labels"]
        const selected = this.props.args["selected"]

        const { theme } = this.props

        const row_label_style: React.CSSProperties = {}
        const cell_style: React.CSSProperties = {}
        if (theme) {
            row_label_style.fontWeight = "bold"
            row_label_style.textAlign = "center"
            row_label_style.verticalAlign = "middle"
            row_label_style.width = `${100 / (num_cols + 1)}%`

            cell_style.textAlign = "center"
            cell_style.verticalAlign = "middle"
            cell_style.width = `${100 / (num_cols + 1)}%`
        }

        return (
            <div style={div_style}>
                <table style={table_style}>
                    <tbody>
                        <tr>
                            <td className="row_label"></td>
                            {col_labels.map((label: string) => (
                                <td className="row_label" style={row_label_style}>{label}</td>
                            ))}
                        </tr>
                        {row_labels.map((label: string, i: number) => (
                            <tr>
                                <td className="cell">{label}</td>
                                {paths.slice(i * num_cols, (i + 1) * num_cols).map((path: string, j: number) => (
                                    <td className="cell" style={cell_style}>
                                        <img className={this.is_selected(selected, i * num_cols + j) ? "selected" : "unselected"}
                                             style={img_style}
                                             id={"img_" + (i * num_cols + j).toString()}
                                             src={path}
                                             alt={titles[i * num_cols + j]}
                                             onClick={() => {
                                                 this.setState((prevState) => (
                                                     this.modifyCell(prevState.Selected, i * num_cols + j)), () => {
                                                        Streamlit.setComponentValue(this.state.Selected)
                                                 }
                                                )}
                                             }
                                             onLoad={this.onLoad}
                                        />
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        )
    }

    public onLoad = (): void => {
        Streamlit.setFrameHeight()
    }

    public is_selected = (selected: number, target: number): boolean => {
        return (selected === target) || (this.state.Selected === target)
    }

    public modifyCell = (prev: number, target: number): State => {
        let elem = document.getElementById("img_" + prev.toString())
        if (elem != null)
            elem.className = "unselected"
        else {
            // Streamlit connection magic might occasionally cause react state to reset to default
            // In that case, previous state is lost and we simply reset all
            let elems = document.getElementsByClassName("selected")
            for (let i = 0; i < elems.length; i++) {
                elems[i].className = "unselected"
            }
        }
        return {
            Selected: target
        }
    }
}
export default withStreamlitConnection(Grid)
