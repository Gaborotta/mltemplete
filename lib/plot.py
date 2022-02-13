# %%
from matplotlib import markers
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import plotly.express as px
import numpy as np
from typing import List
# %%
pdf = px.data.iris()
pdf.head(1)
pdf.columns
# %%
cols = [
    'sepal_length', 'sepal_width',
    'petal_length', 'petal_width',
    # 'species', 'species_id'
]
default_x = "sepal_length"
default_y = "sepal_width"
color_col = "species_id"
text_col = "species"

# %%


def plot_scatter_select_axis(pdf: pd.DataFrame, cols: List[str], color_col: str = None, text_col: str = None, isShow: bool = True) -> go.Figure:
    """軸の選択が可能な散布図

    Args:
        pdf (pd.DataFrame): データセット
        cols (List[str]): 軸として選択できるカラム名のリスト
        color_col (str, optional): マーカの色分けに使用するカラム名。数値型のみ。Noneは色分けしない。Defaults to None.
        text_col (str, optional): ラベルに使用するカラム名。Noneはラベルを使用しない。Defaults to None.
        isShow (bool, optional):呼び出し時にshowするかどうか。 Defaults to True.

    Returns:
        go.Figure: plotlyのfigureオブジェクト

    example:
        fig = plot_scatter_select_axis(
            pdf=pdf, cols=cols,
            color_col=color_col, text_col=text_col, isShow=True
        )
    """

    default_x = cols[0]
    default_y = cols[0]
    hovertext = pdf[text_col] if text_col is not None else None
    color = pdf[color_col] if color_col is not None else None

    print(default_x, default_y)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=pdf[default_x], y=pdf[default_y],
            mode='markers',
            marker=dict(
                color=color,
                colorscale=qualitative.Light24,
                line_width=1,  # マーカーの線の太さ
            ),
            hovertext=hovertext,
        )
    )
    # fig.show()

    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                buttons=[
                    dict(
                        args=[{axis: [pdf[c]]}],
                        label=c,
                        method="update"
                    )
                    for i, c in enumerate(cols)
                ],
                direction="down", pad={"r": 10, "t": 10},
                showactive=True, x=0.05,
                xanchor="left", y=y_posi,
                yanchor="top"
            )
            for axis, y_posi in [("x", 1.45), ("y", 1.30)]
        ]
    )

    fig.update_layout(
        annotations=[
            go.layout.Annotation(text=axis, x=0, xref="paper", y=y_posi, yref="paper",
                                 align="left", showarrow=False)
            for axis, y_posi in [("x", 1.38), ("y", 1.23)]
        ])

    if isShow:
        fig.show()
    return fig

# %%


def plot_scatter_matrix(pdf: pd.DataFrame, cols: List[str] = None, color_col: str = None, isShow: bool = True) -> go.Figure:
    """散布図行列

    Args:
        pdf (pd.DataFrame): データセット
        cols (List[str], optional): 軸として使用するカラム名のリスト Defaults to None.
        color_col (str, optional): マーカーの色分けに使用するカラム名.Noneは色分けしない Defaults to None.
        isShow (bool, optional): 呼び出し時にshowするかどうか。 Defaults to True.

    Returns:
        go.Figure: [description]

    examples:
        plot_scatter_matrix(pdf, color_col="species")
    """
    cols = cols if cols is not None else pdf.columns
    fig = px.scatter_matrix(
        pdf,
        dimensions=cols,
        color=color_col
    )
    fig.update_traces(diagonal_visible=False)

    if isShow:
        fig.show()
    return fig
