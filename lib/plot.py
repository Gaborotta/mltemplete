# %%
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import plotly.express as px
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

    Example:
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
        go.Figure: plotlyのfigureオブジェクト

    Examples:
        fig = plot_scatter_matrix(pdf, cols=cols, color_col="species")
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

# %%


def plot_line(pdf: pd.DataFrame, x: str, y: str, color_col: str = None, isShow: bool = True) -> go.Figure:
    """折れ線図

    Args:
        pdf (pd.DataFrame): データセット
        x (str): X軸のカラム名
        y (str): Y軸のカラム名
        color_col (str, optional): マーカーの色分けに使用するカラム名.Noneは色分けしない Defaults to None.
        isShow (bool, optional): 呼び出し時にshowするかどうか。 Defaults to True.

    Returns:
        go.Figure: plotlyのfigureオブジェクト

    Examples:
        fig = plot_line(pdf=pdf, x=default_x, y=default_y,color_col=color_col, isShow=True)

    """

    fig = px.line(pdf, x=x, y=y, color=color_col, markers=True)

    if isShow:
        fig.show()
    return fig

# %%


def plot_muluti_violins(pdf: pd.DataFrame, cols: List[str], plot_all_points: bool = True, color_col: str = None, isShow: bool = True) -> List[go.Figure]:
    """指定した軸のバイオリンプロット

    Args:
        pdf (pd.DataFrame): データセット
        cols (List[str]): 表示する軸のカラム名のリスト
        plot_all_points (bool, optional): 点を全てプロットするかどうか. Defaults to True.
        color_col (str, optional): マーカーの色分けに使用するカラム名.Noneは色分けしない Defaults to None.
        isShow (bool, optional): 呼び出し時にshowするかどうか。 Defaults to True.

    Returns:
        List[go.Figure]: plotlyのfigureオブジェクト
    Examples:
        fig_list = plot_muluti_violins(pdf=pdf, cols=cols, plot_all_points=True,color_col=color_col, isShow=True)
    """

    points = "all" if plot_all_points else "outliers"

    fig_list = []
    for c in cols:
        fig = px.violin(
            pdf, y=c,
            # x=color_col,
            color=color_col,
            box=True,
            points=points,
            hover_name="species",
            # violinmode="overlay"
        )
        fig_list.append(fig)
        if isShow:
            fig.show()

    return fig_list


# %%
def plot_redar_chart(pdf: pd.DataFrame, cols: List[str], color_col: str, text_col: str,  isShow: bool = True) -> go.Figure:
    """color_col、text_colごとに指定した軸でレーダーチャートを作成。
       color_col、text_colが必須なので注意。

    Args:
        pdf (pd.DataFrame): データセット
        cols (List[str]): 表示する軸のカラム名のリスト
        color_col (str): マーカーの色分けに使用するカラム名
        text_col (str): ラベルに使用するカラム名
        isShow (bool, optional): 呼び出し時にshowするかどうか。 Defaults to True.

    Returns:
        go.Figure: plotlyのfigureオブジェクト
    Examples:
        fig = plot_redar_chart(pdf=pdf, cols=cols, color_col=color_col, text_col=text_col)
    """

    mean_pdf = (
        pdf.loc[:, cols + [color_col, text_col]]
        .groupby([color_col, text_col])
        .mean()
        .reset_index()
    )
    mean_pdf

    for tc in cols:

        tc_mean = mean_pdf[tc].mean()
        tc_std = mean_pdf[tc].std()

        mean_pdf[tc] = ((mean_pdf[tc] - tc_mean) / (tc_std)) * 5 + 1

    fig = go.Figure()
    for row in mean_pdf.to_numpy().tolist():
        row[0]
        fig.add_trace(go.Scatterpolar(
            r=row[2:],
            theta=cols,
            fill='toself',
            name=f'{row[0]}-{row[1]}'
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-6.5, 6.5])),
        showlegend=True
    )

    if isShow:
        fig.show()
    return fig
