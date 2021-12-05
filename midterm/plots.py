import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import seaborn as sns
import yfinance


def financial_crisis_times():
    """Collect times of financial crisis into a dataframe"""
    great_recession = ["2007-12-01", "2009-06-01"]
    european_crisis = ["2010-04-01", "2012-06-01"]
    covid = ["2020-03-01", "2020-08-01"]


def load_data(gtype: str = "spearman_25"):
    gtype = gtype + "/" if "/" not in gtype else gtype
    path = gtype + "graph_features_transformed.csv"
    df = pd.read_csv(path)
    df.set_index("Time", inplace=True)
    df.drop(labels="Unnamed: 0", axis=1)

    df = df[
        [
            "num_edges",
            "density",
            "avg_clustering",
            "average_weight",
            "avg_curvature",
            "sp500",
        ]
    ]

    df = df.rename(
        columns={
            "num_edges": "Number of Edges",
            "density": "Density",
            "avg_clustering": "Average Clustering Coef.",
            "average_weight": "Average Edge Weight",
            "avg_curvature": "Average Curvature",
            "sp500": "S&P500",
        }
    )

    return df


def triang(cormat, triang="lower"):

    if triang == "upper":
        rstri = pd.DataFrame(
            np.triu(cormat.values), index=cormat.index, columns=cormat.columns
        ).round(3)
        rstri = rstri.iloc[:, 1:]
        rstri.drop(rstri.tail(1).index, inplace=True)

    if triang == "lower":
        rstri = pd.DataFrame(
            np.tril(cormat.values), index=cormat.index, columns=cormat.columns
        ).round(3)
        rstri = rstri.iloc[:, :-1]
        rstri.drop(rstri.head(1).index, inplace=True)

    rstri.replace(to_replace=[0, 1], value="", inplace=True)

    return rstri


def heatmap(show: bool = False, gtype: str = "spearman_25"):
    df = load_data(gtype)
    pd.set_option("display.max_columns", 10)
    corr = df.corr().astype(float)

    fig = sns.heatmap(
        corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="RdBu",
        annot=True,
    )
    if show:
        plt.show()
    return fig


def edges_over_time(show: bool = False, gtype: str = "spearman_25"):
    date_decoder = {"25": "Feb 2005", "125": "Jul 2005"}
    lagnum = gtype.split("_")[1]
    lag_number = lagnum[:-1] if "/" in lagnum else lagnum
    df = load_data(gtype)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Number of Edges",
            mode="lines",
            x=df.index,
            y=df["Number of Edges"],
            xperiodalignment="end",
        )
    )
    fig.add_shape(  # Add rectangle for Great Recession
        type="rect",
        x0="2007-12-01",
        y0=0,
        x1="2009-06-01",
        y1=20000,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for European Debt Crisis
        type="rect",
        x0="2010-04-01",
        y0=0,
        x1="2012-06-01",
        y1=20000,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for COVID-19
        type="rect",
        x0="2020-03-01",
        y0=0,
        x1="2020-08-01",
        y1=20000,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.update_xaxes(dtick="M6", tickformat="%b\n%Y")
    fig.update_layout(
        title=f"Number of Edges Over Time, {date_decoder[lag_number]} - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Edge Number",
    )
    if show:
        fig.show()
    return fig


def curvature(show: bool = False, gtype: str = "spearman_25"):
    date_decoder = {"25": "Feb 2005", "125": "Jul 2005"}
    lagnum = gtype.split("_")[1]
    lag_number = lagnum[:-1] if "/" in lagnum else lagnum
    df = load_data(gtype)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Average Curvature",
            mode="lines",
            x=df.index,
            y=df["Average Curvature"],
            xperiodalignment="end",
        )
    )
    fig.add_shape(  # Add rectangle for Great Recession
        type="rect",
        x0="2007-12-01",
        y0=0.6,
        x1="2009-06-01",
        y1=1,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for European Debt Crisis
        type="rect",
        x0="2010-04-01",
        y0=0.6,
        x1="2012-06-01",
        y1=1,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for COVID-19
        type="rect",
        x0="2020-03-01",
        y0=0.6,
        x1="2020-08-01",
        y1=1,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.update_xaxes(dtick="M6", tickformat="%b\n%Y")
    fig.update_layout(
        title=f"Average Curvature Over Time, {date_decoder[lag_number]} - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Average Curvature",
    )
    if show:
        fig.show()
    return fig


def clustering_coef(show: bool = False, gtype: str = "spearman_25"):
    date_decoder = {"25": "Feb 2005", "125": "Jul 2005"}
    lagnum = gtype.split("_")[1]
    lag_number = lagnum[:-1] if "/" in lagnum else lagnum
    df = load_data(gtype)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Average Clustering Coef.",
            mode="lines",
            x=df.index,
            y=df["Average Clustering Coef."],
            xperiodalignment="end",
        )
    )
    fig.add_shape(  # Add rectangle for Great Recession
        type="rect",
        x0="2007-12-01",
        y0=0,
        x1="2009-06-01",
        y1=1,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for European Debt Crisis
        type="rect",
        x0="2010-04-01",
        y0=0,
        x1="2012-06-01",
        y1=1,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for COVID-19
        type="rect",
        x0="2020-03-01",
        y0=0,
        x1="2020-08-01",
        y1=1,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.update_xaxes(dtick="M6", tickformat="%b\n%Y")
    fig.update_layout(
        title=f"Average Clustering Coefficient Over Time, {date_decoder[lag_number]} - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Average Clustering Coefficient",
    )
    if show:
        fig.show()
    return fig


def density(show: bool = False, gtype: str = "spearman_25"):
    date_decoder = {"25": "Feb 2005", "125": "Jul 2005"}
    lagnum = gtype.split("_")[1]
    lag_number = lagnum[:-1] if "/" in lagnum else lagnum
    df = load_data(gtype)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Density",
            mode="lines",
            x=df.index,
            y=df["Density"],
            xperiodalignment="end",
        )
    )
    fig.add_shape(  # Add rectangle for Great Recession
        type="rect",
        x0="2007-12-01",
        y0=0,
        x1="2009-06-01",
        y1=1.25,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for European Debt Crisis
        type="rect",
        x0="2010-04-01",
        y0=0,
        x1="2012-06-01",
        y1=1.25,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for COVID-19
        type="rect",
        x0="2020-03-01",
        y0=0,
        x1="2020-08-01",
        y1=1.25,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.update_xaxes(dtick="M6", tickformat="%b\n%Y")
    fig.update_layout(
        title=f"Graph Density Over Time, {date_decoder[lag_number]} - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Density",
    )
    if show:
        fig.show()
    return fig


def avg_weight(show: bool = False, gtype: str = "spearman_25"):
    date_decoder = {"25": "Feb 2005", "125": "Jul 2005"}
    lagnum = gtype.split("_")[1]
    lag_number = lagnum[:-1] if "/" in lagnum else lagnum
    df = load_data(gtype)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Average Edge Weight",
            mode="lines",
            x=df.index,
            y=df["Average Edge Weight"],
            xperiodalignment="end",
        )
    )
    fig.add_shape(  # Add rectangle for Great Recession
        type="rect",
        x0="2007-12-01",
        y0=0,
        x1="2009-06-01",
        y1=1,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for European Debt Crisis
        type="rect",
        x0="2010-04-01",
        y0=0,
        x1="2012-06-01",
        y1=1,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for COVID-19
        type="rect",
        x0="2020-03-01",
        y0=0,
        x1="2020-08-01",
        y1=1,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.update_xaxes(dtick="M6", tickformat="%b\n%Y")
    fig.update_layout(
        title=f"Average Edge Weight Over Time, {date_decoder[lag_number]} - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Average Edge Weight",
    )
    if show:
        fig.show()
    return fig


def delta_edges_over_time(show: bool = False):
    date_decoder = {"25": "Feb 2005", "125": "Jul 2005"}
    lagnum = gtype.split("_")[1]
    lag_number = lagnum[:-1] if "/" in lagnum else lagnum
    df = load_data()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="End-aligned",
            mode="lines",
            x=df.index,
            y=df["Change in Edge Number"],
            xperiodalignment="end",
        )
    )
    fig.add_shape(  # Add rectangle for Great Recession
        type="rect",
        x0="2007-12-01",
        y0=-7000,
        x1="2009-06-01",
        y1=7000,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for European Debt Crisis
        type="rect",
        x0="2010-04-01",
        y0=-7000,
        x1="2012-06-01",
        y1=7000,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for COVID-19
        type="rect",
        x0="2020-03-01",
        y0=-7000,
        x1="2020-08-01",
        y1=7000,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.update_xaxes(dtick="M6", tickformat="%b\n%Y")
    fig.update_layout(
        title=f"Change in Edge Number Over Time, {date_decoder[lag_number]} - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Change in Edge Number",
    )
    if show:
        fig.show()
    return fig


def main(gtype: str = "spearman_25"):
    gtype = gtype + "/" if "/" not in gtype else gtype
    path = gtype + "figures/"
    mst = "mst" in gtype.lower()

    # Create Figures

    pio.kaleido.scope.default_width = 1600
    pio.kaleido.scope.default_height = 1000
    plt.figure(figsize=(16, 16))
    heat = heatmap(show=False, gtype=gtype)
    curvatureplot = curvature(gtype=gtype)
    edgeweight = avg_weight(gtype=gtype)
    if not mst:
        edgeplot = edges_over_time(gtype=gtype)
        clustering = clustering_coef(gtype=gtype)
        densityplot = density(gtype=gtype)

    # Save Figures

    heat.figure.savefig(path + "heatmap.png", dpi=400)
    curvatureplot.write_image(path + "curvature.png")
    edgeweight.write_image(path + "edgeweight.png")

    if not mst:
        edgeplot.write_image(path + "edgenumber.png")
        clustering.write_image(path + "clustering.png")
        densityplot.write_image(path + "density.png")


if __name__ == "__main__":
    pio.templates.default = "plotly_white"
    main(gtype="spearman_125")
    main(gtype="spearman_25")
    main(gtype="spearman_25_mst")
    main(gtype="spearman_125_mst")
