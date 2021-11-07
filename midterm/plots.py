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


def load_data():

    df = pd.read_csv("graph_features.csv")
    df["delta_k"] = df["kfix"].diff()
    df["delta_edges"] = df["number_of_edges"].diff()
    df["delta_density"] = df["density"].diff()
    df["delta_avg_clustering"] = df["avg_clustering"].diff()
    df.set_index("time", inplace=True)

    sp = yfinance.download(
        tickers="SPY", period="max", interval="1d", auto_adjust=True
    )["Close"]
    sp = np.log10(sp).diff()
    sp = sp.loc["2005-02-09":"2021-10-20"]

    df["SP500"] = list(sp)

    df = df[
        [
            "kfix",
            "density",
            "avg_clustering",
            "number_of_edges",
            "delta_k",
            "delta_edges",
            "delta_density",
            "delta_avg_clustering",
            "SP500",
        ]
    ]

    df = df.rename(
        columns={
            "kfix": "Curvature",
            "density": "Density",
            "avg_clustering": "Average Clustering Coef.",
            "number_of_edges": "Number of Edges",
            "delta_k": "Change in Curvature",
            "delta_edges": "Change in Edge Number",
            "delta_density": "Change in Density",
            "delta_avg_clustering": "Change in Avg. Clustering",
            "SP500": "S&P500",
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


def heatmap(show: bool = False):
    pd.set_option("display.max_columns", 10)
    df = load_data()
    print(df.columns)
    corr = df.corr().astype(float)
    print(corr)

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


def edges_over_time(show: bool = False):
    df = load_data()
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
        title="Edge Number Over Time, Jan 2005 - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Edge Number",
    )
    if show:
        fig.show()
    return fig


def curvature(show: bool = False):
    df = load_data()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Curvature",
            mode="lines",
            x=df.index,
            y=df["Curvature"],
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
        title="Average Curvature Over Time, Jan 2005 - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Average Curvature",
    )
    if show:
        fig.show()
    return fig


def clustering_coef(show: bool = False):
    df = load_data()
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
        title="Average Clustering Coefficient Over Time, Jan 2005 - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Average Clustering Coefficient",
    )
    if show:
        fig.show()
    return fig


def density(show: bool = False):
    df = load_data()
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
        y0=-2,
        x1="2009-06-01",
        y1=2,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for European Debt Crisis
        type="rect",
        x0="2010-04-01",
        y0=-2,
        x1="2012-06-01",
        y1=2,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.add_shape(  # Add rectangle for COVID-19
        type="rect",
        x0="2020-03-01",
        y0=-2,
        x1="2020-08-01",
        y1=2,
        line=dict(color="LightGray", width=0.5),
        fillcolor="LightGray",
        opacity=0.4,
    )
    fig.update_xaxes(dtick="M6", tickformat="%b\n%Y")
    fig.update_layout(
        title="Graph Density Over Time, Jan 2005 - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Density",
    )
    if show:
        fig.show()
    return fig


def delta_edges_over_time(show: bool = False):
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
        title="Change in Edge Number Over Time, Jan 2005 - Oct 2021",
        xaxis_title="Date",
        yaxis_title="Change in Edge Number",
    )
    if show:
        fig.show()
    return fig


def main():
    pio.kaleido.scope.default_width = 1600
    pio.kaleido.scope.default_height = 1000
    plt.figure(figsize=(16, 16))
    heat = heatmap()
    edgeplot = edges_over_time()
    edgeplot_delta = delta_edges_over_time()
    curvatureplot = curvature()
    clustering = clustering_coef()
    densityplot = density()
    # df = load_data()
    # print(df.columns)

    # heat.figure.savefig("figures/heatmap_spearman_25.png", dpi=400)
    edgeplot.write_image("figures/edges.png")
    edgeplot_delta.write_image("figures/edges_delta.png")
    curvatureplot.write_image("figures/curvature.png")
    clustering.write_image("figures/clustering.png")
    densityplot.write_image("figures/densityplot.png")


if __name__ == "__main__":
    pio.templates.default = "plotly_white"
    main()
