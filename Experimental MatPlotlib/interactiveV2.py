import plotly.express as px
import ipywidgets as widgets

df = px.data.tips()


def plot_graph(density):
    fig = px.histogram(df, x="total_bill", nbins=20, histnorm='density' if density else None)
    fig.show()


checkbox = widgets.Checkbox(
    value=False,
    description='Show Density',
    disabled=False
)

widgets.interactive(plot_graph, density=checkbox)