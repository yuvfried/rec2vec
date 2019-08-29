import itertools
import plotly.graph_objs as go
import chart_studio.plotly as py
import plotly.offline as offline

import numpy as np
import pandas as pd
import plotly.express as px
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
# import MulticoreTSNE as MTSNE
import cmake


def run_tsne(nemb):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, verbose=1)
    return tsne.fit_transform(nemb)


# constructing DBs
model = Word2Vec.load("ex1.model")
ing_names = ["carrot", "carrot", "carrot", "carrot", "honey", "sugar", "onion", "garlic", "potato"]
ing_vecs = [model.wv[ing] for ing in ing_names]  # import ingredients model vectors
ing_vecs = np.random.rand(100, 100)

colors = [1] * 100
# for i in range(9):
#     ing_vecs[i * 10] = [x * 2 for x in ing_vecs[i * 100]]
#     colors[i * 10] = colors[i * 100] = i

# ing_vecs = TSNE(n_components=2).fit_transform(ing_vecs)  # projecting the ingredients vectors on R^2
ing_vecs = run_tsne(ing_vecs)

# prepare the DB to be plotted
db = pd.DataFrame(ing_vecs)
db.columns = ["x", "y"]
# db["ing_names"] = ing_names
db["ing_names"] = [i for i in range(100)]

db["color"] = colors
print(db.color.to_string(index=False))
# plot data
fig = px.scatter(db, x="x", y="y", hover_name="ing_names", hover_data=["ing_names"], color="color")
fig.show()

# layout = go.Layout(
#     xaxis=dict(range=[0.75, 5.25], autorange=True),
#     yaxis=dict(range=[0, 8], autorange=True),
# )
# fig = go.Figure(data=db[["x", "y"]], layout=layout)
# plot_url = py.plot(fig, filename='text-hover')




# flatten = lambda l: [item for sublist in l for item in sublist]
#
# legend_order = [
# 'African',
# 'LatinAmerican',
# 'NorthAmerican',
# 'EastAsian',
# 'SouthAsian',
# 'SoutheastAsian',
# 'MiddleEastern',
# 'NorthernEuropean',
# 'EasternEuropean',
# 'WesternEuropean',
# 'SouthernEuropean',
# ]
#
# # These are the "Tableau 20" colors as RGB.
# tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# tableau20_rgb = ['rgb' + str(triplet) for triplet in tableau20]
#
# np.random.seed(1234)
# tableau20_sample = np.random.choice(tableau20_rgb, len(cuisines), replace=False)
# cuisine2color = {cuisine: tableau20_sample[i] for i, cuisine in enumerate(cuisines)}
#
# def make_plot(name, points, labels, legend_labels, legend_order, legend_label_to_color, pretty_legend_label, publish):
#     lst = zip(points, labels, legend_labels)
#     full = sorted(lst, key=lambda x: x[2])
#     traces = []
#     for legend_label, group in itertools.groupby(full, lambda x: x[2]):
#         group_points = []
#         group_labels = []
#         for tup in group:
#             point, label, _ = tup
#             group_points.append(point)
#             group_labels.append(label)
#         group_points = np.stack(group_points)
#         traces.append(go.Scattergl(
#             x=group_points[:, 0],
#             y=group_points[:, 1],
#             mode='markers',
#             marker=dict(
#                 color=legend_label_to_color[legend_label],
#                 size=8,
#                 opacity=0.6,
#                 # line = dict(width = 1)
#             ),
#             text=['{} ({})'.format(label, pretty_legend_label(legend_label)) for label in group_labels],
#             hoverinfo='text',
#             name=legend_label
#         )
#         )
#     # order the legend
#     ordered = [[trace for trace in traces if trace.name == lab] for lab in legend_order]
#     traces_ordered = flatten(ordered)
#
#     def _set_name(trace):
#         trace.name = pretty_legend_label(trace.name)
#         return trace
#
#     traces_ordered = list(map(_set_name, traces_ordered))
#     layout = go.Layout(
#         xaxis=dict(
#             autorange=True,
#             showgrid=False,
#             zeroline=False,
#             showline=False,
#             autotick=True,
#             ticks='',
#             showticklabels=False
#         ),
#         yaxis=dict(
#             autorange=True,
#             showgrid=False,
#             zeroline=False,
#             showline=False,
#             autotick=True,
#             ticks='',
#             showticklabels=False
#         )
#     )
#     fig = go.Figure(data=traces_ordered, layout=layout)
#     if publish:
#         plotter = py.iplot
#     else:
#         plotter = offline.plot
#     plotter(fig, filename=name + '.html')
#
#
# make_plot("demo", db[["x", "y"]],)
# make_plot("demo", db[["x", "y"]], db[["ing_names"]],db[["ing_names"]],db[["color"]],db[["ing_names"]])
