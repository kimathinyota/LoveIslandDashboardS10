import networkx as nx
import nx_altair as nxa
import altair as alt
from altair import datum



def draw_graphs(x_vs_y_vs_weight, centrality_map, islanders):
    df = x_vs_y_vs_weight
    limited_graph = df.loc[(df.x.isin(islanders) & df.y.isin(islanders) & (df.weight > 0))]
    if len(limited_graph) == 0:
        return None
    G = nx.from_pandas_edgelist(limited_graph, 'x', 'y', ['weight'])
    pos = nx.spring_layout(G)
    # Add attributes to nodes
    for n in G.nodes():
        G.nodes[n]['name'] = n
        G.nodes[n]['centrality'] = "{:.2f}".format(centrality_map[n] * 100) + "%"
        G.nodes[n]['closest_neighbor'] = max(G[n], key=lambda value: G[n][value]['weight'])
    # Add attributes to edges
    for e in G.edges():
        G.edges[e[0], e[1]]['n'] = e[0] + " to " + e[1]
    chart = nxa.draw_networkx(
        G=G,
        pos=pos,
        width='weight',
        node_tooltip=['name', 'centrality', 'closest_neighbor'],
        edge_color='white',
        edge_tooltip=['n'],
        node_color='name',
        node_size=600
    )
    return chart


def standardise_edges_dataframe(x_vs_y_vs_weight_df):
    return x_vs_y_vs_weight_df.loc[x_vs_y_vs_weight_df.weight > 0].rename(columns={'x': 'name', 'y': 'partner'})



def create_interactive_influence_graph(filtered_edge_df_by_day, islander_to_centrality, picked_islanders):


    most_central_islander = max(picked_islanders, key=lambda islander: islander_to_centrality[islander])

    chart = draw_graphs(filtered_edge_df_by_day, islander_to_centrality, picked_islanders)

    if chart is None:
        return None
    chart_edges = chart.layer[0]
    chart_nodes = chart.layer[1]

    pick_name = alt.selection_point(fields=['name'],  value=[{'name': most_central_islander}], empty='none')
    chart_nodes = chart_nodes.encode(
        opacity = alt.condition(pick_name, alt.value(1.0), alt.value(0.3))
    ).add_params(
        pick_name
    )

    edges_dataframe_display = standardise_edges_dataframe(filtered_edge_df_by_day)

    donut = alt.Chart(edges_dataframe_display).transform_filter(
        pick_name
    ).mark_arc(innerRadius=100).encode(
        theta='weight',
        color=alt.Color('partner', sort=alt.EncodingSortField('weight', op='mean', order='descending')),
        tooltip=['partner', 'weight', 'name']
    ).interactive()


    donut_title = alt.Chart(edges_dataframe_display).transform_filter(
        pick_name
    ).mark_text(
        align='center', fontSize=26, color='white', opacity=0.2
    ).encode(
        text='name'
    )

    return alt.vconcat((chart_nodes + chart_edges).interactive(), donut + donut_title)
