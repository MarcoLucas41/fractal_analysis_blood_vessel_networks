import torch_geometric.nn
import torch
from pytorch_dataset.link_dataset import LinkVesselGraph
from pytorch_dataset.node_dataset import NodeVesselGraph

from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import torch_geometric.transforms as T
import plotly.graph_objects as go
import networkx as nx

import numpy as np



from networkx.algorithms.community import greedy_modularity_communities

def generate_brain_plot(data):
    # Extract node data
    node_features_np = data.x.cpu().numpy()
    node_coords = node_features_np[:, :3]  # columns: xn, yn, zn
    node_region_labels = node_features_np[:, 3].astype(int)

    # Extract edge index
    edge_index_np = data.edge_index.cpu().numpy()  # shape (2, num_edges)

    # Optional: downsample nodes if too large
    MAX_NODES = 100_000
    if node_coords.shape[0] > MAX_NODES:
        sampled_indices = np.random.choice(node_coords.shape[0], MAX_NODES, replace=False)
        node_coords = node_coords[sampled_indices]
        node_region_labels = node_region_labels[sampled_indices]

        # Filter edge_index to only keep edges between sampled nodes
        node_id_map = {old: new for new, old in enumerate(sampled_indices)}
        mask = np.isin(edge_index_np[0], sampled_indices) & np.isin(edge_index_np[1], sampled_indices)
        edge_index_np = edge_index_np[:, mask]
        # remap to new indices
        edge_index_np = np.vectorize(node_id_map.get)(edge_index_np)

    # Create edge traces (as lines)
    edge_x, edge_y, edge_z = [], [], []
    for src, dst in edge_index_np.T:
        x0, y0, z0 = node_coords[src]
        x1, y1, z1 = node_coords[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(100,100,100,0.3)', width=1),
        hoverinfo='none',
        name='Edges'
    )

    # Create node trace
    node_trace = go.Scatter3d(
        x=node_coords[:, 0],
        y=node_coords[:, 1],
        z=node_coords[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=node_region_labels,
            colorscale='Viridis',
            colorbar=dict(title='Region'),
            opacity=0.8
        ),
        name='Nodes'
    )

    # Combine into a figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="3D Vessel Graph by Region",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

# for multi-class labeling

def graph_analysis(G):
    num_edges = G.number_of_edges()
    print("Number of edges:", num_edges)

    # For undirected graph: average degree = 2 * num_edges / num_nodes
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print("Average degree:", avg_degree)

    # degrees = [deg for node, deg in G.degree()]
    # plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), align='left', rwidth=0.8)
    # plt.xlabel("Degree")
    # plt.ylabel("Number of Nodes")
    # plt.title("Degree Distribution")
    # plt.grid(True)
    # plt.show()

    density = nx.density(G)
    print("Density of the network:", density)

    # ecc = nx.eccentricity(G)
    # print("Eccentricity of each node:", ecc)

    # diameter = nx.diameter(G)
    # print("Diameter of the network:", diameter)

    # radius = nx.radius(G)
    # print("Radius of the network:", radius)

    # bc = nx.betweenness_centrality(G)
    # top_10 = dict(sorted(bc.items(), key=operator.itemgetter(1), reverse=True)[:10])
    #
    # plt.bar(top_10.keys(), top_10.values())
    # plt.xlabel("Node")
    # plt.ylabel("Betweenness Centrality")
    # plt.title("Top 10 Nodes by Betweenness Centrality")
    # plt.show()

    communities = list(greedy_modularity_communities(G))

    # Print how many communities
    print("Number of communities found:", len(communities))

    # Plotting
    # color_map = {}
    # for i, com in enumerate(communities):
    #     for node in com:
    #         color_map[node] = i
    #
    # colors = [color_map[node] for node in G.nodes()]
    # nx.draw(G, node_color=colors, with_labels=True, cmap=plt.cm.tab10)
    # plt.title("Communities in the Network")
    # plt.show()

def get_link_vessel_graph(dataset,splitting_strategy):
    link_dataset = LinkVesselGraph(root='data',
                                   name=dataset,
                                   splitting_strategy=splitting_strategy,
                                   number_of_workers=2,
                                   val_ratio=0.1,
                                   test_ratio=0.1,
                                   seed=123,
                                   )
    data = link_dataset[0]
    # print(data)
    print(dataset)
    print('==============================================================')

    # Gather some statistics about the graph.
    # print(f'Number of nodes in graph: {data.num_nodes}')
    # print(f'Number of edges in graph: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    # print(f'Contains self-loops: {data.contains_self_loops()}')
    # print(f'Is Undirected: {data.is_undirected()}')
    #
    # print(f'Number of undirected edges', data.edge_index_undirected.size(dim=1))
    # print(f'Number of training edges', data.train_pos_edge_index.size(dim=1))
    # print(f'Number of validation edges', data.val_pos_edge_index.size(dim=1))
    # print(f'Number of test edges', data.test_pos_edge_index.size(dim=1))

    # Caution: if you would like to convert all edges to networkx graph, please
    # overwrite data.edge_index with data.edge_index_undirected.
    # The link dataset adheres to the convention that only training edges are
    # present in the data.edge_index. However, to obtain the full graph, we have to pass
    # all edges to the networkx function.

    data_undirected = Data(x=data.x, edge_index=data.edge_index_undirected,
                           edge_attr=data.edge_attr_undirected)
    return data_undirected


def get_node_vessel_graph(dataset):
    print('==============================================================')

    dataset = NodeVesselGraph(root='data', name=dataset, pre_transform=T.LineGraph(force_directed=False))
    # print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    # Gather some statistics about the graph.
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    # print(f'Contains self-loops: {data.has_self_loops()}')
    # print(f'Is directed: {data.is_directed()}')

    data= Data(x=data.x, edge_index=data.edge_index,
                           edge_attr=data.edge_attr)

    return data

# c = torch_geometric.nn.voxel_grid(data_unidirected)
# portanto, isto é suposto criar hipercubos 3d à volta do cérebro
# o stor escolheu o maior hipercubo que terá vários subgrafos dentro dele (porque as dimensões podem ser flexiveis)
# e depois ele separou por camadas e buscou o subgrafo com mais nós
# e para esse subgrafo, podemos calcular a dimensão fractal usando https://github.com/ChatzigeorgiouGroup/FractalDimension

# o stor usou uma versão do LinkVesselGraph alterada por ele


# indicate dimensions of 3D hypercubes
# retirar nós isolados
# separar por camadas
# buscar o componente com mais nós

def generate_training_graph(data: Data, voxel_dim: list = (100.0, 100.0, 100.0),
                            low_degree_threshold: float = 0.01):

    c = torch_geometric.nn.voxel_grid(data.x[:, 0:3], list(voxel_dim))

    # Count unique elements in c tensor
    unique, counts = torch.unique(c, return_counts=True)
    # Get the index of the most common element
    most_common_index = unique[counts.argmax()]

    clustered_data = data.clone()
    filtered_nodes = torch.argwhere(c == most_common_index).squeeze()
    clustered_data.edge_index = clustered_data.edge_index[:, np.all(np.isin(clustered_data.edge_index, filtered_nodes),
                                                                    axis=0)]
    # This removes isolated nodes
    r_isolated_nodes = torch_geometric.transforms.RemoveIsolatedNodes()
    r_isolated_nodes(clustered_data)

    # Get the degree of each node
    nodes_degree = torch.bincount(clustered_data.edge_index[0])
    # Calculate unique degree frequencies and return counts
    unique_degree_freq, counts = torch.unique(nodes_degree, return_counts=True)
    # Discard degree frequencies that occur less than 5% of the time
    degree_freq = unique_degree_freq[counts > low_degree_threshold * sum(counts)]

    # Filter out nodes whose unique degree frequency is less than 5% of the total number of nodes
    filtered_nodes = torch.argwhere(torch.isin(nodes_degree, degree_freq)).squeeze()
    # Filter out not in filtered nodes
    clustered_data.edge_index = clustered_data.edge_index[:, np.all(np.isin(clustered_data.edge_index, filtered_nodes),
                                                                    axis=0)]

    # Get the largest connected component
    largest_component = torch_geometric.transforms.LargestConnectedComponents()
    r_isolated_nodes(clustered_data)

    largest_component_data = largest_component(clustered_data)
    G = to_networkx(largest_component_data, to_undirected=False)

    return G, largest_component_data

def main():

    dataset = ['BALBc_no1','synthetic_graph_1']

    splitting_strategy = 'random'
    data = get_link_vessel_graph(dataset[0],splitting_strategy)
    #data = get_node_vessel_graph(dataset[0])

    # generate_brain_plot(data)
    print("Node feature shape:", data.x.shape)
    
    nx_graph, largest_component_data = generate_training_graph(data)
    
    #generate_brain_plot(data)


    print("Node feature shape:", data.x.shape)
    




    #G = to_networkx(data, to_undirected=False)
    #graph_analysis(G)




    # ATTEMPT TO OUTPUT A 3D GRAPH PLOT







if __name__ == "__main__":
    main()