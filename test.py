import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GCNConv, Node2Vec
from torch_geometric.datasets import KarateClub
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class DummyModel(nn.Module):

     def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(3, 1)

     def forward(self, x):
        return self.linear(x)


def print_hi():
    print("Torch version:", torch.__version__)
    print("PyG version:", torch_geometric.__version__)

    #x = torch.tensor([1, 2, 3])
    # print(x)

    # x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    # y = x * 2
    # z = y.sum()
    # z.backward()  # Computes gradients of z with respect to x
    # print(x.grad)
    # model = DummyModel()
    # print(model)

    # This is a set of graphs
    dataset = KarateClub()

    # We will use the first graph: let's call it "Joe"

    data = dataset[0]
    torch.manual_seed(12345)
    perm = torch.randperm(data.num_nodes)
    train_size = int(0.8 * data.num_nodes)

    data.train_mask = perm[:train_size]
    data.test_mask = perm[train_size:]



    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Number of features per node: {data.num_node_features}')
    print(f'Number of classes: {dataset.num_classes}')


    # what is a "walk"?
    # max number of hoops

    # Each input is a subgraph of "Joe": it generates 20 sized paths
    model = Node2Vec(data.edge_index, embedding_dim=64, walk_length=20, context_size=10, walks_per_node = 10, sparse = True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Each batch is a set of paths, there
    loader = model.loader(batch_size=128,shuffle=True,num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(
            train_z=z.index_select(0, data.train_mask),
            train_y=data.y.index_select(0, data.train_mask),
            test_z=z.index_select(0, data.test_mask),
            test_y=data.y.index_select(0, data.test_mask),
            max_iter=150,
        )
        return acc

    for epoch in range(100):
        loss = train()
        acc = test()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')


    embeddings = model()
    print("Shape of node embeddings:", embeddings.shape)


    @torch.no_grad()
    def plot_points(colors):
        model.eval()
        z = model().cpu().numpy()
        z = TSNE(n_components=2).fit_transform(z)
        y = data.y.cpu().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(dataset.num_classes):
            plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
        plt.axis('off')
        plt.show()

    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
    ]
    plot_points(colors)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
