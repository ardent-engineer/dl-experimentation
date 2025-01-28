# %% imports
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from sklearn.manifold import TSNE
import seaborn as sns

# %%
dataset =  Planetoid(root="data/Planetoid", name="PubMed", transform=NormalizeFeatures())
dataset[0].edge_index
data = dataset[0]
# %%
class GNNPubmed(nn.Module):
    def __init__(self, num_hidden, num_features, num_classes, heads = 8):
        super().__init__()
        self.conv1 = GATConv(num_features, num_hidden,heads)
        self.conv2 = GATConv(heads*num_hidden, num_classes,heads)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.3)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)
        x = self.conv2(x, edge_index)
        return x

# %%
model = GNNPubmed(25, dataset.num_features, dataset.num_classes, heads=8)
optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
loss_fn = nn.CrossEntropyLoss()
losses = []
NUM_EPOCHS = 1000
# %%
for epoch in range (NUM_EPOCHS):
    optimizer.zero_grad()
    y_hat = model(data.x, data.edge_index)
    loss = loss_fn(y_hat[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(losses[-1])
sns.lineplot(x = range(len(losses)), y=losses)
# %%
model.eval()
with torch.no_grad():
    y_pred = model(data.x, data.edge_index)
    y_pred_cls = y_pred.argmax(dim=1)
    correct_pred = y_pred_cls[data.test_mask] == data.y[data.test_mask]
    test_acc = int(correct_pred.sum()) / int(data.test_mask.sum())
print(f'Test Accuracy: {test_acc}')
# %%
