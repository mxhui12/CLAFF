import torch.nn.functional as F
import torchmetrics
import torch
from tqdm import tqdm
def center_embedding(input, index, class_select):
    device = input.device
    c = torch.zeros(len(class_select), input.size(1), device=device)
    class_counts = torch.zeros(len(class_select), 1, device=device, dtype=input.dtype)
    real_labels = torch.tensor(class_select, device=device)
    for i, cls in enumerate(class_select):
        mask = index == cls
        if mask.sum() > 0:
            c[i] = input[mask].mean(dim=0)
            class_counts[i] = mask.sum()
        else:
            c[i] = torch.zeros(input.size(1), device=device)
            class_counts[i] = 0
    return c,real_labels, class_counts
def Evaluate(support_loader, query_loader, gnn, tuing, class_select, device):
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(class_select)).to(device)

    accumulated_centers = None
    accumulated_counts = None
    with torch.no_grad():
        for support_batch in support_loader:
            support_batch = support_batch.to(device)
            support_out = gnn(support_batch.x, support_batch.edge_index, support_batch.batch, tuing)
            center, real_labels,class_counts = center_embedding(support_out, support_batch.y, class_select)
            if accumulated_centers is None:
                accumulated_centers = center
                accumulated_counts = class_counts
            else:
                accumulated_centers += center * class_counts
                accumulated_counts += class_counts
        mean_centers = accumulated_centers / accumulated_counts
        for query_batch in query_loader:
            query_batch = query_batch.to(device)
            query_out = gnn(query_batch.x, query_batch.edge_index, query_batch.batch, tuing)
            similarity_matrix = F.cosine_similarity(query_out.unsqueeze(1), mean_centers.unsqueeze(0), dim=-1)
            pred = similarity_matrix.argmax(dim=1)
            pred_labels = real_labels[pred]
            accuracy.update(pred_labels, query_batch.y)

    return (
        accuracy.compute().item()
    )
