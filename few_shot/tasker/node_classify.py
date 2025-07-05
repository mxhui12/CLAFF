import torch
from torch_geometric.loader import DataLoader
from few_shot.utils import Evaluate
from few_shot.model import GCN
from torch import nn, optim
import time
import warnings
import numpy as np
from few_shot.data import random_split_classes, node_sample_and_save,test_node_sample_and_save,GraphDataset
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch.nn as nn


def center_embedding(input, index, myclass):
    device=input.device
    c = torch.zeros(len(myclass), input.size(1)).to(device)
    class_counts = torch.zeros(len(myclass), 1, device=device, dtype=input.dtype)
    for i, cls in enumerate(myclass):
        mask = index == cls
        if mask.sum() > 0:
            c[i] = input[mask].mean(dim=0)
            class_counts[i] = mask.sum()
        else:
            c[i] = torch.zeros(input.size(1), device=device)
            class_counts[i] = 0

    return c, class_counts
class ec_loss(nn.Module):
      def __init__(self, tau=0.1):
            super(ec_loss, self).__init__()
            self.tau = tau

      def forward(self, embedding, center_embedding, labels, class_select):
            device = embedding.device
            label_to_idx = {cls.item(): idx for idx, cls in enumerate(class_select)}
            mapped_labels = torch.tensor([label_to_idx[label.item()] for label in labels], device=device)
            similarity_matrix = F.cosine_similarity(embedding.unsqueeze(1), center_embedding.unsqueeze(0),
                                                    dim=-1) / self.tau
            exp_similarities = torch.exp(similarity_matrix)
            pos_neg = torch.sum(exp_similarities, dim=1, keepdim=True)
            pos = exp_similarities.gather(1, mapped_labels.view(-1, 1))
            L_tune = -torch.log(pos / pos_neg)
            loss = torch.sum(L_tune)
            return loss

class featureTuning(torch.nn.Module):
    def __init__(self,input_dim):
        super(featureTuning, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, node_embeddings):
        node_embeddings=node_embeddings*self.weight
        return node_embeddings
class NodeClassify:
      def __init__(self, data, input_dim, output_dim, graphs_list = None, pre_train_model_path='None',
                 hid_dim=128, num_layer=2, dataset_name='Cora',
                 epochs=100, shot_num=10, way_num=1, train_num=2, device: int = 5,
                 lr=0.001, wd=5e-4, batch_size=16, search=False):
            self.data = data
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.graphs_list = graphs_list
            self.pre_train_model_path = pre_train_model_path
            self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
            self.hid_dim = hid_dim
            self.num_layer = num_layer
            self.dataset_name = dataset_name
            self.shot_num = shot_num
            self.way_num = way_num
            self.train_num = train_num
            self.epochs = epochs
            self.lr = lr
            self.wd = wd
            self.batch_size = batch_size
            self.initialize_lossfn()


      def initialize_lossfn(self):
            self.criterion = ec_loss()

      def initialize_optimizer(self):
            self.pg_opi = optim.Adam(self.tuing.parameters(), lr=self.lr, weight_decay=self.wd)

      def initialize_tuing(self):
            self.tuing = featureTuning(self.hid_dim).to(self.device)

      def initialize_gnn(self):
            self.gnn = GCN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
            self.gnn.to(self.device)
            print(self.gnn)

            if self.pre_train_model_path != 'None':
                  self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location='cpu'))
                  self.gnn.to(self.device)
                  print("Successfully loaded pre-trained weights!")





      def train(self, train_loader,train_query_loader,myclass):
            self.tuing.train()
            total_loss = 0.0
            accumulated_centers = None
            accumulated_counts = None

            for batch in train_loader:
                  batch = batch.to(self.device)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, tuing=self.tuing)

                  center, class_counts = center_embedding(out, batch.y,myclass)
                  if accumulated_centers is None:
                        accumulated_centers = center
                        accumulated_counts = class_counts
                  else:
                        accumulated_centers += center * class_counts
                        accumulated_counts += class_counts

            mean_centers = (accumulated_centers / accumulated_counts).detach()

            train_query_total_loss = 0.0
            for query_batch in train_query_loader:
                  self.pg_opi.zero_grad()
                  query_batch = query_batch.to(self.device)

                  query_out = self.gnn(query_batch.x, query_batch.edge_index, query_batch.batch, tuing=self.tuing)

                  query_criterion = ec_loss()
                  query_loss = query_criterion(query_out, mean_centers, query_batch.y,myclass)

                  query_loss.backward()
                  self.pg_opi.step()
                  train_query_total_loss += query_loss.item()

            return train_query_total_loss / len(train_query_loader), mean_centers

      def run(self):
            test_accs = []
            f1s = []
            rocs = []
            prcs = []
            batch_best_loss = []

            for i in range(1, 6):
                  self.initialize_gnn()
                  self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                                torch.nn.Softmax(dim=1)).to(self.device) 
                  self.initialize_tuing()
                  self.initialize_optimizer()
                  train_class, test_class = random_split_classes(data=self.data, train_num=self.train_num)
                  node_embedding = self.gnn(self.data.x, self.data.edge_index)
                  patience = 20
                  best = 1e9
                  cnt_wait = 0
                  best_loss = 1e9
                  for epoch in range(1, self.epochs):
                        t0 = time.time()

                        train_idx, train_query_idx, train_lbls, train_query_lbls, train_graphs, train_query_graphs,myclass = node_sample_and_save(
                              data=self.data,
                              k=self.shot_num,
                              n=self.way_num,
                              graphs_list=self.graphs_list,
                              train_class=train_class,
                              device=self.device
                        )
                        train_dataset = GraphDataset(train_graphs)
                        train_query_dataset = GraphDataset(train_query_graphs)

                        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                        train_query_loader = DataLoader(train_query_dataset, batch_size=self.batch_size, shuffle=False)
                        loss, center = self.train(train_loader,train_query_loader,myclass)
                        if loss < best:
                              best = loss
                              cnt_wait = 0
                              torch.save(self.gnn.state_dict(),
                                         "./Experiment/bestgnn/{}/{}.pth".format(self.dataset_name,
                                                                                    str(self.hid_dim) + 'hidden_dim'))
                        else:
                              cnt_wait += 1
                              if cnt_wait == patience:
                                    print('-' * 100)
                                    print('Early stopping at '+str(epoch) +' eopch!')
                                    break
                        print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))
                  import math
                  if not math.isnan(loss):
                        batch_best_loss.append(loss)
                        test_num = 100
                        print("Loading the best model for evaluation...")
                        self.gnn.load_state_dict(torch.load(
                              "./Experiment/bestgnn/{}/{}.pth".format(self.dataset_name,
                                                                         str(self.hid_dim) + 'hidden_dim')))  # 加载模型权重
                        self.gnn.eval()
                        acc_list, f1_list, auroc_list, auprc_list = [], [], [], []
                        for j in range(test_num):
                              if j % 10 == 0 and j != 0:
                                    print(f"Running test {i + 1}/{test_num}...")
                              support_idx, support_lbls, query_idx, query_lbls, support_graphs, query_graphs,classes = test_node_sample_and_save(
                                    data=self.data,
                                    k=self.shot_num,
                                    n=self.way_num,
                                    graphs_list=self.graphs_list,
                                    test_class=test_class,
                                    device=self.device
                              )
                              support_dataset = GraphDataset(support_graphs)
                              support_loader = DataLoader(support_dataset, batch_size=self.batch_size,
                                                          shuffle=False)
                              query_dataset = GraphDataset(query_graphs)
                              query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False)
                              acc = Evaluate(
                                    support_loader=support_loader,
                                    query_loader=query_loader,
                                    gnn=self.gnn,
                                    tuing=self.tuing,
                                    class_select=classes,
                                    device=self.device
                              )
                              acc_list.append(acc)
                        test_acc = sum(acc_list) / test_num
                        print("Final Results after Multiple Tests:")
                        print(f"Average Accuracy: {test_acc:.4f}")
                  print(f"Epoch {epoch}: Test Acc: {test_acc:.4f}")
                  print("best_loss",  batch_best_loss)
                  test_accs.append(test_acc)
                  print(f"Task {i}: Accuracy {test_acc:.4f}")
            mean_test_acc = np.mean(test_accs)
            std_test_acc = np.std(test_accs)
            print(" Final best | test Accuracy {:.4f}±{:.4f}(std)".format(mean_test_acc, std_test_acc))
            mean_best = np.mean(batch_best_loss)
            return  mean_best, mean_test_acc, std_test_acc





                  


