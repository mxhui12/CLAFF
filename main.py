from torchsummary import summary
import pickle
import numpy as np
import os
import pandas as pd
from few_shot.tasker import NodeClassify
from few_shot.utils import seed_everything,get_args
from few_shot.data import load_data,split_subgraphs

def load_subgraph(dataset_name, data, device):
    folder_path = './Experiment/subgraph/' + dataset_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = folder_path + '/subgraph.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            print('loading subgraph...')
            graphs_list = pickle.load(f)
            print('Done!!!')
    else:
        print('Begin split_subgraphs.')
        split_subgraphs(data, folder_path, device)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list


args = get_args()
seed_everything(args.seed)

print('dataset_name', args.dataset_name)

data, input_dim, output_dim = load_data(args.dataset_name)
data = data.to(args.device)
graphs_list = load_subgraph(args.dataset_name, data, args.device)

tasker = NodeClassify(pre_train_model_path=args.pre_train_model_path,
                      dataset_name=args.dataset_name, num_layer=args.num_layer, hid_dim=args.hid_dim,
                      epochs=args.epochs, shot_num=args.shot_num, device=args.device, lr=args.lr, wd=args.decay,
                      batch_size=args.batch_size, data=data, input_dim=input_dim, output_dim=output_dim,
                      graphs_list=graphs_list)

_, test_acc, std_test_acc = tasker.run()

print("Final Accuracy {:.4f}±{:.4f}(std)".format(test_acc, std_test_acc))



