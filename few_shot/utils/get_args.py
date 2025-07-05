import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of few-shot node classification')
    parser.add_argument('--dataset_name', type=str, default='Cora',help='Choose the dataset of downstream task')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--hid_dim', type=int, default=128,
                        help='hideen layer of GNN dimensions (default: 128)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--shot_num', type=int, default = 1, help='Number of shots')
    parser.add_argument('--way_num', type=int, default = 2, help='Number of ways')
    parser.add_argument('--train_num', type=int, default=2, help='Number of train_nums')
    parser.add_argument('--pre_train_model_path', type=str, default='None', 
                        help='add pre_train_model_path to the downstream task')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='Number of GNN message passing layers (default: 2).')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")


    args = parser.parse_args()
    return args
