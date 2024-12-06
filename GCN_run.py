import numpy as np
import pandas as pd
import argparse
import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, auc, recall_score, precision_recall_curve, roc_curve,accuracy_score
import torch
import torch.nn.functional as F
from gcn_model import GCN
from gcn_model import multiGATModelAE
from util import load_data
from util import accuracy
from torch_geometric.utils import dense_to_sparse
import torch.nn as nn
import re

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def train(epoch, optimizer, features, adj, labels, idx_train):
    '''
    :param features: the  features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_train: the index of trained samples
    '''
    labels.to(device)

    GCN_model.train()
    optimizer.zero_grad()

    edge_index, _ = dense_to_sparse(adj)
    edge_index = edge_index.to(torch.long)
    data = {'x': features, 'edge_index': edge_index}
    output = GCN_model(data)

    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    softmax_output = F.softmax(output[idx_train], dim=1)
    try:
        auc_train = roc_auc_score(labels[idx_train].cpu(), softmax_output[:, 1].detach().cpu())
    except ValueError:
        auc_train = float('nan')
    loss_train.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(
            f'Epoch: {epoch + 1:.2f} | Loss Train: {loss_train.item():.4f} | Acc Train:  {acc_train:.4f} | AUC Train: {auc_train:.4f}')
    return loss_train.data.item()


def test(features, adj, labels, idx_test):
    '''
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_test: the index of tested samples
    '''
    GCN_model.eval()
    # output = GCN_model(features, adj)

    edge_index, _ = dense_to_sparse(adj)
    edge_index = edge_index.to(torch.long)

    data = {'x': features, 'edge_index': edge_index}
    output = GCN_model(data)
    ##### GAT
    # output, _ = GCN_model(features, adj)  # 假设 features 是输入特征

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    #output is the one-hot label
    ot = output[idx_test].detach().cpu().numpy()
    #change one-hot label to digit label
    ot = np.argmax(ot, axis=1)
    #original label
    lb = labels[idx_test].detach().cpu().numpy()
    print('predict label: ', ot)
    print('original label: ', lb)

    #calculate the f1 score
    f1 = f1_score(lb, ot, average='weighted')
    ##################################
    # predicted_labels = output[idx_test].detach().cpu().numpy()
    # predicted_labels = np.argmax(predicted_labels, axis=1)
    # fpr, tpr, thresholds = roc_curve(labels[idx_test].cpu(), predicted_labels)
    #
    # with open("TPR_FPR.txt", 'w') as f:
    #     f.write("TPR: " + str(tpr.tolist()) + "\n")
    #     f.write("FPR: " + str(fpr.tolist()) + "\n")

    # Calculate AUC
    # Ensure that 'ot' contains probability scores
    auc_score = roc_auc_score(lb, ot, multi_class='ovr')  # Adjust parameters as needed

    # Calculate AUPR
    precision, recall, _ = precision_recall_curve(lb, ot)
    aupr = auc(recall, precision)

    # Calculate Recall and Precision
    rec = recall_score(lb, ot, average='weighted')
    prec = precision_score(lb, ot, average='weighted')

    # Print all metrics
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "f1_score= {:.4f}".format(f1),
          "AUC= {:.4f}".format(auc_score),
          "AUPR= {:.4f}".format(aupr),
          "Recall= {:.4f}".format(rec),
          "Precision= {:.4f}".format(prec))
    # Return all metrics
    return acc_test.item(), f1, auc_score, aupr, rec, prec



def predict(features, adj, sample, idx):
    GCN_model.eval()
    edge_index, _ = dense_to_sparse(adj)
    edge_index = edge_index.to(torch.long)
    data = {'x': features, 'edge_index': edge_index}
    output = GCN_model(data)

    # output = GCN_model(features, adj)
    # output, _ = GCN_model(features, adj)

    predict_label = output.detach().cpu().numpy()
    predict_label = np.argmax(predict_label, axis=1).tolist()
    res_data = pd.DataFrame({'Sample':sample, 'predict_label':predict_label})
    res_data = res_data.iloc[idx,:]
    #print(res_data)
    res_data.to_csv('result/GCN_predicted_data.csv', header=True, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--featuredata', '-fd', type=str, default="./result/latent_Cirrhosis.csv", help='The vector feature file.')
    parser.add_argument('--phylogeneTreedata', '-ph', type=str, default="./data/phylogenTree_p_Cirrhosis.csv", help='The vector phylogenetic Tree file.')
    parser.add_argument('--adjdata', '-ad', type=str, default="./Similarity/fused_Cirrhosis_matrix.csv", help='The adjacency matrix file.')
    parser.add_argument('--labeldata', '-ld', type=str, default="./data/labels_Cirrhosis.csv", help='The sample label file.')
    parser.add_argument('--testsample', '-ts', type=str, help='Test sample names file.',default="./data/test_sample.csv")
    parser.add_argument('--mode', '-m', type=int, choices=[0,1], default=0,
                        help='mode 0: 10-fold cross validation; mode 1: train and test a model.')
    parser.add_argument('--seed', '-s', type=int, default=1421, help='Random seed')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Training on cpu or gpu, default: gpu.')
    parser.add_argument('--epochs', '-e', type=int, default=500, help='Training epochs, default: 500.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, help='Learning rate, default: 0.001.')
    parser.add_argument('--weight_decay', '-w', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters), methods to avoid overfitting, default: 0.01')
    parser.add_argument('--hidden', '-hd',type=int, default=64, help='Hidden layer dimension, default: 64.')
    parser.add_argument('--dropout', '-dp', type=float, default=0.4, help='Dropout rate, methods to avoid overfitting, default: 0.5.')
    parser.add_argument('--threshold', '-t', type=float, default=0.004, help='Threshold to filter edges, default: 0.005') # 注意
    parser.add_argument('--nclass', '-nc', type=int, default=2, help='Number of classes, default: 2')
    parser.add_argument('--patience', '-p', type=int, default=20, help='Patience')
    args = parser.parse_args()

    # Check whether GPUs are available
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seed
    setup_seed(args.seed)

    # adj, data, label = load_data(args.adjdata, args.labeldata, args.threshold)
    # adj, data, label = load_data(args.adjdata, args.featuredata, args.labeldata, args.threshold)
    adj, data, label = load_data(args.adjdata, args.featuredata, args.phylogeneTreedata, args.labeldata, args.threshold)

    # change dataframe to Tensor
    adj = torch.tensor(adj, dtype=torch.float, device=device)
    features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float, device=device)
    labels = torch.tensor(label.iloc[:, 1].values, dtype=torch.long, device=device)

    print('Begin training model...')

    # 10-fold cross validation
    if args.mode == 0:
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        # acc_res, f1_res = [], []  #record accuracy and f1 score
        # Initialize lists to record all metrics
        acc_res, f1_res, auc_res, aupr_res, recall_res, precision_res = [], [], [], [], [], []
        with open('Cross_Validation_results.txt', 'w') as f:
            fold = 0
            f.write("Fold\tAccuracy\tF1 Score\tAUC\tAUPR\tRecall\tPrecision\n")  # Write headers
            # split train and test data
            for idx_train, idx_test in skf.split(data.iloc[:, 1:], label.iloc[:, 1]):
                ## uses GCN
                # GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout)

                ## uses DeepGCN
                GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, n_blocks=4, dropout=args.dropout)

                ## uses GAT
                # nfeat = features.shape[1]
                # nhid = args.hidden
                # nclass = args.nclass
                # dropout = args.dropout
                # alpha = 0.2
                # nheads = 4
                # npatient = data.shape[0]
                # GCN_model = multiGATModelAE(nfeat, nhid, nclass, dropout, alpha, nheads, npatient)

                GCN_model.to(device)

                # define the optimizer
                optimizer = torch.optim.Adam(GCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)

                idx_train, idx_test= torch.tensor(idx_train, dtype=torch.long, device=device), torch.tensor(idx_test, dtype=torch.long, device=device)
                for epoch in range(args.epochs):
                    train(epoch, optimizer, features, adj, labels, idx_train)

                # calculate the accuracy and f1 score
                ac, f1, auc_score, aupr, recall, precision = test(features, adj, labels, idx_test)
                acc_res.append(ac)
                f1_res.append(f1)
                auc_res.append(auc_score)
                aupr_res.append(aupr)
                recall_res.append(recall)
                precision_res.append(precision)
                fold=fold+1
                f.write(f"{fold}\t{ac:.4f}\t{f1:.4f}\t{auc_score:.4f}\t{aupr:.4f}\t{recall:.4f}\t{precision:.4f}\n")
            # Print average and standard deviation for each metric
        print('10-fold Cross Validation Results:')
        print('Accuracy: Mean=%.4f, Std=%.4f' % (np.mean(acc_res), np.std(acc_res)))
        print('F1 Score: Mean=%.4f, Std=%.4f' % (np.mean(f1_res), np.std(f1_res)))
        print('AUC: Mean=%.4f, Std=%.4f' % (np.mean(auc_res), np.std(auc_res)))
        print('AUPR: Mean=%.4f, Std=%.4f' % (np.mean(aupr_res), np.std(aupr_res)))
        print('Recall: Mean=%.4f, Std=%.4f' % (np.mean(recall_res), np.std(recall_res)))
        print('Precision: Mean=%.4f, Std=%.4f' % (np.mean(precision_res), np.std(precision_res)))

        match = re.search(r'latent_(\w+)\.csv$', args.featuredata)
        if match:
            disease_name = match.group(1)
        else:
            disease_name = "Noname"

        # Create a dictionary to save parameters and results
        results = {
            'mode': [args.mode],
            'seed': [args.seed],
            'featuredata': [args.featuredata],
            'phylogeneTreedata': [args.phylogeneTreedata],
            'adjdata': [args.adjdata],
            'labeldata': [args.labeldata],
            'testsample': [args.testsample],
            'device': [args.device],
            'epochs': [args.epochs],
            'learningrate': [args.learningrate],
            'weight_decay': [args.weight_decay],
            'hidden': [args.hidden],
            'dropout': [args.dropout],
            'threshold': [args.threshold],
            'nclass': [args.nclass],
            'patience': [args.patience],
            'accuracy_mean': [np.mean(acc_res)],
            'accuracy_std': [np.std(acc_res)],
            'f1_score_mean': [np.mean(f1_res)],
            'f1_score_std': [np.std(f1_res)],
            'auc_mean': [np.mean(auc_res)],
            'auc_std': [np.std(auc_res)],
            'aupr_mean': [np.mean(aupr_res)],
            'aupr_std': [np.std(aupr_res)],
            'recall_mean': [np.mean(recall_res)],
            'recall_std': [np.std(recall_res)],
            'precision_mean': [np.mean(precision_res)],
            'precision_std': [np.std(precision_res)]
        }

        # Convert the results dictionary to a DataFrame
        results_df = pd.DataFrame(results)

        # Determine the mode for saving the file
        file_path = f'model/{disease_name}_GCN_results.csv'
        mode = 'a' if os.path.exists(file_path) else 'w'
        header = not os.path.exists(file_path)

        # Save results to CSV file with disease name in the filename
        results_df.to_csv(file_path, mode=mode, header=header, index=False)


    elif args.mode == 1:
        # load test samples
        test_sample_df = pd.read_csv(args.testsample, header=0, index_col=None)
        test_sample = test_sample_df.iloc[:, 0].tolist()
        all_sample = data['Sample'].tolist()
        train_sample = list(set(all_sample) - set(test_sample))
        # get index of train samples and test samples
        train_idx = data[data['Sample'].isin(train_sample)].index.tolist()
        test_idx = data[data['Sample'].isin(test_sample)].index.tolist()
        # GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout)

        GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, n_blocks=4, dropout=args.dropout)
        GCN_model.to(device)
        optimizer = torch.optim.Adam(GCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)
        idx_train, idx_test = torch.tensor(train_idx, dtype=torch.long, device=device), torch.tensor(test_idx,
                                                                                                     dtype=torch.long,
                                                                                                     device=device)
        '''
        save a best model (with the minimum loss value)
        if the loss didn't decrease in N epochs，stop the train process.
        N can be set by args.patience 
        '''
        loss_values = []  # record the loss value of each epoch
        # record the times with no loss decrease, record the best epoch
        bad_counter, best_epoch = 0, 0
        best = 1000  # record the lowest loss value
        for epoch in range(args.epochs):
            loss_values.append(train(epoch, optimizer, features, adj, labels, idx_train))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1  # In this epoch, the loss value didn't decrease
            if bad_counter == args.patience:
                break
            # save model of this epoch
            torch.save(GCN_model.state_dict(), 'model/GCN/{}.pkl'.format(epoch))
            # reserve the best model, delete other models
            files = glob.glob('model/GCN/*.pkl')
            for file in files:
                name = file.split('\\')[1]
                epoch_nb = int(name.split('.')[0])
                # print(file, name, epoch_nb)
                if epoch_nb != best_epoch:
                    os.remove(file)
        print('Training finished.')
        print('The best epoch model is ', best_epoch)
        GCN_model.load_state_dict(torch.load('model/GCN/{}.pkl'.format(best_epoch)))
        predict(features, adj, all_sample, test_idx)

    print('Finished!')