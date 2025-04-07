import argparse
import subprocess
import itertools
import os
import csv

def run_ae(args):
    ae_command = [
        'python', 'AE_run.py',
        '--mode', str(args['mode']),
        '--seed', str(args['seed']),
        '--path1', args['path1'],
        '--path2', args['path2'],
        '--batchsize', str(args['batchsize']),
        '--learningrate', str(args['learningrate']),
        '--epoch', str(args['epoch']),
        '--latent', str(args['latent']),
        '--device', args['device'],
        '--a', str(args['a']),
        '--b', str(args['b']),
        '--topn', str(args['topn'])
    ]

    subprocess.run(ae_command)

def run_gcn(args):
    gcn_command = [
        'python', 'GCN_run.py',
        '--featuredata', args['featuredata'],
        '--phylogeneTreedata', args['phylogeneTreedata'],
        '--adjdata', args['adjdata'],
        '--labeldata', args['labeldata'],
        '--mode', str(args['gc_mode']),
        '--seed', str(args['gc_seed']),
        '--device', args['device'],
        '--epochs', str(args['gc_epochs']),
        '--learningrate', str(args['gc_learningrate']),
        '--weight_decay', str(args['weight_decay']),
        '--hidden', str(args['hidden']),
        '--dropout', str(args['dropout']),
        '--threshold', str(args['threshold']),
        '--nclass', str(args['nclass']),
        '--patience', str(args['patience'])
    ]

    if args['testsample'] is not None and args['testsample'] != '':
        gcn_command.extend(['--testsample', args['testsample']])

    result = subprocess.run(gcn_command, capture_output=True, text=True)

    lines = result.stdout.split('\n')
    cv_results = {}
    for line in lines:
        if "10-fold Cross Validation Results:" in line:
            break
    for line in lines:
        if "10-fold Cross Validation Results:" in line:
            continue
        if "Finished!" in line:
            break
        if ": Mean=" in line:
            key, value = line.split(": Mean=")
            mean, std = value.split(", Std=")
            cv_results[key.strip()] = (float(mean), float(std))

    return cv_results

def save_results(results, filename, header=False):
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['AE_seed', 'AE_batchsize', 'AE_learningrate', 'AE_epoch', 'AE_latent',
                      'GCN_seed', 'GCN_epochs', 'GCN_learningrate', 'GCN_weight_decay', 'GCN_dropout', 'GCN_hidden',
                      'Accuracy_mean', 'Accuracy_std', 'F1_Score_mean', 'F1_Score_std',
                      'AUC_mean', 'AUC_std', 'AUPR_mean', 'AUPR_std', 'Recall_mean', 'Recall_std', 'Precision_mean',
                      'Precision_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if header:
            writer.writeheader()
        writer.writerow(results)

def grid_search():
    ae_params = {
        'mode': [0],
        'seed': [101],
        'path1': ['./data/abundance_Cirrhosis.csv'],
        'path2': ['./data/marker_Cirrhosis.csv'],
        'batchsize': [32,64],
        'learningrate': [0.0008,0.001, 0.0005],
        'epoch': [500,600],
        'latent': [300,400],
        'device': ['gpu'],
        'a': [0.4],
        'b': [0.6],
        'topn': [100]
    }

    # 定义 GCN_run.py 的参数范围
    gcn_params = {
        'featuredata': ['./result/latent_Cirrhosis.csv'],
        'phylogeneTreedata': ['./data/phylogenTree_p_Cirrhosis.csv'],
        'adjdata': ['./Similarity/fused_Cirrhosis_matrix.csv'],
        'labeldata': ['./data/labels_Cirrhosis.csv'],
        'testsample': [''],
        'gc_mode': [0],
        'gc_seed': [1421],
        'device': ['gpu'],
        'gc_epochs': [500,800,1000],
        'gc_learningrate': [0.0001,0.001,0.00001],
        'weight_decay': [0.001,0.005],
        'hidden': [64,128],
        'dropout': [0.4],
        'threshold': [0.004],
        'nclass': [2],
        'patience': [20]
    }

    # 生成所有可能的参数组合
    ae_param_combinations = list(itertools.product(*ae_params.values()))
    gcn_param_combinations = list(itertools.product(*gcn_params.values()))

    best_score = -float('inf')
    best_ae_params = {}
    best_gcn_params = {}

    save_results({}, 'grid_search_results.csv', header=True)

    for ae_param in ae_param_combinations:
        ae_args = dict(zip(ae_params.keys(), ae_param))
        run_ae(ae_args)

        for gcn_param in gcn_param_combinations:
            gcn_args = dict(zip(gcn_params.keys(), gcn_param))
            cv_results = run_gcn(gcn_args)
            auc_mean = cv_results.get('AUC', (0, 0))[0]

            if auc_mean > best_score:
                best_score = auc_mean
                best_ae_params = ae_args
                best_gcn_params = gcn_args

            result = {
                'AE_seed': ae_args['seed'],
                'AE_batchsize': ae_args['batchsize'],
                'AE_learningrate': ae_args['learningrate'],
                'AE_epoch': ae_args['epoch'],
                'AE_latent': ae_args['latent'],
                'GCN_seed': gcn_args['gc_seed'],
                'GCN_epochs': gcn_args['gc_epochs'],
                'GCN_learningrate': gcn_args['gc_learningrate'],
                'GCN_weight_decay': gcn_args['weight_decay'],
                'GCN_dropout': gcn_args['dropout'],
                'GCN_hidden': gcn_args['hidden'],
                'Accuracy_mean': cv_results.get('Accuracy', (0, 0))[0],
                'Accuracy_std': cv_results.get('Accuracy', (0, 0))[1],
                'F1_Score_mean': cv_results.get('F1 Score', (0, 0))[0],
                'F1_Score_std': cv_results.get('F1 Score', (0, 0))[1],
                'AUC_mean': cv_results.get('AUC', (0, 0))[0],
                'AUC_std': cv_results.get('AUC', (0, 0))[1],
                'AUPR_mean': cv_results.get('AUPR', (0, 0))[0],
                'AUPR_std': cv_results.get('AUPR', (0, 0))[1],
                'Recall_mean': cv_results.get('Recall', (0, 0))[0],
                'Recall_std': cv_results.get('Recall', (0, 0))[1],
                'Precision_mean': cv_results.get('Precision', (0, 0))[0],
                'Precision_std': cv_results.get('Precision', (0, 0))[1]
            }
            save_results(result, 'grid_search_results.csv')

            print(f"AE params: {ae_args}")
            print(f"GCN params: {gcn_args}")
            print(f"10 CV results: {cv_results}")
            print("-" * 80)

    print(f"Best AUC: {best_score}")
    print(f"Best AE params: {best_ae_params}")
    print(f"Best GCN params: {best_gcn_params}")

if __name__ == '__main__':
    grid_search()