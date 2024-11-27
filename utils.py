import numpy as np
import torch
import numpy as np 
import pdb
import os
from datetime import datetime
import sys

#删除目录下扩展名为.o,.exe,.bak的文件
def delete_file(save_model_path):
    targetDir = save_model_path
    for file in os.listdir(targetDir):
        targetFile = os.path.join(targetDir, file)
        # if suffix(file, '.avi', '.bak', '.exe'): .ckpt.data-00000-of-00001
        # if suffix(file, '.ckpt.data-00000-of-00001', '.ckpt.index', '.ckpt.meta'):
        if suffix(file, '.pt'):
            os.remove(targetFile)


#获取文件后缀名
def suffix(file, *suffixName):
    array = map(file.endswith, suffixName)
    if True in array:
        return True
    else:
        return False


def log_print(log_path, string):
    ''' print the output results and save them into log
    '''
    with open(log_path, 'a') as f:
        print(string, file = f)
    print(string)


def set_path(argv):#将结果保存到log.txt文件中
    path_front= f'/home/wsco/wy/HFFT/result_{argv.dataset}'
    results_path = os.path.join(path_front,argv.dataset +'_'+argv.label_type+'_results_' + str(datetime.now().strftime('%Y%m%d-%H%M%S')))
    # make dirs of results
    if not os.path.isdir(results_path):os.makedirs(results_path)
    # set the path of log and save the log on terminal
    results_log_path = results_path + '/log.txt'
    log_print(results_log_path,'folder: {} is created!'.format(results_path))
    return results_path, results_log_path

def self_makedirs(path, fold_index):

    fold_path = os.path.join(path + '/fold'+str(fold_index)) # path/foldX
    # Create parent path if it doesn't exist
    if not os.path.isdir(fold_path): os.makedirs(fold_path)
    fold_log = fold_path + '/fold' + str(fold_index)+'_log.txt' # path/foldX/foldX_log.txt
    return fold_path, fold_log


def parse_features(args):
    data_path = args.root_dir + '/' + args.feat_type+'.npz'
    data = np.load(data_path,allow_pickle=True)
    Data=data['data']
    if args.dataset=='deap':
        sub_label, sub_resp, sub_arousal_label, sub_valence_label,sub_gender_label=[],[],[],[],[]
        all_id,all_resp,all_arousal_label,all_valence_label,all_gender_label=[],[],[],[],[]
        for sub_num in range(1,33):
            for i in range(1280):
                if sub_num==int(Data[i]['exp_id']):
                    sub_label.append(sub_num)
                    sub_resp.append(Data[i]['resp'])
                    sub_arousal_label.append(Data[i]['arousal_label'])
                    sub_valence_label.append(Data[i]['valence_label'])
                    sub_gender_label.append(Data[i]['gender_label'])
        all_id.append(sub_label)
        all_resp.append(sub_resp)
        all_arousal_label.append(sub_arousal_label)
        all_valence_label.append(sub_valence_label)
        all_gender_label.append(sub_gender_label)
    elif args.dataset=='mahnob':
        sub_label, sub_resp, sub_arousal_label, sub_valence_label,sub_gender_label=[],[],[],[],[]
        all_id,all_resp,all_arousal_label,all_valence_label,all_gender_label=[],[],[],[],[]
        for sub_num in range(1,28):
            for i in range(527):
                if sub_num==int(Data[i]['exp_id']):
                    sub_label.append(sub_num)
                    sub_resp.append(Data[i]['resp'])
                    sub_arousal_label.append(Data[i]['arousal_label'])
                    sub_valence_label.append(Data[i]['valence_label'])
                    sub_gender_label.append(Data[i]['gender_label'])
        all_id.append(sub_label)
        all_resp.append(sub_resp)
        all_arousal_label.append(sub_arousal_label)
        all_valence_label.append(sub_valence_label)
        all_gender_label.append(sub_gender_label)

    return np.array(all_id[0]),np.array(all_resp[0]),np.array(all_arousal_label[0]),np.array(all_valence_label[0]),np.array(all_gender_label[0])

def load_training(tra_feats, tra_lab,tra_lab_gender, batch_size, kwargs):
    tra_data = torch.tensor(tra_feats, dtype=torch.float32)
    tra_lab = torch.tensor(tra_lab, dtype=torch.long)
    tra_lab_gender =torch.tensor(tra_lab_gender, dtype=torch.long)
    tra_dataset = torch.utils.data.TensorDataset(tra_data,tra_lab,tra_lab_gender)
    train_loader = torch.utils.data.DataLoader(tra_dataset,batch_size=batch_size,shuffle=True,**kwargs)

    return train_loader

def load_testing(te_feats, te_lab,te_lab_gender, batch_size, kwargs):
    te_data = torch.tensor(te_feats, dtype=torch.float32)
    te_lab = torch.tensor(te_lab, dtype=torch.long)
    te_lab_gender = torch.tensor(te_lab_gender, dtype=torch.long)
    te_dataset = torch.utils.data.TensorDataset(te_data, te_lab,te_lab_gender)
    test_loader = torch.utils.data.DataLoader(te_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return test_loader
