from __future__ import print_function
import argparse
import math
import os
import pdb
import sys
import time
from datetime import datetime
import numpy as np
import scipy.io as scio
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils import model_zoo
import utils
import model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import log_print as log
from model import HFFT as Net
from tqdm import tqdm


def train(args, model, train_loader, epoch, global_iter, device, fold_log,fold_id):
    
    global_iter += 1
    tra_correct = 0.
    tra_correct_gender = 0.
    tra_loss = 0.
    tra_num = 0
    model.train()
    for batch_idx, (tra_data, tra_label,tra_label_gender) in enumerate(train_loader):#len(train_loader.dataset)=1240
        tra_data, tra_label, tra_label_gender = tra_data.to(device), tra_label.to(device),tra_label_gender.to(device)# tra_data.shape:(40,1,7680)
        # print(tra_data.shape)
        # print( tra_label.shape)
        # pdb.set_trace()
        # print('tra_lab:',tra_label)
        # print("tra_label_gender:",tra_label_gender)
        # import pdb; pdb.set_trace()
        optimizer.zero_grad()#清空模型参数的梯度
        tra_pred = model(tra_data)
        # print('tra_pred:',tra_pred)
        # pdb.set_trace()
        pred_label = F.log_softmax(tra_pred, dim=1)
        # print(pred_label)
        # pdb.set_trace()
        
        loss_emo = F.nll_loss(pred_label, tra_label)
        
        loss = loss_emo 
        # print("loss:",loss)
        loss.backward()
        optimizer.step()
        # calculate the loss and acc of training data
        tra_loss += loss.item()  # sum up batch loss
        # print('batch loss:',tra_loss)
        tra_pred = tra_pred.data.max(1)[1] # get the index of the max log-probability
        
        # print('tra_pred index:',tra_pred)
        # import pdb; pdb.set_trace()    
        tra_correct += tra_pred.eq(tra_label.data.view_as(tra_pred)).cpu().sum()
        # print('tra_correct:',tra_correct)
        tra_num += len(tra_data)
        # pdb.set_trace()
        # print('tra_num:',tra_num)
        # pdb.set_trace()

        if batch_idx % args.log_interval == 0 or batch_idx == math.floor(len(train_loader.dataset)/args.batch_size)-1:
            log(fold_log,'Fold:{}/{}\t  Train Epoch: {}/{} [{}/{} ({:.0f}%)]\t Loss all: {:.6f},  Accuarcy_emo: {:.2f}%  '.format(fold_id,args.fold_num,
                epoch,args.epoch_num, tra_num, len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss.item(), 100. * tra_correct / tra_num))
        # pdb.set_trace()


def test(args,model, device, test_loader, epoch,fold_log,fold_id):
    model.eval()
    test_loss = 0
    correct = 0
    correct_gender=0
    pred_label,pred_label_gender = [],[]
    true_label,true_label_gender = [],[]
    with torch.no_grad():
        for data, target,target_gender in test_loader:
            data, target,target_gender = data.to(device), target.to(device),target_gender.to(device)
            # print('true label:',target)
            # print('true gender label:',target_gender)
            # pdb.set_trace()
            output = model(data)
            # print('output:',output)
            output = F.log_softmax(output, dim=1)
            
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            # print('batch loss:',test_loss)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            # print('pred gender index:',pred_gender)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # print('correct_gender:',correct_gender)
            # pdb.set_trace()
            pred_label_batch, true_label_batch = pred.cpu().numpy(), target.cpu().numpy()
            pred_label += pred_label_batch.tolist()
            true_label += true_label_batch.tolist()
  
            # print('pred_label:',pred_label)
            # print('true_label:',true_label)
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    
    for i in range(len(pred_label)): 
        pred_label[i] = pred_label[i][0]

    # print('pred_label:',pred_label)
    # pdb.set_trace()

    log(fold_log,'\nFold:{}/{}, Test epoch:{}/{}, Test set: Average emotion loss: {:.4f}, all emo acc: [{}/{}]  {:.4f}%'.format(fold_id,args.fold_num,epoch,args.epoch_num,test_loss, 
        correct, len(true_label), 100. * correct / len(test_loader.dataset))) 
    # log(fold_log, 'testing confusion matrix: \n{}'.format(confusion_matrix(true_label, pred_label)))              
    # log(fold_log, 'testing classification report: \n{}'.format(classification_report(true_label, pred_label, digits=4)))
    # import pdb; pdb.set_trace()
    return test_acc,pred_label, true_label,test_loss

def find_max_acc(args, model, fold_path, epoch_acc, max_acc, epoch, max_acc_epoch, epoch_pred_label, epoch_true_label, pred_label_fold, true_label_fold):

    if epoch_acc > max_acc:
        max_acc = epoch_acc
        max_acc_epoch = epoch                                  
        pred_label_fold = epoch_pred_label
        true_label_fold = epoch_true_label
        # # saving model
        # model_path = os.path.join(fold_path, args.model_type + '_epoch' + str(max_acc_epoch) + '_' + str(datetime.now().strftime('%Y%m%d-%H%M%S')) + '.pt')
        # #删除上一个.pt文件
        # utils.delete_file(fold_path)
        # torch.save(model.state_dict(), model_path)
        # log(fold_log, "{} model checkpoint saved at {}".format(datetime.now(), model_path))
        log(fold_log, 'testing confusion matrix: \n{}'.format(confusion_matrix(epoch_true_label, epoch_pred_label)))              
        log(fold_log, 'testing classification report: \n{}'.format(classification_report(epoch_true_label, epoch_pred_label, digits=6)))
    
    return  max_acc, max_acc_epoch, pred_label_fold, true_label_fold


    
        
def print_result(args, results_path, results_log_path, fold_num, pred_label, true_label, max_acc):

    '''save the true and predicted labels'''
    # for i in range(len(pred_label)): pred_label[i] = pred_label[i][0]
    mat_name = results_path + '/' + args.model_type + '_max_accuracy_' +'fold'+str(fold_num) + '_results.mat'
    log(results_log_path, "{} saving results of folds ...".format(datetime.now()))
    scio.savemat(mat_name, {'pred_lab': pred_label, 'tru_lab': true_label, 'acc_fold': max_acc})
    log(results_log_path, "{} mat file is saved at {}\n".format(datetime.now(), mat_name))
    '''print the results information'''
    log(results_log_path, 'max acc in fold{}: {}\n'.format(fold_num,max_acc))
    accuracy_all = np.mean(np.equal(pred_label, true_label))
    log(results_log_path, "time used: {:4f}s \nfinal accuaary: {:6f}\n".format(time.perf_counter() - start, accuracy_all))
    log(results_log_path,'testing confusion matrix: \n{}'.format(confusion_matrix(true_label, pred_label)))
    log(results_log_path, 'testing classification report: \n{}'.format(classification_report(true_label, pred_label, digits=6)))


        
def print_result_all(args, results_path, results_log_path,  pred_label, true_label, max_acc):

    '''save the true and predicted labels'''
    # for i in range(len(pred_label)): pred_label[i] = pred_label[i][0]
    mat_name = results_path + '/' + args.model_type + '_Max_Accuracy_all_' + str(args.fold_num)+ 'fold_results.mat'
    log(results_log_path, "{} saving results of folds ...".format(datetime.now()))
    scio.savemat(mat_name, {'pred_lab': pred_label, 'tru_lab': true_label, 'acc_fold': max_acc})
    log(results_log_path, "{} mat file is saved at {}\n".format(datetime.now(), mat_name))
    '''print the results information'''
    log(results_log_path, 'max acc in each fold: {}\n'.format(max_acc))
    std_all=np.std(max_acc)
    accuracy_all = np.mean(np.equal(pred_label, true_label))
    log(results_log_path, "time used: {:4f}s \nfinal accuaary: {:6f}\nfinal std: {:6f}\n".format(time.perf_counter() - start, accuracy_all,std_all))
    log(results_log_path,'testing confusion matrix: \n{}'.format(confusion_matrix(true_label, pred_label)))
    log(results_log_path, 'testing classification report: \n{}'.format(classification_report(true_label, pred_label, digits=6)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch HFFT Training and Testing')
    #Dataset parameter
    parser.add_argument('--dataset', default='deap')#default='mahnob'
    parser.add_argument('--root_dir', default='/home/wsco/wy/HFFT/data')
    parser.add_argument('--feat_type', default='deap_resp')#default='mahnob_resp'
    parser.add_argument('--fold_num', default=32)#default=27
    parser.add_argument('--label_type', default='valence')#default='arousal'

    #model parameter 
    parser.add_argument('--model_type', default='HFFT')
    parser.add_argument('--fea_size', default=7680,type=int)
    parser.add_argument('--patch_size', default=768,type=int)#default=1536
    parser.add_argument('--inner_stride', default=128,type=int)# default=256
    parser.add_argument('--outer_dim', default=384, type=int)    
    parser.add_argument('--inner_dim', default=64, type=int)   
    parser.add_argument('--depth_frame', default=6, type=int)
    parser.add_argument('--depth_seg_emo', default=6, type=int)
    parser.add_argument('--depth_seg_gen', default=6, type=int)
    parser.add_argument('--outer_num_heads', default=6, type=int)
    parser.add_argument('--inner_num_heads', default=6, type=int)
    parser.add_argument('--drop_rate', default=0, type=float)
    parser.add_argument('--attn_drop_rate', default=0, type=float)
    parser.add_argument('--drop_path_rate', default=0, type=float)

   

    #optimizer parameter
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--wd', default=1e-2, type=float)

    #train/test parameter
    parser.add_argument('--seed', default=16, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--epoch_num', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--w1', type=float, default=1)
    parser.add_argument('--w2', type=float, default=1)
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args()


    start = time.perf_counter()
    ##########################################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if cuda else {}
    ###########################################
    results_path, results_log_path = utils.set_path(args)
    # pdb.set_trace()
    log(results_log_path,args)
    all_id,all_resp,all_arousal_label,all_valence_label,all_gender_label = utils.parse_features(args)
    all_resp=all_resp[:,np.newaxis,:,:]
    # print(all_resp)
    # pdb.set_trace()
    if args.dataset == 'deap' :
        sub_num=32
        idx_all=np.arange(0,1280)
    elif args.dataset=='mahnob':
        sub_num=27
        idx_all=np.arange(0,527)

    '''LOSO processing'''
    pred_label_kfold ,pred_label_kfold_gender= [],[]
    true_label_kfold ,true_label_kfold_gender= [],[]
    max_acc_kfold, max_acc_kfold_gender = [],[]

    for fold_id in range(1,sub_num+1):
        # creat the folder for saving the results of each fold
        fold_path, fold_log = utils.self_makedirs(results_path, fold_id)
        # # Generate batch data
        idx=np.where(all_id==fold_id)
        # print(idx)
        # pdb.set_trace()
        te_idx_start, te_idx_end= np.amin(idx), np.amax(idx)
        # print(te_idx_start)
        # print(te_idx_end)
        te_idx =  idx_all[te_idx_start:te_idx_end+1]
        tra_idx = np.delete(idx_all,te_idx)
        # print(te_idx)
        # print(tra_idx)
        # pdb.set_trace()
        tra_feats, te_feats =  all_resp[tra_idx],  all_resp[te_idx]
        # print(te_feats.shape)
        # pdb.set_trace()
        #提取情绪标签
        if args.label_type=='valence':
            tra_emo_lab, te_emo_lab = all_valence_label[tra_idx], all_valence_label[te_idx]
        elif args.label_type=='arousal':
            tra_emo_lab, te_emo_lab = all_arousal_label[tra_idx], all_arousal_label[te_idx]
        tra_gender_lab,te_gender_lab=all_gender_label[tra_idx],all_gender_label[te_idx]
        emo_num = 2
        tra_data_num = len(tra_emo_lab)
        te_data_num = len(te_emo_lab)
        # print(tra_feats.shape)
        # print(tra_idx.shape)
        # print(te_feats.shape)
        # print(te_idx.shape)
        # pdb.set_trace()
        seq_len, feat_channel = tra_feats.shape[3],1
        # print(seq_len)
        # pdb.set_trace()
        log(fold_log,
            'train data num: {}, test data num: {},\nseq_len: {},  feature channel: {}, emotion num: {}'
            .format(tra_data_num, te_data_num, seq_len, feat_channel, emo_num))
        log(fold_log, args)
        # import pdb; pdb.set_trace()
        train_loader = utils.load_training(tra_feats, tra_emo_lab, tra_gender_lab,args.batch_size, kwargs)
        test_loader =utils.load_testing(te_feats, te_emo_lab,te_gender_lab, args.batch_size, kwargs)

        # import pdb; pdb.set_trace()
        model = Net(img_size=[1,args.fea_size],patch_size=[1,args.patch_size],in_chans=feat_channel,num_classes=emo_num,outer_dim=args.outer_dim,inner_dim=args.inner_dim,
                    depth_frame=args.depth_frame,depth_seg_emo=args.depth_seg_emo,depth_seg_gender=args.depth_seg_gen,outer_num_heads=args.outer_num_heads,inner_num_heads=args.inner_num_heads,
                    drop_rate=args.drop_rate,attn_drop_rate=args.attn_drop_rate,drop_path_rate=args.drop_path_rate,inner_stride=[1,args.inner_stride]).to(device)
        log(fold_log,model)
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,eps=args.eps,weight_decay=args.wd)
        # pdb.set_trace()
        global_iter = 0
        max_acc_fold ,max_acc_fold_gender= 0.,0.
        max_acc_epoch , max_acc_epoch_gender= 0,0
        pred_label_fold,pred_label_fold_gender = [],[]
        true_label_fold,true_label_fold_gender = [],[]

        for epoch in range(1, args.epoch_num + 1):
            train(args, model, train_loader, epoch, global_iter, device, fold_log,fold_id)           
            test_acc,pred_label_epoch, true_label_epoch,test_loss= test(args,model, device, test_loader,epoch, fold_log,fold_id)
            # print("test_acc_gender:",test_acc_gender)
            # pdb.set_trace()
            max_acc_fold, max_acc_epoch, pred_label_fold, true_label_fold = find_max_acc(args, model, fold_path, 
                                                                                    test_acc, max_acc_fold, epoch, max_acc_epoch,
                                                                                    pred_label_epoch, true_label_epoch, 
                                                                                    pred_label_fold, true_label_fold)
        

            log(fold_log,'Test max Accuracy: {:.4f}% in epoch: {}\n'
                                                .format(100. * max_acc_fold, max_acc_epoch))

  
        print('\n')
        # import pdb; pdb.set_trace()
        print_result(args, fold_path, fold_log, fold_id,pred_label_fold, true_label_fold, max_acc_fold)
   
        # import pdb; pdb.set_trace()
        pred_label_kfold.extend(pred_label_fold)
        true_label_kfold.extend(true_label_fold)
        max_acc_kfold.append(max_acc_fold) 


    #
    print_result_all(args, results_path, results_log_path, pred_label_kfold, true_label_kfold, max_acc_kfold)
    log(results_log_path, 'all folds processing is finished!\n')
