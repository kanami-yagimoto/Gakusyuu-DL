import torch
import torch.nn as nn
import os
from model import CNN_ARG_L0, CNN_ARG_L1
import argparse
from tokenizer import FastaTokenizer
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import BatchSampler
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='The path to the input data', default='../example/example.fasta')
parser.add_argument('--l0_model_path', type=str, help='The level 0 model path', default='logs/l0.pt')
parser.add_argument('--l1_model_path', type=str, help='the level 1 model path', default='logs/l1.pt')
parser.add_argument('--l2_model_path', type=str, help='the level 2 model path', default='logs/l2.pt')
parser.add_argument('--exp_name', type=str, help='the output name', default='../result/ooo')


args = parser.parse_args()


def level0_prediction(data_loader, ARG_L0_model):
    pred = []
    for batch_idx, batch in enumerate(data_loader):
        data_i = batch['input_ids']
        if torch.cuda.is_available():
            data_i = data_i.cuda()
        with torch.no_grad():
            batch_pred = ARG_L0_model(data_i).data.argmax(1).cpu().numpy()
        pred.append(batch_pred)

    pred = np.concatenate(pred, axis=0)
    return pred


def level1_prediction(data,arg_index, batch_size, ARG_L1_model):
    l1_drug_pred = []
    l1_mechan_pred = []
    l1_intri_pred = []

    for index in BatchSampler(SequentialSampler(arg_index), batch_size=batch_size, drop_last=False):
        batch = [data[arg_index[k]] for k in index]
        data_i = data.collate_fn(batch)['input_ids']

        if torch.cuda.is_available():
            data_i = data_i.cuda()
        with torch.no_grad():
            drug_pred, mechan_pred, intri_pred = ARG_L1_model(data_i)

        drug_pred = drug_pred.data.argmax(1).cpu().numpy()
        mechan_pred = mechan_pred.data.argmax(1).cpu().numpy()
        intri_pred = intri_pred.data.argmax(1).cpu().numpy()

        l1_drug_pred.append(drug_pred)
        l1_mechan_pred.append(mechan_pred)
        l1_intri_pred.append(intri_pred)

    l1_drug_pred = np.concatenate(l1_drug_pred)
    l1_mechan_pred = np.concatenate(l1_mechan_pred)
    l1_intri_pred = np.concatenate(l1_intri_pred)
    
    l1_drug_pred = [drug_dict[i] for i in l1_drug_pred]
    l1_mechan_pred = [mechanism_dict[i] for i in l1_mechan_pred]
    l1_intri_pred = [intrinsic_dict[i] for i in l1_intri_pred]
    l1_drug_map = {k:v for k,v in zip(arg_index, l1_drug_pred)}

    return l1_drug_map, l1_drug_pred, l1_mechan_pred, l1_intri_pred


if __name__ == "__main__":
    
    # constant dictonary for storing labels
    drug_dict = {'aminoglycoside': 0, 'macrolide-lincosamide-streptogramin': 1, 'polymyxin': 2, 
                  'fosfomycin': 3, 'multidrug': 7, 'bacitracin': 5, 'quinolone': 6, 
                  'trimethoprim': 4, 'chloramphenicol': 8,'tetracycline': 9, 'rifampin': 10, 
                  'beta_lactam': 11, 'others': 12, 'sulfonamide': 13, 'glycopeptide': 14,'nonarg':15}

    mechanism_dict = {'antibiotic target alteration':0, 'antibiotic target replacement':1,
                      'antibiotic target protection':2, 'antibiotic inactivation':3,
                      'antibiotic efflux':4,
                      'reduced permeability to antibiotic':5, 'resistance by absence':5,'others':5,'nonarg':6}


    intrinsic_dict = {'acquired':1, 'intrisic':0}
    beta_lactam_class_dict = {'A':0, 'B1':1, 'B2':2, 'B3':3, 'C':4, 'D':5}
    beta_lactam_class_dict = {v:k for k,v in beta_lactam_class_dict.items()}
    drug_dict = {v:k for k,v in drug_dict.items()}
    mechanism_dict = {v:k for k,v in mechanism_dict.items()}
    intrinsic_dict = {v:k for k,v in intrinsic_dict.items()}

    batch_size = 100

    # load models from path
    ARG_L0_model = torch.load(args.l0_model_path)
    ARG_L1_model = torch.load(args.l1_model_path)
    ARG_L2_model = torch.load(args.l2_model_path)

    data = FastaTokenizer(args.data_path)
    sampler = SequentialSampler(data)
    batch_sampler = BatchSampler(sampler, batch_size, False)
    data_loader = torch.utils.data.DataLoader(data, batch_sampler=batch_sampler, collate_fn = data.collate_fn)

    if torch.cuda.is_available():
        ARG_L0_model = ARG_L0_model.cuda()
        ARG_L1_model = ARG_L1_model.cuda()
        ARG_L2_model = ARG_L2_model.cuda()

    # bulid the level 0 prediction part
    pred = level0_prediction(data_loader=data_loader, ARG_L0_model=ARG_L0_model)
    # the pred are exactly the level 0 prediction results

    
    # bulid the level 1 prediction 

    arg_index = np.where(pred!=0)[0]
    
    if(len(arg_index)!=0):
        l1_drug_map, l1_drug_pred, l1_mechan_pred, l1_intri_pred = level1_prediction(data, arg_index, batch_size, ARG_L1_model)

    # build the level 2 prediction
    
    beta_index = np.array([k for k in arg_index if l1_drug_map[k]=='beta_lactam'])

    if (len(beta_index!=0)):
        beta_preds = []
        for index in BatchSampler(SequentialSampler(beta_index), batch_size=batch_size, drop_last=False):
            batch = [data[beta_index[k]] for k in index]
            data_i = data.collate_fn(batch)['input_ids']

            if torch.cuda.is_available():
                data_i = data_i.cuda()
            with torch.no_grad():
                beta_pred = ARG_L2_model(data_i)

            beta_pred = beta_pred.data.argmax(1).cpu().numpy()
            beta_preds.append(beta_pred)

        beta_preds = np.concatenate(beta_preds)
        l2_beta_pred = [beta_lactam_class_dict[i] for i in beta_preds]

    # store the results
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
    

    with open(args.exp_name + '/prediction.out','w') as f:
        f.write("%s\t%s\t%s\t%s\t%s\t%s\n"%("Id","ARG/Non-ARG","antibiotic target", "resistance mechanism", "transferability", "beta lactamase subtype"))
        for index in range(len(data)):
            if pred[index] == 0:
                out = ['nonarg'] + ['']*4
            else:
                second_index = np.where(arg_index==index)[0]
                assert len(second_index) == 1
                second_index = second_index[0]
                out = ['arg',l1_drug_pred[second_index], l1_mechan_pred[second_index], l1_intri_pred[second_index]]

                if l1_drug_pred[second_index] == 'beta_lactam':
                    third_index = np.where(beta_index==index)[0]
                    assert len(third_index) == 1
                    third_index = third_index[0]
                    out.append(l2_beta_pred[third_index])
                else:
                    out.append('')

            sid = data.get_ids(index)
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t\n"%(sid,out[0],out[1],out[2],out[3],out[4]))
    
