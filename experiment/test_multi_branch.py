import argparse
import torch

import os, json
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ImageLabel, prepare_colon, prepare_colon_test_2, \
                    prepare_gastric, prepare_k19, \
                    prepare_prostate_ubc_data, prepare_prostate_uhu_data
from model.model_cnn import SingleModelMultiBranch
from utils import mapping_type
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

def evaluate(test_dataset, model, args):
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, drop_last=False, num_workers=20)
    device = args.device
    progress = tqdm(total=len(test_dataloader))

    ground_truth_list = []
    prediction_list = []

    for _, (_, img_tensor, label) in enumerate(test_dataloader):
        img_tensor = img_tensor.to(device, dtype=torch.float32)
        
        img_tensor, label = img_tensor.to(device, dtype=torch.float32), label.to(device)
        x_colon, x_gastric, x_prostate, x_k19 = model(img_tensor)

        if args.dataset in ['colon_1','colon_2']:
            logits = x_colon
        elif args.dataset in ['prostate_1', 'prostate_2']:
            logits = x_prostate
        elif args.dataset in ['gastric']:
            logits = x_gastric
        elif args.dataset == 'k19':
            logits = x_k19

        predicted_labels = torch.argmax(logits, dim=1)
        prediction_list += predicted_labels.tolist()

        ground_truth_list += label.tolist()
        progress.update()
        
    progress.close()

    cancer_idx = [i for i in range(len(ground_truth_list)) if ground_truth_list[i] != 0]

    ground_truth_cancer_list = [ground_truth_list[i] for i in cancer_idx]
    prediction_cancer_list = [prediction_list[i] for i in cancer_idx]

    accuracy = accuracy_score(ground_truth_list, prediction_list)
    f1 = f1_score(ground_truth_list, prediction_list, average='macro', labels=list(set(ground_truth_list)))
    accuracy_cancer = accuracy_score(ground_truth_cancer_list, prediction_cancer_list)
    kappa = cohen_kappa_score(ground_truth_list, prediction_list, weights='quadratic', labels=list(set(ground_truth_list)))

    return accuracy, f1, accuracy_cancer, kappa

def save_result(path, output):
    with open(path + '/metrics.json', 'w') as outfile:
        json.dump(output, outfile)

def main():
    parser = argparse.ArgumentParser()

    # CHANGE
    parser.add_argument('--dataset', type=str, default='gastric')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--prefix_outdir', type=str, default="")
    parser.add_argument('--device', type=int, default=4)

    parser.add_argument('--optim_type', type=str, default='adamw')
    parser.add_argument('--cnn_type', type=str, default='convnext_large')

    # FIXED
    parser.add_argument('--out_dir', default='/data1/anhnguyen/image_caption/logs/single_model/')
    parser.add_argument('--valid_every', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--pretrain_path', type=str, default='/data1/anhnguyen/image_caption/logs/single_model/all-convnext_large/all-convnext_large-19.pt')
    parser.add_argument('--save_eval_path', type=str, default='/home/compu/anhnguyen/image_caption/evaluation/')

    args = parser.parse_args()

    if args.prefix_outdir != '':
        args.prefix_outdir = '-'.join((
                                    'all',
                                    args.cnn_type,
                                    args.prefix_outdir,
                                ))
    else:
        args.prefix_outdir = '-'.join((
                                    'all',
                                    args.cnn_type
                                ))

    args.out_dir = args.out_dir + '/' + args.prefix_outdir 

    
    model_name = args.pretrain_path.split('/')[-1].replace('.pt', '')
    args.save_eval_path = args.save_eval_path + '/' + args.dataset + '/' + model_name

    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)
    elif os.path.exists(args.save_eval_path + '/metrics.json'):
        raise(f'Already evaluated: {args.save_eval_path}')

    args.device = torch.device(f'cuda:{args.device}')

    num_class = 0
    if args.dataset == 'colon_1':
        _, _, test_set = prepare_colon("not_caption")
        num_class += mapping_type(args.dataset)
    elif args.dataset == 'colon_2':
       test_set = prepare_colon_test_2("not_caption")
       num_class += mapping_type(args.dataset)
    elif args.dataset == 'prostate_1':
       _, _, test_set = prepare_prostate_uhu_data("not_caption")
       num_class += mapping_type(args.dataset)
    elif args.dataset == 'prostate_2':
       test_set = prepare_prostate_ubc_data("not_caption")
       num_class += mapping_type(args.dataset)
    elif args.dataset == 'gastric':
       _, _, test_set = prepare_gastric(nr_classes=4, label_type='not_caption')
       num_class += mapping_type(args.dataset)
    elif args.dataset == 'k19':
       _, _, test_set = prepare_k19("not_caption")
       num_class += mapping_type(args.dataset)

    model = SingleModelMultiBranch(type=args.cnn_type)
    model = model.to(args.device)

    test_dataset = ImageLabel(test_set)

    checkpoint = torch.load(args.pretrain_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict']) 
    with torch.no_grad():
        accuracy, f1, accuracy_cancer, kappa = evaluate(test_dataset, model, args)

    print(accuracy)
    print(f1)
    print(accuracy_cancer)
    print(kappa)

if __name__ == '__main__':
    main()
