import argparse
import torch
import shutil
from utils import mapping_type
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ImageLabel, prepare_colon, prepare_colon_test_2, \
                    prepare_gastric, prepare_k19
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, \
                            precision_score, recall_score

from dataset import prepare_prostate_prostate_2_data
from model.model_cnn import SingleModel
import os
import json

from dataset import prepare_prostate_prostate_1_data

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def evaluate(test_dataset, model, args):
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, drop_last=False, num_workers=20)
    device = args.device
    progress = tqdm(total=len(test_dataloader))

    ground_truth_list = []
    prediction_list = []

    for _, (_, img_tensor, label) in enumerate(test_dataloader):
        img_tensor = img_tensor.to(device, dtype=torch.float32)
        
        img_tensor, label = img_tensor.to(device, dtype=torch.float32), label.to(device)
        logits = model(img_tensor)
        predicted_labels = torch.argmax(logits, dim=1)
        ground_truth_list += label.tolist()
        prediction_list += predicted_labels.tolist()
        
        progress.update()
        
    progress.close()

    cancer_idx = [i for i in range(len(ground_truth_list)) if ground_truth_list[i] != 0]

    ground_truth_cancer_list = [ground_truth_list[i] for i in cancer_idx]
    prediction_cancer_list = [prediction_list[i] for i in cancer_idx]

    accuracy = accuracy_score(ground_truth_list, prediction_list)
    f1 = f1_score(ground_truth_list, prediction_list, average='macro', labels=list(set(ground_truth_list)))
    accuracy_cancer = accuracy_score(ground_truth_cancer_list, prediction_cancer_list)
    kappa = cohen_kappa_score(ground_truth_list, prediction_list, weights='quadratic', labels=list(set(ground_truth_list)))
    precision = precision_score(ground_truth_list, prediction_list, labels=list(set(ground_truth_list)), average='macro')
    recall = recall_score(ground_truth_list, prediction_list, labels=list(set(ground_truth_list)), average='macro')
    print(accuracy, f1, accuracy_cancer, kappa, precision, recall)

    return accuracy, f1, accuracy_cancer, kappa

def prepare_for_testing(args):
    num_class = 0
    if args.dataset == 'colon_1':
        _, _, test_set = prepare_colon("not_caption")
        num_class += mapping_type(args.dataset)
    elif args.dataset == 'colon_2':
       test_set = prepare_colon_test_2("not_caption")
       num_class += mapping_type(args.dataset)
    elif args.dataset == 'prostate_1':
       _, _, test_set = prepare_prostate_prostate_1_data("not_caption")
       num_class += mapping_type(args.dataset)
    elif args.dataset == 'prostate_2':
       test_set = prepare_prostate_prostate_2_data("not_caption")
       num_class += mapping_type(args.dataset)
    elif args.dataset == 'gastric':
       _, _, test_set = prepare_gastric(nr_classes=4, label_type='not_caption')
       num_class += mapping_type(args.dataset)
    elif args.dataset == 'k19':
       _, _, test_set = prepare_k19("not_caption")
       num_class += mapping_type(args.dataset)

    model = SingleModel(type=args.cnn_type, num_class=num_class)
    model = model.to(args.device)
    model.eval()

    test_dataset = ImageLabel(test_set)
    
    if args.pretrain_path is not None:
        checkpoint = torch.load(args.pretrain_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model, test_dataset

def modify_args(args):
    model_name = args.pretrain_path.split('/')[-1].replace('.pt', '')
    args.save_eval_path = args.save_eval_path + '/' + args.dataset + '/' + model_name

    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)
    elif os.path.exists(args.save_eval_path + '/metrics.json'):
        print(f'Already evaluated: {args.save_eval_path}')
    elif len(os.listdir(args.save_eval_path)) == 0:
        shutil.rmtree(args.save_eval_path)
        os.makedirs(args.save_eval_path)

    args.device = torch.device(f'cuda:{args.device}')

    return args


def save_result(path, output):
    with open(path + '/metrics.json', 'w') as outfile:
        json.dump(output, outfile)

def main():
    parser = argparse.ArgumentParser()

    # CHANGE
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--dataset', default='colon_1')
    parser.add_argument('--list', nargs='+')
    parser.add_argument('--cnn_type', type=str, default='')

    # FIXED
    parser.add_argument('--freeze_lm', type=bool, default=False)
    parser.add_argument('--mapping_type', type=str, choices='mlp', default='mlp')
    parser.add_argument('--pretrain_path', type=str, default='', help='Path of .pt file of pre-trained model')
    parser.add_argument('--save_eval_path', type=str, default='/home/compu/anhnguyen/image_caption/evaluation/')

    args = parser.parse_args()
    args.cnn_type = args.pretrain_path.split('-')[-2]
    args.device = 5
    args = modify_args(args)

    model, test_dataset = prepare_for_testing(args)
    output = evaluate(test_dataset, model, args)
    save_result(args.save_eval_path, output)


if __name__ == '__main__':
    main()