import argparse
import torch

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from torch.nn import functional as nnf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ImageLabel, prepare_colon, \
                    prepare_prostate_uhu_data, prepare_gastric, prepare_k19
from model.model_cnn import SingleModelMultiBranch
from utils import save_config, mapping_type
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, recall_score, precision_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.transforms import Resize

def train(args, train_dataset, valid_dataset, model, checkpoint=None):
    batch_size = args.bs
    device = args.device
    epochs = args.epochs
    
    model.train()
    if args.optim_type == 'adamw':  
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=10)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=10)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs//3, T_mult=1, eta_min=args.lr*0.1, last_epoch=-1)

    start_epoch = 0

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['model_state_dict'])
        scheduler.load_state_dict(checkpoint['schedulerr_state_dict'])
        start_epoch = checkpoint['epoch']+1
        

    writer = SummaryWriter(args.out_dir)

    fake_label = torch.tensor(0).to(device)

    for epoch in range(start_epoch, epochs):
        print(f">>> Training epoch {epoch}")
        progress = tqdm(total=len(train_dataloader), desc=args.prefix_outdir)
        total_train_loss = 0

        for idx, (img_path, img_tensor, label) in enumerate(train_dataloader):
            """
            - img_path: tuple, len = batch_size
            - img_tensors: tensor, shape (bs, c, w, h)
            - label: label with int type
            """
            model.zero_grad()
            img_tensor, label = img_tensor.to(device, dtype=torch.float32), label.to(device)
            if args.cnn_type in ['vit_b_16', 'clip']:
                img_tensor = Resize(224)(img_tensor)
            x_colon, x_gastric, x_prostate, x_k19 = model(img_tensor)

            loss = 0
            for i in range(args.bs):
                if 'gastric' in img_path[i]:
                    loss += nnf.cross_entropy(x_gastric[i], label[i]) \
                            + 0*nnf.cross_entropy(x_colon[i], fake_label) \
                            + 0*nnf.cross_entropy(x_prostate[i], fake_label) \
                            + 0*nnf.cross_entropy(x_k19[i], fake_label) \
                            
                elif 'colon_1' in img_path[i]:
                    loss += nnf.cross_entropy(x_colon[i], label[i]) \
                            + 0*nnf.cross_entropy(x_gastric[i], fake_label) \
                            + 0*nnf.cross_entropy(x_prostate[i], fake_label) \
                            + 0*nnf.cross_entropy(x_k19[i], fake_label) \

                elif 'prostate_harvard' in img_path[i]:
                    loss += nnf.cross_entropy(x_prostate[i], label[i]) \
                            + 0*nnf.cross_entropy(x_colon[i], fake_label) \
                            + 0*nnf.cross_entropy(x_gastric[i], fake_label) \
                            + 0*nnf.cross_entropy(x_k19[i], fake_label) \

                elif 'Domain_Invariance' in img_path[i]:
                    loss += nnf.cross_entropy(x_k19[i], label[i]) \
                            + 0*nnf.cross_entropy(x_colon[i], fake_label) \
                            + 0*nnf.cross_entropy(x_prostate[i], fake_label) \
                            + 0*nnf.cross_entropy(x_gastric[i], fake_label) \

                else:
                    raise ValueError(f'check {img_path[i]}')

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'schedulerr_state_dict': scheduler.state_dict(),
            'loss': loss,
            }, 
            os.path.join(args.out_dir, f"{args.prefix_outdir}-{epoch}.pt"),
        )
        ground_truth_list = []
        prediction_list = []

        if epoch % args.valid_every == 0:
            print(f">>> Evaluating epoch {epoch}")
            progress = tqdm(total=len(valid_dataloader), desc=args.prefix_outdir)
            total_valid_loss = 0
            for idx, (img_path, img_tensor, label) in enumerate(valid_dataloader):
                with torch.no_grad():
                    if args.cnn_type in ['vit_b_16', 'clip']:
                        img_tensor = Resize(224)(img_tensor)
                    img_tensor, label = img_tensor.to(device, dtype=torch.float32), label.to(device)
                    
                    x_colon, x_gastric, x_prostate, x_k19 = model(img_tensor)

                    loss = 0
                    for i in range(args.bs):
                        if 'gastric' in img_path[i]:
                            loss += nnf.cross_entropy(x_gastric[i], label[i]) \
                                    + 0*nnf.cross_entropy(x_colon[i], fake_label) \
                                    + 0*nnf.cross_entropy(x_prostate[i], fake_label) \
                                    + 0*nnf.cross_entropy(x_k19[i], fake_label) 
                            
                            prediction_list.append(torch.argmax(x_gastric[i]).cpu())
                                    
                        elif 'colon_1' in img_path[i]:
                            loss += nnf.cross_entropy(x_colon[i], label[i]) \
                                    + 0*nnf.cross_entropy(x_gastric[i], fake_label) \
                                    + 0*nnf.cross_entropy(x_prostate[i], fake_label) \
                                    + 0*nnf.cross_entropy(x_k19[i], fake_label) 

                            prediction_list.append(torch.argmax(x_colon[i]).cpu())

                        elif 'prostate_harvard' in img_path[i]:
                            loss += nnf.cross_entropy(x_prostate[i], label[i]) \
                                    + 0*nnf.cross_entropy(x_colon[i], fake_label) \
                                    + 0*nnf.cross_entropy(x_gastric[i], fake_label) \
                                    + 0*nnf.cross_entropy(x_k19[i], fake_label) 
                            
                            prediction_list.append(torch.argmax(x_prostate[i]).cpu())
                            

                        elif 'Domain_Invariance' in img_path[i]:
                            loss += nnf.cross_entropy(x_k19[i], label[i]) \
                                    + 0*nnf.cross_entropy(x_colon[i], fake_label) \
                                    + 0*nnf.cross_entropy(x_prostate[i], fake_label) \
                                    + 0*nnf.cross_entropy(x_gastric[i], fake_label) 
                            
                            prediction_list.append(torch.argmax(x_k19[i]).cpu())

                        else:
                            raise ValueError(f'check {img_path[i]}')

                    ground_truth_list += label.tolist()
                    total_valid_loss += loss.item()
                    progress.set_postfix({"loss": loss.item()})
                    progress.update()
            progress.close()
        
        cancer_idx = [i for i in range(len(ground_truth_list)) if ground_truth_list[i] != 0]

        ground_truth_cancer_list = [ground_truth_list[i] for i in cancer_idx]
        prediction_cancer_list = [prediction_list[i] for i in cancer_idx]

        accuracy = accuracy_score(ground_truth_list, prediction_list)
        f1 = f1_score(ground_truth_list, prediction_list, average='macro', labels=list(set(ground_truth_list)))
        accuracy_cancer = accuracy_score(ground_truth_cancer_list, prediction_cancer_list)
        kappa = cohen_kappa_score(ground_truth_list, prediction_list, weights='quadratic', labels=list(set(ground_truth_list)))
        recall = recall_score(ground_truth_list, prediction_list, average='macro')
        precision = precision_score(ground_truth_list, prediction_list, average='macro')

        if args.dataset in ['k19']:
            avg_metrics = (accuracy + recall + f1 + precision)/4
        else:
            avg_metrics = (accuracy + kappa + accuracy_cancer + f1)/4


        if args.dataset in ['k19']:
            writer.add_scalars('Loss', {'train_loss':total_train_loss/len(train_dataset), 
                                    'valid_loss':total_valid_loss/len(valid_dataset),
                                    'val_accuracy':accuracy,
                                    'val_f1':f1,
                                    'val_re':recall,
                                    'val_pre':precision,
                                    'avg': avg_metrics
                                }, epoch)
        else:
            writer.add_scalars('Loss', {'train_loss':total_train_loss/len(train_dataset), 
                                        'valid_loss':total_valid_loss/len(valid_dataset),
                                        'val_accuracy':accuracy,
                                        'val_f1':f1,
                                        'val_acc_cancer':accuracy_cancer,
                                        'val_kappa':kappa,
                                        'avg': avg_metrics
                                    }, epoch)
    return model


def main():
    parser = argparse.ArgumentParser()

    # CHANGE
    parser.add_argument('--dataset', nargs='+', default=['colon_1', 'prostate_1', 'gastric', 'k19'])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--prefix_outdir', type=str, default="")
    parser.add_argument('--device', type=int, default=3)

    parser.add_argument('--optim_type', type=str, default='adamw')
    parser.add_argument('--cnn_type', type=str, default='resnet50')

    # FIXED
    parser.add_argument('--out_dir', default='/data1/anhnguyen/image_caption/logs/single_model/')
    parser.add_argument('--valid_every', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--pretrain_path', type=str, default=None, help='Path of .pt file of pre-trained model')
    
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

    if args.pretrain_path is None:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        else:
            raise ValueError(f'The path already existed: {args.out_dir}')
        
        save_config(args)
    
    args.device = torch.device(f'cuda:{args.device}')
    
    train_set = []
    valid_set = []
    num_class = 0
    for dataset in args.dataset:
        if dataset == 'colon_1':
            train_set_t, valid_set_t, _ = prepare_colon('not_caption')
            train_set += train_set_t
            valid_set += valid_set_t
            num_class += mapping_type(dataset)

        elif dataset == 'prostate_1':
            train_set_t, valid_set_t, _ = prepare_prostate_uhu_data('not_caption')
            train_set += train_set_t
            valid_set += valid_set_t
            num_class += mapping_type(dataset)
        
        elif dataset == 'gastric':
            train_set_t, valid_set_t, _ = prepare_gastric(nr_classes=4, label_type='not_caption')
            train_set += train_set_t
            valid_set += valid_set_t
            num_class += mapping_type(dataset)
        
        elif dataset == 'k19':
            train_set_t, valid_set_t, _ = prepare_k19('not_caption')
            train_set += train_set_t
            valid_set += valid_set_t
            num_class += mapping_type(dataset)
        
        else:
            raise ValueError(f'Invalid dataset: {dataset}')

    model = SingleModelMultiBranch(type=args.cnn_type)
    model = model.to(args.device)

    train_dataset = ImageLabel(train_set)
    valid_dataset = ImageLabel(valid_set)

    checkpoint = None
    if args.pretrain_path is not None:
        checkpoint = torch.load(args.pretrain_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict']) 
    train(args, train_dataset, valid_dataset, model, checkpoint)


if __name__ == '__main__':
    main()