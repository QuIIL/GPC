import os
import json
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
                            recall_score, precision_score

def save_config(args):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix_outdir}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)

def compare_output(cap, gen_cap):
    if cap == gen_cap:
        return True
    return False

def save_evaluation_result(path, output):
    with open(path + '/true.json', 'w') as outfile:
        json.dump(output['true_predict'], outfile)
    
    with open(path + '/wrong.json', 'w') as outfile:
        json.dump(output['false_predict'], outfile)
    
    with open(path + '/invalid.json', 'w') as outfile:
        json.dump(output['invalid_predict'], outfile)
    
    with open(path + '/true_post.json', 'w') as outfile:
        json.dump(output['true_post_process'], outfile)

    with open(path + '/false_post.json', 'w') as outfile:
        json.dump(output['false_post_process'], outfile)

def save_metrics(path, metrics):
    with open(path + '/metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)

def generate(
    model,
    image_tensor,
    entry_length=30,  # maximum number of words
    # top_p=0.8,
    temperature=1.0,
    # stop_token: str = ".",
    device=None,
    args=None
):
    model.eval()

   # model = model.module
    tokenizer = model.get_tokenizer()

    # FAKE
    prompt = "What are some phases for cancer?"
    inputs = tokenizer(prompt, return_tensors="pt")

    token_embeddings = model.lm.get_token_embeddings().to(device)
    bs_generated = token_embeddings(inputs.input_ids.to(args.device))  # 1, length, 768

    gen_ids = []
    
    for _ in range(entry_length+1):
        outputs = model.lm(inputs_embeds=bs_generated,                              # bs x prefix_len x 768
                            attention_mask=None)     # TODO: check attention_mask
        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)       # bs x 50272
        next_token = torch.argmax(logits, -1).unsqueeze(1)                          # bs x 1
        gen_ids.append(next_token)
        next_token_embed = token_embeddings(next_token).to(device)

        bs_generated = torch.cat((bs_generated, next_token_embed), dim=1)           # bs x prefix_len + 1 x 768



    with torch.no_grad():
        # bs_generated = model.encoder_forward(image_tensor)  # bs x prefix_length x emb_dim
        if args.scaling is not None:
            bs_generated_list = []
            for image_tensor_i in image_tensor:
                bs_generated = model.encoder(image_tensor_i).squeeze((-1,-2))
                bs_generated = model.mlp(bs_generated).view(-1, args.prefix_length, args.embedding_size)
                bs_generated_list.append(bs_generated)
            bs_generated = torch.cat(bs_generated_list, dim=1).to(args.device)
        else:
            if args.encoder == 'clip':
                bs_generated = model.encoder(image_tensor).image_embeds.squeeze((-1,-2))
            else:
                bs_generated = model.encoder(image_tensor).squeeze((-1,-2))
            bs_generated = model.mlp(bs_generated).view(-1, args.prefix_length, args.embedding_size) # bs x prefix_len x lm_dim
        token_embeddings = model.lm.get_token_embeddings().to(device)

        tokens = None

        for _ in range(entry_length+1):
            outputs = model.lm(inputs_embeds=bs_generated,                              # bs x prefix_len x 768
                               attention_mask=None)     # TODO: check attention_mask
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)       # bs x 50272
            next_token = torch.argmax(logits, -1).unsqueeze(1)                          # bs x 1
            next_token_embed = token_embeddings(next_token).to(device)                  # bs x 1 x 768  (dim 1 to concat with prefix)

            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)                         # bs x (len+1)
            bs_generated = torch.cat((bs_generated, next_token_embed), dim=1)           # bs x prefix_len + 1 x 768

        output_list = tokens.tolist() # drop the '/s' token at the beginning
        output_text = tokenizer.batch_decode(output_list)
        for i in range(len(output_text)):
            output_text[i] = output_text[i].replace('</s>','').replace('<s>','')
            output_text[i] = output_text[i].split('.')[0] + '.'

    return output_text

def check_valid(type, caption):
    colon_caption = [
        "benign.",
        "cancer moderately differentiated.",
        "cancer poorly differentiated.",
        "cancer well differentiated."
    ]

    prostate_caption = [
        "benign.",
        "cancer grade 3.",
        "cancer grade 4.",
        "cancer grade 5."
    ]
    if type == 'colon_1' or type == 'colon_2':
        if caption in colon_caption:
            return True
        else:
            return False
    
    if type == 'panda':
        if caption in prostate_caption:
            return True
        else:
            return False

def compute_metrics(output, args):
    if args.dataset in ['colon_1','colon_2', 'gastric']:
        labels=['BN', 'WD', 'MD', 'PD']
    elif args.dataset == 'uhu' or args.dataset == 'ubc':
        labels=['BN', '3', '4', '5']
    elif args.dataset == 'k16' or args.dataset == 'k19':
        labels=['ADI', 'BACK', 'DEB', 'LYM', 'NORM', 'STR', 'TUM']
    else:
        raise ValueError('invalid dataset')
    
    accuracy = accuracy_score(output['ground_truth_list'], output['prediction_list'])
    f1 = f1_score(output['ground_truth_list'], output['prediction_list'], average='macro', labels=labels)
    accuracy_cancer = accuracy_score(output['ground_truth_cancer_list'], output['prediction_cancer_list'])
    kappa = cohen_kappa_score(output['ground_truth_list'], output['prediction_list'], weights='quadratic', labels=list(set(output['ground_truth_list'])))
    
    c_matrix = confusion_matrix(output['ground_truth_list'], output['prediction_list'])
    disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix)
    return accuracy, accuracy_cancer, f1, kappa, c_matrix, disp


def compute_metrics_2(gt, pred, args):
    if args.dataset in ['colon_1','colon_2', 'gastric']:
        labels=['BN', 'WD', 'MD', 'PD']
    elif args.dataset == 'uhu' or args.dataset == 'ubc':
        labels=['BN', '3', '4', '5']
    elif args.dataset == 'k16' or args.dataset == 'k19':
        labels=['ADI', 'BACK', 'DEB', 'LYM', 'NORM', 'STR', 'TUM']
    else:
        raise ValueError('invalid dataset')
    
    accuracy = accuracy_score(gt, pred)
    f1 = f1_score(gt, pred, average='macro', labels=labels)
    if args.dataset not in ['k16', 'k19']:
        idx_cancer = []
        for i in range(len(gt)):
            if gt[i] != 'BN':
                idx_cancer.append(i)
        gt_cancer = [gt[i] for i in idx_cancer]
        pred_cancer = [pred[i] for i in idx_cancer]
        accuracy_cancer = accuracy_score(gt_cancer, pred_cancer)
        kappa = cohen_kappa_score(gt, pred, weights='quadratic', labels=labels)
        return (accuracy, accuracy_cancer, f1, kappa)
    else:
        rec = recall_score(gt, pred, labels=labels, average="macro")
        prec = precision_score(gt, pred, labels=labels, average="macro")
        return (accuracy, prec, rec, f1)


def post_process(gen_cap, type='colon_1'):
    if gen_cap[0:2] == 'be':
        gen_cap = 'benign.'
    elif gen_cap[0] == ' ':
        gen_cap = 'cancer' + ' '.join(gen_cap.split(' ')[:3])
    return gen_cap

def mapping_type(dataset=None, caption=None):
    if dataset == 'colon_1' or dataset == 'colon_2':
        mapping_dict = {
            'benign.': 'BN',
            'moderately differentiated cancer.': 'MD',
            'poorly differentiated cancer.': 'PD',
            'well differentiated cancer.': 'WD',
        }
    elif dataset == 'uhu' or dataset == 'ubc':
        mapping_dict = {
            'benign.': 'BN',
            'grade 3 cancer.': '3',
            'grade 4 cancer.': '4',
            'grade 5 cancer.': '5',
        }
    elif dataset == 'k16':
        mapping_dict = {
            'adipole tissue.': 'ADI',
            'background tissue.': 'BACK',
            'debris tissue.': 'DEB',
            'lymphocyte tissue.': 'LYM',
            'normal tissue.': 'NORM',
            'stroma tissue.': 'STR',
            'tumor tissue.': 'TUM'
        }
    elif dataset == 'k19':
        mapping_dict = {
            'adipole tissue.': 'ADI',
            'background tissue.': 'BACK',
            'debris tissue.': 'DEB',
            'lymphocyte tissue.': 'LYM',
            'mucus tissue.': 'MUC',
            'muscle tissue.': 'MUS',
            'normal tissue.': 'NORM',
            'stroma tissue.': 'STR',
            'tumor tissue.': 'TUM'
        }
    elif dataset == 'gastric':
        mapping_dict = {
            'benign.': 'BN',
            'tubular well differentiated cancer.': 'WD',
            'tubular moderately differentiated cancer.': 'MD',
            'tubular poorly differentiated cancer.': 'PD',
        }
    else:
        raise ValueError(f'Invalid dataset: {dataset}')
    
    if caption == None:
        return len(mapping_dict)
    
    if caption in mapping_dict: 
        result = mapping_dict[caption]
    else:
        result = 'others'
    return result


def mapping_type_to_num(caption):
    mapping_dict = {
        'benign.': 0,

        'cancer moderately differentiated.': 1,
        'cancer poorly differentiated.': 2,
        'cancer well differentiated.': 3,

        'cancer grade 3.': 4,
        'cancer grade 4.': 5,
        'cancer grade 5.': 6,

        'cancer tubular well differentiated.': 7,
        'cancer tubular moderately differentiated.': 8,
        'cancer tubular poorly differentiated.': 9,


        'tissue adipole.': 10,
        'tissue background.': 11,
        'tissue debris.': 12,
        'tissue lymphocyte.': 13,
        'tissue mucus.': 14,
        'tissue muscle.': 15,
        'tissue normal.': 16,
        'tissue stroma.': 17,
        'tissue tumor.': 18
    }

    if caption in mapping_dict: 
        result = mapping_dict[caption]
    else:
        result = 20
    return result
