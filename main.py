import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
from torchsummary import summary


def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window

    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w if 'LSTM' in args.model or 'EPAADNet' in args.model or 'TimesNet' in args.model or 'DCdetector' in args.model else w.view(-1))

    return torch.stack(windows)


def load_dataset(dataset):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        if dataset == 'UCR': file = '136_' + file
        # if dataset == 'MBA': file = file + '_1'
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file + '_1'
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    if args.less:
        loader[0] = cut_array(0.2, loader[0])
    print('train', loader[0].shape, loader[1].shape, loader[2].shape)

    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1;
        accuracy_list = []

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]

    if 'LSTM' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch  # if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        num_batches = len(data) // bs + 1
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2).to(device)
                elem = window[-1, :, :].view(1, local_bs, feats)
                # print('elem', elem.shape)
                z = model(window)
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple):
                    z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            loss = torch.zeros(0, feats).to(device)
            z1 = torch.zeros(0, feats).to(device)
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2).to(device)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window)
                # if isinstance(z, tuple): z = z[1]
                lossa = l(z, elem)[0]
                loss = torch.cat((loss, lossa), dim=0)
                za = z[0]
                z1 = torch.cat((z1, za), dim=0)
            # print('z1', z1.shape)
            return loss.detach().cpu().numpy(), z1.detach().cpu().numpy()  # numpy格式的数据要转移到cpu上面进行操作

    elif 'EPAADNet' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data);
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch  # if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        num_batches = len(data) // bs + 1
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                print('d', d.shape)
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple):
                    z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            loss = torch.zeros(0, feats)
            z1 = torch.zeros(0, feats)
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                # if isinstance(z, tuple): z = z[1]
                lossa = l(z, elem)[0]
                loss = torch.cat((loss, lossa), dim=0)
                za = z[0]
                z1 = torch.cat((z1, za), dim=0)
            # print('z1', z1.shape)
            return loss.detach().cpu().numpy(), z1.detach().cpu().numpy()  # numpy格式的数据要转移到cpu上面进行操作

    elif 'TimesNet' in model.name or 'DCdetector' in model.name:
        # Autoencoder-style: input window (N,T,C) → reconstruct window (N,T,C)
        l = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch
        dataloader = DataLoader(dataset, batch_size=bs)
        feats = dataO.shape[1]
        l1s = []
        if training:
            for d, _ in dataloader:
                window = d                                  # (bs, T, C)
                z = model(window)                           # (bs, T, C)
                l1 = l(z, window)                           # (bs, T, C)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            loss_list = []
            z_list = []
            for d, _ in dataloader:
                window = d
                z = model(window)
                l1 = l(z, window)                                  # (bs, T, C)
                # Average over T dim for per-feature anomaly scores (N, C)
                l1_feat = torch.mean(l1, dim=1)                    # (bs, C)
                loss_list.append(l1_feat)
                z_list.append(z)
            loss_out = torch.cat(loss_list, dim=0)                 # (N, C)
            z_out = torch.cat(z_list, dim=0)                       # (N, T, C)
            return loss_out.detach().cpu().numpy(), z_out.detach().cpu().numpy()

    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == '__main__':
    train_loader, test_loader, labels = load_dataset(args.dataset)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("GPU 可用！")
    else:
        print('不可用')

    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])
    # model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, 5)

    #model = model.to(device)

    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['Attention', 'TranAD', 'EPAADNet', 'TimesNet', 'DCdetector'] or 'LSTM' in model.name:
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

    ### Training phase
    if not args.test:
        f1_list = []
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = 5
        e = epoch + 1
        start = time()
        # print('model', model)
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            model.train()
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)

            # 在验证集上评估模型
            model.eval()
            with torch.no_grad():
                lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
                loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
                lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
                labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
                # labelsFinal = labels
                result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
                f1_list.extend([result['f1']])
                # values_list = [result['f1'], result['precision'], result['recall']]
                print('f1', result['f1'], ' ', 'pre', result['precision'], ' ', 'recall', result['recall'])
                maxf1 = 0
                for i in range(len(f1_list)):
                    if f1_list[i] > maxf1:
                        maxf1 = f1_list[i]
                        print(i, end=' ')
                print('maxf1', maxf1)

            accuracy_list.append((lossT, lr))
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list)
    # plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase
    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    start = time()
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    ### Scores
    # df = pd.DataFrame()
    df_list = []
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    pprint(result)

    # Point-wise F1 (无 PA)
    pred_pw = (lossFinal > result['threshold']).astype(float)
    TP_pw = np.sum(pred_pw * labelsFinal)
    FP_pw = np.sum(pred_pw * (1 - labelsFinal))
    FN_pw = np.sum((1 - pred_pw) * labelsFinal)
    p_pw = TP_pw / (TP_pw + FP_pw + 1e-8)
    r_pw = TP_pw / (TP_pw + FN_pw + 1e-8)
    f1_pw = 2 * p_pw * r_pw / (p_pw + r_pw + 1e-8)
    print(f'{color.BLUE}F1(PA)={result["f1"]:.4f}  F1(PW)={f1_pw:.4f}  '
          f'Precision={p_pw:.4f}  Recall={r_pw:.4f}{color.ENDC}')

    # beep(4)
    print(color.BOLD + 'Test time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
