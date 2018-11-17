import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import datetime
from tensorboardX import SummaryWriter
from utils.eval import *
from torch.utils.checkpoint import checkpoint_sequential


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIMEPOINT = datetime.datetime.now().strftime('%Y_%M_%D')

def train(model, data_loader, optimizer, criterion,
          batch_size=10, epochs=10,eval_func = IOU, eval_step =10, log_params = False, device='cpu', save_model = True):

    writer = SummaryWriter()
    model.to(device)
    model.train()

    assert device in ['cpu','cuda'],'device should be cpu or cuda'
    if device=='cuda':
        log_device = 'cpu'
    else:
        log_device = 'cuda'

    niter = 0
    eval_results = []
    for epoch in range(epochs):
        for batch_idx, sample in enumerate(data_loader):
            optimizer.zero_grad()
            input, label = sample['image'].to(device),sample['label'].to(device)
            input.requires_grad_(requires_grad=False)
            output = model(input)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            niter += batch_size

            writer.add_histogram('loss', loss.clone().to('cpu').data.numpy(), niter)

            if (batch_idx+1)*batch_size % eval_step == 0:
                #
                eval = eval_func(output, label)
                info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Eval: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(input), len(data_loader.dataset),
                           100. * (batch_idx+1) / len(data_loader), loss.item(),eval)
                print(info)
                writer.add_text('Text',info)
                del info
                writer.add_histogram('eval', eval, niter)
                eval_results.append(float(eval))

    # print('Model Train Finished,Eval:{:.6f}'.format(eval_results/len(eval_results)))

    if  save_model:
        if not os.path.exists('./save'):
            os.makedirs('./save')
        path = os.path.join('./save',file)
        torch.save(model,path)