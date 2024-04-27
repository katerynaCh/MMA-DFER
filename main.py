from models import models_vit
import sys
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.Generate_Model import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
from dataloader.video_dataloader import train_data_loader, test_data_loader
from sklearn.metrics import confusion_matrix
import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random

seed = 1
random.seed(seed)  
np.random.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int)

    parser.add_argument('--exper-name', type=str)
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=224)

    args = parser.parse_args()
    return args

def main(set, args):
    
    data_set = set+1
    
    if args.dataset == "DFEW":
        print("*********** DFEW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.txt'
        log_curve_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.png'
        log_confusion_matrix_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/' + 'cn.png'
        checkpoint_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/'+'checkpoint/'+'model.pth'
        train_annotation_file_path = "./annotation/DFEW_set_"+str(data_set)+"_train.txt"
        test_annotation_file_path = "./annotation/DFEW_set_"+str(data_set)+"_test.txt"
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/checkpoint/')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/models')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/AudioMAE')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/dataloader')

        for filename in ['main.py', 'train_DFEW.sh', 'train_MAFW.sh', 'models/Generate_Model.py', 'models/Temporal_Model.py', 'dataloader/video_dataloader.py', 'dataloader/video_transform.py', 'models/models_vit.py', 'AudioMAE/audio_models_vit.py']:
            shutil.copyfile(filename, './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/'+filename)

    elif args.dataset == "MAFW":
        print("*********** MAFW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.txt'
        log_curve_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.png'
        log_confusion_matrix_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/' + 'cn.png'
        checkpoint_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/'+'checkpoint/'+'model.pth'

        train_annotation_file_path = "./annotation/MAFW_set_"+str(data_set)+"_train_faces.txt"
        test_annotation_file_path = "./annotation/MAFW_set_"+str(data_set)+"_test_faces.txt"
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/checkpoint/')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/models')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/AudioMAE')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/dataloader')

        for filename in ['main.py', 'train_DFEW.sh', 'train_MAFW.sh', 'models/Generate_Model.py', 'models/Temporal_Model.py', 'dataloader/video_dataloader.py', 'dataloader/video_transform.py', 'models/models_vit.py', 'AudioMAE/audio_models_vit.py']:
            shutil.copyfile(filename, './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/'+filename)

    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print('The training name: ' + time_str)
       
    model = GenerateModel(args=args)
  
    # only open learnable part
    for name, param in model.named_parameters():
        param.requires_grad = True #False

    for name, param in model.named_parameters():
        if "image_encoder" in name:
            param.requires_grad = False 
        if "audio_model" in name:
            param.requires_grad = False

        if "our_classifier" in name:
            param.requires_grad = True
        if "positional_embedding" in name:
            param.requires_grad = True
        if "learnable_prompts" in name:
            param.requires_grad = True
        if "pos_embed" in name:
            param.requires_grad = True
        if "audio_proj" in name:
            param.requires_grad = True
        if "temporal" in name:
            param.requires_grad = True
        if "gate" in name:
            param.requires_grad = True
        if "context_att" in name:
            param.requires_grad = True
        if "learnable_q" in name:
            param.requires_grad = True
        if "audio_att" in name:
            param.requires_grad = True
        if "norm_xt" in name:
            param.requires_grad = True
        if "norm_xt_2" in name:
            param.requires_grad = True
        if "norm_qs" in name:
            param.requires_grad = True

    model_parameters = model.parameters()
    model = torch.nn.DataParallel(model).cuda()

    # print params   
    print('************************')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('************************')
    
    with open(log_txt_path, 'a') as f:
        for k, v in vars(args).items():
            f.write(str(k) + '=' + str(v) + '\n')
    
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()
    
    # define optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) 
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(list_file=train_annotation_file_path,
                                   num_segments=16,
                                   duration=1,
                                   image_size=args.img_size,
                                   args=args)

    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=args.img_size)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    for epoch in range(0, args.epochs):

        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate_0 = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            print(inf)
            f.write('Current learning rate: ' + str(current_learning_rate_0) + '\n')
            print('Current learning rate: ', current_learning_rate_0)        
            
        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)

        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, criterion, args, log_txt_path)
        
        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best,
                        checkpoint_path)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.2f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: ' + str(epoch_time) + 's' + '\n')

    last_uar, last_war = computer_uar_war(val_loader, model, checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set, args.class_names)
  
    return last_uar, last_war


def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch),
                             log_txt_path=log_txt_path)

    # switch to train mode
    model.train()

    for i, (images, target, audio) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()
        audio = audio.cuda()
        # compute output
        output = model(images, audio)        
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ',
                             log_txt_path=log_txt_path)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target, audio) in enumerate(val_loader):
            
            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()
            # compute output
            output = model(images, audio)        
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, checkpoint_path):
    torch.save(state, checkpoint_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

def computer_uar_war(val_loader, model, checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set, class_names):
    
    pre_trained_dict = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(pre_trained_dict)
    
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, (images, target, audio) in enumerate(tqdm.tqdm(val_loader)):
            
            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()
            output = model(images, audio)        

            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

    war = 100. * correct / len(val_loader.dataset)
    
    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()
        
    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))

    if args.dataset == "DFEW":
        title_ = "Confusion Matrix on DFEW fold "+str(data_set)
    elif args.dataset == "MAFW":
        title_ = "Confusion Matrix on MAFW fold "+str(data_set)

    plot_confusion_matrix(normalized_cm, classes=class_names, normalize=True, title=title_)
    plt.savefig(os.path.join(log_confusion_matrix_path))
    plt.close()
    
    with open(log_txt_path, 'a') as f:
        f.write('************************' + '\n')
        f.write(checkpoint_path)
        f.write("Confusion Matrix Diag:" + '\n')
        f.write(str(list_diag.tolist()) + '\n')
        f.write('UAR: {:.2f}'.format(uar) + '\n')        
        f.write('WAR: {:.2f}'.format(war) + '\n')
        f.write('************************' + '\n')
    
    return uar, war


if __name__ == '__main__':
    args = parse_args() 
    UAR = 0.0
    WAR = 0.0
    now = datetime.datetime.now()
    time_str = now.strftime("%y%m%d%H%M")
    time_str = time_str + args.exper_name

    print('************************')
    for k, v in vars(args).items():
        print(k,'=',v)
    print('************************')

    if args.dataset == "DFEW":
        args.number_class = 7
        args.class_names = [
	'happiness.',
	'sadness.',
	'neutral.',
	'anger.',
	'surprise.',
	'disgust.',
	'fear.'
        ]

        all_fold = 5
    elif args.dataset == "MAFW":
        all_fold = 5
        args.number_class = 11
        args.class_names = ["1", '2', '3', '4','5', '6', '7', '8', '9', '10', '11']

    for set in range(all_fold):
        uar, war = main(set, args)
        UAR += float(uar)
        WAR += float(war)
        
    print('********* Final Results *********')   
    print("UAR: %0.2f" % (UAR/all_fold))
    print("WAR: %0.2f" % (WAR/all_fold))
    print('*********************************')
