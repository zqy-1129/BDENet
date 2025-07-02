import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from misc.metric_tool import ConfuseMatrixMeter
import loss.losses as losses

from misc.logger_tool import Logger, Timer
from torch.optim import lr_scheduler

from build import Builder


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'CosineAnnealing':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6, last_epoch=-1, verbose=False)
    elif args.lr_policy == 'poly':
        scheduler = lr_scheduler.PolynomialLR(optimizer, power=0.9, total_iters=300, last_epoch=-1, verbose=False)
    elif args.lr_policy == 'ReduceLRO':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001,
                                                   threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=False)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        # nn.MultiLabelSoftMarginLoss
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target):
        fl = FocalLoss()
        return fl(logit, target)


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logit = logit.permute(0, 2, 3, 4, 1)
        target = target.permute(0, 2, 3, 1)

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


def weighted_bce(bd_pre, target):
    log_p = bd_pre.permute(0, 1, 3, 4, 2).contiguous().view(1, -1)
    target_t = target.reshape(1,
                              target.shape[0] * target.shape[1] * target.shape[2] * target.shape[3])  # [1, 131072

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss

        return loss


class DiceLoss(nn.Module):
    """
    soft dice loss, 直接使用预测概率而不是使用阈值或将它们转换为二进制mask
    """

    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        # pred不需要转bool变量，如https://github.com/yassouali/pytorch-segmentation/blob/master/utils/losses.py#L44
        # soft dice loss, 直接使用预测概率而不是使用阈值或将它们转换为二进制mask
        pred = torch.sigmoid(predict).view(num, -1)
        targ = target.view(num, -1)

        intersection = (pred * targ).sum()  # 利用预测值与标签相乘当作交集
        union = (pred + targ).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class CDTrainer:
    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders
        self.n_class = args.n_class
        self.edge = args.edge
        self.muti_scale = args.muti_scale

        build = Builder(args)
        self.net_G = build.build_model()
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr
        self.optimizer_G = optim.Adam(self.net_G.to(self.device).parameters(), lr=self.lr, betas=(0.9, 0.999))

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)  # A 时相
        self.running_metric_B = ConfuseMatrixMeter(n_class=self.n_class)  # B 时相
        self.running_metric_change = ConfuseMatrixMeter(n_class=2)  # 变化

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.epoch_acc_B = 0
        self.epoch_acc_CH = 0
        self.epoch_Acc = 0
        self.cur_score = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.G_loss_total = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.G_pred = None
        self.G_pred_B = None
        self.G_pred_CH = None
        self.G_bdpred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.G_bdloss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir

        # define the loss functions
        if args.loss == 'ce':
            self._pxl_loss = SegmentationLosses()
            # self._pxl_loss = FocalLoss
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
            # self._pxl_loss = nn.BCELoss
        elif args.loss == 'dice':
            self._pxl_loss = DiceLoss()

        else:
            raise NotImplemented(args.loss)

        # self.boundaryloss = BondaryLoss()

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        if self.edge:
            if self.muti_scale:
                # A Seg
                target = self.batch[4].to(self.device).detach()
                G_pred = self.G_pred.detach()
                G_pred = torch.argmax(G_pred, dim=1)

                # B Seg
                target_B = self.batch[9].to(self.device).detach()
                G_pred_B = self.G_pred_B.detach()
                G_pred_B = torch.argmax(G_pred_B, dim=1)

                # Change
                target_CH = self.batch[10].to(self.device).detach()
                G_pred_CH = self.G_pred_CH.detach()
                G_pred_CH = torch.argmax(G_pred_CH, dim=1)
            else:
                # A Seg
                target = self.batch[3].to(self.device).detach()
                G_pred = self.G_pred.detach()
                G_pred = torch.argmax(G_pred, dim=1)

                # B Seg
                target_B = self.batch[7].to(self.device).detach()
                G_pred_B = self.G_pred_B.detach()
                G_pred_B = torch.argmax(G_pred_B, dim=1)

                # Change
                target_CH = self.batch[8].to(self.device).detach()
                G_pred_CH = self.G_pred_CH.detach()
                G_pred_CH = torch.argmax(G_pred_CH, dim=1)

        else:
            if self.muti_scale:
                # A Seg
                target = self.batch[2].to(self.device).detach()
                G_pred = self.G_pred.detach()
                G_pred = torch.argmax(G_pred, dim=1)

                # B Seg
                target_B = self.batch[5].to(self.device).detach()
                G_pred_B = self.G_pred_B.detach()
                G_pred_B = torch.argmax(G_pred_B, dim=1)

                # Change
                target_CH = self.batch[6].to(self.device).detach()
                G_pred_CH = self.G_pred_CH.detach()
                G_pred_CH = torch.argmax(G_pred_CH, dim=1)
            else:
                # A Seg
                target = self.batch[1].to(self.device).detach()
                G_pred = self.G_pred.detach()
                G_pred = torch.argmax(G_pred, dim=1)

                # B Seg
                target_B = self.batch[3].to(self.device).detach()
                G_pred_B = self.G_pred_B.detach()
                G_pred_B = torch.argmax(G_pred_B, dim=1)

                # Change
                target_CH = self.batch[4].to(self.device).detach()
                G_pred_CH = self.G_pred_CH.detach()
                G_pred_CH = torch.argmax(G_pred_CH, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        current_score_B = self.running_metric_B.update_cm(pr=G_pred_B.cpu().numpy(), gt=target_B.cpu().numpy())
        current_score_CH = self.running_metric_change.update_cm(pr=G_pred_CH.cpu().numpy(), gt=target_CH.cpu().numpy())

        self.cur_score = current_score + current_score_B + current_score_CH

        return self.cur_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, G_loss_B: %.5f, G_loss_CH: %.5f , running_mf1: %.5f\n' % \
                      (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m,
                       imps * self.batch_size, est,
                       self.G_loss.item(), self.G_loss_B.item(), self.G_loss_CH.item(), running_acc)
            self.logger.write(message)

    def _collect_epoch_states(self, epoch):
        scores = self.running_metric.get_scores()
        scores_B = self.running_metric_B.get_scores()
        scores_CH = self.running_metric_change.get_scores()
        self.epoch_acc = scores['miou']
        self.epoch_acc_B = scores_B['miou']
        self.epoch_acc_CH = scores_CH['miou']
        self.epoch_Acc = self.epoch_acc + self.epoch_acc_B + self.epoch_acc_CH
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
                          (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.epoch_Acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        message += '\n'
        for k, v in scores_B.items():
            message += '%s: %.5f ' % (k, v)
        message += '\n'
        for k, v in scores_CH.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message + '\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
                          % (self.epoch_Acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_Acc > self.best_val_acc:
            self.best_val_acc = self.epoch_Acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_Acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_Acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()
        self.running_metric_B.clear()
        self.running_metric_change.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        if self.edge:
            if self.muti_scale:
                img_A_1 = batch[0].to(self.device)
                img_A_2 = batch[1].to(self.device)
                img_B_1 = batch[5].to(self.device)
                img_B_2 = batch[6].to(self.device)
                edge_A = batch[2].to(self.device)
                edge_B = batch[7].to(self.device)
                edge_M_A = batch[3].to(self.device)
                edge_M_B = batch[8].to(self.device)
                self.G_pred, self.G_pred_B, self.G_pred_CH = self.net_G([img_A_1, img_A_2], [edge_A, edge_M_A],
                                                                        [img_B_1, img_B_2], [edge_B, edge_M_B])
            else:
                img_A = batch[0].to(self.device)
                img_B = batch[4].to(self.device)
                edge_A = batch[1].to(self.device)
                edge_B = batch[5].to(self.device)
                edge_M_A = batch[2].to(self.device)
                edge_M_B = batch[6].to(self.device)
                self.G_pred, self.G_pred_B, self.G_pred_CH = self.net_G(img_A, [edge_A, edge_M_A],
                                                                        img_B, [edge_B, edge_M_B])
        else:
            if self.muti_scale:
                img_A_1 = batch[0].to(self.device)
                img_A_2 = batch[1].to(self.device)
                img_B_1 = batch[3].to(self.device)
                img_B_2 = batch[4].to(self.device)
                self.G_pred, self.G_pred_B, self.G_pred_CH = self.net_G([img_A_1, img_A_2], [img_B_1, img_B_2])
            else:
                img_A = batch[0].to(self.device)
                img_B = batch[2].to(self.device)
                self.G_pred, self.G_pred_B, self.G_pred_CH = self.net_G(img_A, img_B)

    def _backward_G(self):
        if self.edge:
            if self.muti_scale:
                # A Seg
                gt_A = self.batch[4].to(self.device).long()
                # B Seg
                gt_B = self.batch[9].to(self.device).long()
                # Change
                gt_CH = self.batch[10].to(self.device).long()
            else:
                # A Seg
                gt_A = self.batch[3].to(self.device).long()
                # B Seg
                gt_B = self.batch[7].to(self.device).long()
                # Change
                gt_CH = self.batch[8].to(self.device).long()
        else:
            if self.muti_scale:
                # A Seg
                gt_A = self.batch[2].to(self.device).long()
                # B Seg
                gt_B = self.batch[5].to(self.device).long()
                # Change
                gt_CH = self.batch[6].to(self.device).long()
            else:
                # A Seg
                gt_A = self.batch[1].to(self.device).long()
                # B Seg
                gt_B = self.batch[3].to(self.device).long()
                # Change
                gt_CH = self.batch[4].to(self.device).long()

        self.G_loss = self._pxl_loss.FocalLoss(self.G_pred, gt_A.squeeze(dim=1))
        self.G_loss_B = self._pxl_loss.FocalLoss(self.G_pred_B, gt_B.squeeze(dim=1))
        self.G_loss_CH = self._pxl_loss.FocalLoss(self.G_pred_CH, gt_CH.squeeze(dim=1))

        self.G_loss_total = self.G_loss_CH + self.G_loss + self.G_loss_B
        self.G_loss_total.backward()

    def train_models(self):
        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()

            self._collect_epoch_states(epoch=self.epoch_id)
            self._update_training_acc_curve()
            self._update_lr_schedulers()

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states(epoch=self.epoch_id)

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()
