import numpy as np


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x + 1e-6) ** -1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


# def cm2score(confusion_matrix):
#     hist = confusion_matrix
#     n_class = hist.shape[0]
#     tp = np.diag(hist)
#     sum_a1 = hist.sum(axis=1)
#     sum_a0 = hist.sum(axis=0)
#     acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
#     recall = tp / (sum_a1 + np.finfo(np.float32).eps)
#     precision = tp / (sum_a0 + np.finfo(np.float32).eps)
#     F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
#     mean_F1 = np.nanmean(F1)
#     iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
#     mean_iu = np.nanmean(iu)
#
#     cls_iou = dict(zip(['iou_' + str(i) for i in range(n_class)], iu))
#     cls_precision = dict(zip(['precision_' + str(i) for i in range(n_class)], precision))
#     cls_recall = dict(zip(['recall_' + str(i) for i in range(n_class)], recall))
#     cls_F1 = dict(zip(['F1_' + str(i) for i in range(n_class)], F1))
#
#     score_dict = {'acc': acc, 'miou': mean_iu, 'mf1': mean_F1}
#     score_dict.update(cls_iou)
#     score_dict.update(cls_F1)
#     score_dict.update(cls_precision)
#     score_dict.update(cls_recall)
#     return score_dict

def cm2Kappa(confusion_matrix):
    """
    Compute Cohen's Kappa score from the confusion matrix.

    :param confusion_matrix: <np.ndarray> Confusion matrix
    :return: <float> Cohen's Kappa score
    """
    hist = confusion_matrix
    n = hist.sum()  # Total samples
    sum_po = np.diag(hist).sum()  # Observed agreement
    sum_pe = (hist.sum(axis=0) * hist.sum(axis=1)).sum() / (n ** 2)  # Expected agreement

    kappa = (sum_po / n - sum_pe) / (1 - sum_pe + np.finfo(np.float32).eps)
    return kappa

def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    # 计算 accuracy, recall, precision, F1
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)

    # 计算 IoU
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)




    # 计算Kappa系数 (SeK)
    # Kappa = cm2Kappa(hist)

    # ------------------------- Kappa系数计算 -------------------------
    # 计算 P_o (观察到的一致性)
    # P_o = np.sum(tp) / np.sum(hist)
    #
    # # 计算 P_e (期望一致性)
    row_sum = np.sum(hist, axis=1)
    col_sum = np.sum(hist, axis=0)
    P_e = np.sum(row_sum * col_sum) / (np.sum(hist) ** 2)
    #
    # # 计算 Kappa 系数
    # kappa = (P_o - P_e) / (1 - P_e)

    kappa = cm2Kappa(hist)
    # ------------------------- SeK计算 -------------------------
    η = P_e  # 期望一致性
    ρ = kappa  # Kappa系数
    SeK = (ρ - η) / (1 - η) * np.exp(mean_iu - 1)



    # # ------------------------- Kc计算 -------------------------
    # total_sum = np.sum(hist)
    # row_sum = np.sum(hist, axis=1)
    # col_sum = np.sum(hist, axis=0)
    #
    # K_c = np.sum(np.outer(row_sum, col_sum)) / (total_sum ** 2)
    #
    # # ------------------------- 观察到一致性 P_A -------------------------
    # P_A = np.sum(tp) / total_sum
    #
    # # ------------------------- Kappa系数计算 -------------------------
    # kappa = (P_A - K_c) / (1 - K_c) if (1 - K_c) != 0 else 0  # 防止除零错误
    #
    # # ------------------------- SeK计算 -------------------------
    # η = K_c  # 期望一致性
    # ρ = kappa  # Kappa系数
    # SeK = (ρ - η) / (1 - η) * np.exp(mean_iu - 1) if (1 - η) != 0 else 0  # 防止除零错误



    # # # 计算Kappa系数 (SeK)
    # ρ = np.sum(tp) / (np.sum(sum_a1) + np.sum(sum_a0) - np.sum(tp))
    # η = (np.sum(hist) + np.sum(hist.T)) / (np.sum(sum_a1) + np.sum(sum_a0) - np.sum(tp))
    # SeK = np.exp(mean_iu - 1) * (ρ - η) / (1 - η)
    #
    # # Kappa = (ρ - η) / (1 - η)
    # # SeK = Kappa
    #




    # # 计算Score
    Score = 0.3 * mean_iu + 0.7 * SeK

    # 每个类别的IoU、F1、precision、recall
    cls_iou = dict(zip(['iou_' + str(i) for i in range(n_class)], iu))
    cls_precision = dict(zip(['precision_' + str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_' + str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_' + str(i) for i in range(n_class)], F1))

    # 返回字典，包括所有指标
    score_dict = {
        'acc': acc,
        'miou': mean_iu,
        'mf1': mean_F1,
        'SeK': SeK,
        'Score': Score,
        'Kappa': kappa
    }
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)

    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    def __fast_hist(label_gt, label_pred):
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']


###################      SeK & Score       ###################
def calculate_kappa(confusion_matrix):
    """计算Kappa系数"""
    n_class = confusion_matrix.shape[0]
    P_ii = np.diag(confusion_matrix)
    P_ij = confusion_matrix.sum(axis=1)
    P_ji = confusion_matrix.sum(axis=0)
    total = confusion_matrix.sum()

    ρ = np.sum(P_ii) / (np.sum(P_ij) + np.sum(P_ji) - np.sum(P_ii))
    η = (np.sum(confusion_matrix) + np.sum(confusion_matrix.T)) / (np.sum(P_ij) + np.sum(P_ji) - np.sum(P_ii))

    # Kappa coefficient (SeK)
    SeK = (ρ - η) / (1 - η) * np.exp(np.nanmean(cm2score(confusion_matrix)['miou']) - 1)
    return SeK


def calculate_score(confusion_matrix):
    """计算最终的Score"""
    mIoU = cm2score(confusion_matrix)['miou']
    SeK = calculate_kappa(confusion_matrix)
    Score = 0.3 * mIoU + 0.7 * SeK
    return Score


def get_SeK_and_score(num_classes, label_gts, label_preds):
    """返回SeK和Score"""
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    SeK = calculate_kappa(confusion_matrix)
    Score = calculate_score(confusion_matrix)
    return SeK, Score
