from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from ignite.metrics import EpochMetric
import numpy as np



def multi_class_roc_wrapper(n_classes=3):
    def multi_class_roc(y_test, y_score):
        y_test = y_test.detach().cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        y_test = label_binarize(y_test, classes=[i for i in range(n_classes)])
        if n_classes == 2:
            y_test = np.concatenate([1-y_test, y_test], axis=1)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], t = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        return roc_auc
    return multi_class_roc


class ROC_AUC_Multi(EpochMetric):
    def __init__(self, n_class, output_transform=lambda x: x, check_compute_fn: bool = False):
        roc_func = multi_class_roc_wrapper(n_classes=n_class)
        super(ROC_AUC_Multi, self).__init__(
            roc_func, output_transform=output_transform, check_compute_fn=check_compute_fn
        )


if __name__ == '__main__':
    import numpy as np
    import torch
    y_test = np.array([1, 1, 0, 1, 0, 0, 0, 1])
    y_score = np.random.random((8, 2))
    y_score = np.exp(y_score) / np.exp(y_score).sum(axis=1)[:,  np.newaxis]

    roc_auc_func = multi_class_roc_wrapper(n_classes=2)
    roc_auc = roc_auc_func(torch.FloatTensor(y_test), torch.LongTensor(y_score))
    print(roc_auc)