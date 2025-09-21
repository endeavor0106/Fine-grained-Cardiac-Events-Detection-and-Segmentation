import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error


# F1 Score
def calculate_f1(y_true, y_pred):
    f1_scores = f1_score(y_true, y_pred, average='macro')
    return f1_scores


def calculate_f1_every_class(y_true, y_pred):
    p_class, r_class, f1_class, support_micro = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                                                labels=[0, 1, 2, 3], average=None)
    return f1_class


# RMSE
def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def calculate_interval_error(y_true, y_pred):

    # interval
    true_p_wave = np.where(y_true == 1)[0]
    true_qrs_wave = np.where(y_true == 2)[0]
    true_t_wave = np.where(y_true == 3)[0]

    pred_p_wave = np.where(y_pred == 1)[0]
    pred_qrs_wave = np.where(y_pred == 2)[0]
    pred_t_wave = np.where(y_pred == 3)[0]

    p_wave_error = abs(len(true_p_wave) - len(pred_p_wave))
    qrs_wave_error = abs(len(true_qrs_wave) - len(pred_qrs_wave))
    t_wave_error = abs(len(true_t_wave) - len(pred_t_wave))

    # point
    true_p_onset = true_p_wave[0]
    true_p_offset = true_p_wave[-1]
    true_qrs_onset = true_qrs_wave[0]
    true_qrs_offset = true_qrs_wave[-1]
    true_t_onset = true_t_wave[0]
    true_t_offset = true_t_wave[-1]

    true_pr = true_qrs_onset - true_p_onset
    true_qt = true_t_offset - true_qrs_onset

    if len(pred_p_wave) == 0 or len(pred_qrs_wave) == 0 or len(pred_t_wave) == 0:
        pr_error = true_pr
        qt_error = true_qt

    else:
        pred_p_onset = pred_p_wave[0]
        pred_p_offset = pred_p_wave[-1]

        pred_qrs_onset = pred_qrs_wave[0]
        pred_qrs_offset = pred_qrs_wave[-1]

        pred_t_onset = pred_t_wave[0]
        pred_t_offset = pred_t_wave[-1]

        pred_pr = pred_qrs_onset - pred_p_onset
        pred_qt = pred_t_offset - pred_qrs_onset

        pr_error = abs(true_pr - pred_pr)
        qt_error = abs(true_qt - pred_qt)

    return len(true_p_wave), len(pred_p_wave), len(true_qrs_wave), len(pred_qrs_wave), len(true_t_wave), len(pred_t_wave), p_wave_error, qrs_wave_error, t_wave_error, pr_error, qt_error


def eval_functions_all(y_true, y_pred, save_dir='', info='', uid='', visual_flag=False, ECG=''):
    batch_size = y_true.shape[0]
    f1_scores = []
    avg_error_list = []
    p_wave_error_list = []
    qrs_wave_error_list = []
    t_wave_error_list = []
    pr_error_list = []
    qt_error_list = []

    for i in range(batch_size):
        f1 = calculate_f1(y_true[i], y_pred[i])
        f1_class = calculate_f1_every_class(y_true[i], y_pred[i])

        true_p_wave, pred_p_wave, true_qrs_wave, pred_qrs_wave, true_t_wave, pred_t_wave, p_wave_error, qrs_wave_error, t_wave_error, pr_error, qt_error = calculate_interval_error(y_true[i], y_pred[i])
        avg_error = (p_wave_error + qrs_wave_error + t_wave_error + pr_error + qt_error) / 5

        if save_dir == '':
            f1_scores.append(f1)
            avg_error_list.append(avg_error)
            p_wave_error_list.append(p_wave_error)
            qrs_wave_error_list.append(qrs_wave_error)
            t_wave_error_list.append(t_wave_error)
            pr_error_list.append(pr_error)
            qt_error_list.append(qt_error)

        else:
            f1 = np.round(float(f1), 4)
            f1_p = np.round(float(f1_class[1]), 4)
            f1_qrs = np.round(float(f1_class[2]), 4)
            f1_t = np.round(float(f1_class[3]), 4)

            # ---------------info------------------------
            if info['diagnosis'][i] == 'NSR':
                sort = 0
            if 'I-AVB' in info['diagnosis'][i]:
                sort = 1
            elif 'ST-T' in info['diagnosis'][i]:
                sort = 5
            elif 'LBBB' in info['diagnosis'][i]:
                sort = 2
            elif 'RBBB' in info['diagnosis'][i]:
                sort = 3
            elif 'T wave' in info['diagnosis'][i]:
                sort = 4
            elif 'ST' in info['diagnosis'][i]:
                sort = 5
            else:
                sort = 100

            f1_scores.append(f1)
            avg_error_list.append(avg_error)
            p_wave_error_list.append(p_wave_error)
            qrs_wave_error_list.append(qrs_wave_error)
            t_wave_error_list.append(t_wave_error)
            pr_error_list.append(pr_error)
            qt_error_list.append(qt_error)

            if visual_flag:
                # visual
                fig = plt.figure()
                plt.plot(y_true[i], label='true')
                plt.plot(y_pred[i], label='pred')
                plt.plot(ECG[i], label='ECG')
                plt.title(str(info['num'][i]) + '-' + str(info['slice_num'][i].item()) + '-' + str(f1) + '-' + str(f1_p) + '-' + str(f1_qrs) + '-' + str(f1_t) + '-' + '-sort ' + str(sort) + '-' + str(true_p_wave) + '-' + str(pred_p_wave) + '-' + str(true_qrs_wave) + '-' + str(pred_qrs_wave) + '-' + str(true_t_wave) + '-' + str(pred_t_wave), fontsize=12)
                # plt.plot()
                plt.legend()
                plt.savefig(os.path.join(save_dir, str(info['num'][i]) + '-' + str(info['slice_num'][i].item()) + '-' + str(f1) + '-' + str(f1_p) + '-' + str(f1_qrs) + '-' + str(f1_t) + '-' + str(sort) + '-' + str(true_p_wave) + '-' + str(pred_p_wave) + '-' + str(true_qrs_wave) + '-' + str(pred_qrs_wave) + '-' + str(true_t_wave) + '-' + str(pred_t_wave) + '-' + str(info['diagnosis'][i]) + '.png'))
                # plt.show()
                plt.close(fig)

            disease_path = os.path.join(save_dir, uid + ".txt")

            if os.path.exists(disease_path):
                with open(disease_path, "a") as file:
                    file.write("\n" + str(info['num'][i]) + '-' + str(info['slice_num'][i].item()) + '-' + str(f1) + '-' + str(f1_p) + '-' + str(f1_qrs) + '-' + str(f1_t) + '-' + str(sort) + '-' + str(true_p_wave) + '-' + str(pred_p_wave) + '-' + str(true_qrs_wave) + '-' + str(pred_qrs_wave) + '-' + str(true_t_wave) + '-' + str(pred_t_wave) + '-' + str(pr_error) + '-' + str(qt_error) + '-' + str(info['diagnosis'][i]))
            else:
                with open(disease_path, "w") as file:
                    file.write(str(info['num'][i]) + '-' + str(info['slice_num'][i].item()) + '-' + str(f1) + '-' + str(f1_p) + '-' + str(f1_qrs) + '-' + str(f1_t) + '-' + str(sort) + '-' + str(true_p_wave) + '-' + str(pred_p_wave) + '-' + str(true_qrs_wave) + '-' + str(pred_qrs_wave) + '-' + str(true_t_wave) + '-' + str(pred_t_wave) + '-' + str(pr_error) + '-' + str(qt_error) + '-' + str(info['diagnosis'][i]))

    return (np.mean(f1_scores), np.mean(avg_error_list), np.mean(p_wave_error_list),
            np.mean(qrs_wave_error_list), np.mean(t_wave_error_list), np.mean(pr_error_list), np.mean(qt_error_list))


def eval_functions_F1(y_true, y_pred):
    batch_size = y_true.shape[0]
    f1_scores = []
    for i in range(batch_size):
        f1 = calculate_f1(y_true[i], y_pred[i])
        f1_scores.append(f1)

    return np.mean(f1_scores)

def eval_functions_RMSE(y_true, y_pred):
    batch_size = y_true.shape[0]
    rmse_scores = []
    for i in range(batch_size):
        rmse = calculate_rmse(y_true[i], y_pred[i])
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

def eval_functions(y_true, y_pred):
    batch_size = y_true.shape[0]
    f1_scores = []
    rmse_scores = []
    for i in range(batch_size):
        f1 = calculate_f1(y_true[i], y_pred[i])
        f1_scores.append(f1)
        rmse = calculate_rmse(y_true[i], y_pred[i])
        rmse_scores.append(rmse)

    return np.mean(f1_scores), np.mean(rmse_scores)
