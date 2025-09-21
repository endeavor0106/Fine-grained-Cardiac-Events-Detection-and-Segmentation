import os
import argparse
import copy
import datetime
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from tqdm import tqdm

from modules.data_loader import MyData
from modules.evaluation import eval_functions, eval_functions_all
from modules.model_selector import select_model

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"set seed: {seed}")


def prepare_dataloaders(data_dir, batch_size, data_set_len):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = MyData(train_dir, sort='train', length=data_set_len)
    val_dataset = MyData(val_dir, sort='val', length=data_set_len)
    test_dataset = MyData(test_dir, sort='test', length=data_set_len)

    point_nums = train_dataset.get_point_nums()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, point_nums


def train_epoch(model, dataloader, loss_fn, optimizer, device, PCA_flag=False):
    model.train()
    total_loss = 0
    all_f1_scores = []
    all_RMSE = []

    for i, batch_data in enumerate(dataloader):
        radar = batch_data['radar'].to(device)
        label = batch_data['label'].to(device).squeeze(1)

        mean = torch.mean(radar, dim=(0, 2), keepdim=True)
        std = torch.std(radar, dim=(0, 2), keepdim=True)
        radar = (radar - mean) / std

        label_data_int = label.to(torch.int64)
        one_hot_labels = F.one_hot(label_data_int, num_classes=4).float().permute(0, 2, 1)

        if PCA_flag:
            outputs_softmax, mid_feature = model(radar)
        else:
            outputs_softmax = model(radar)

        loss = loss_fn(outputs_softmax, one_hot_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        max_indices = torch.argmax(outputs_softmax, dim=1)

        if i % 50 == 0:
            label_numpy = label.cpu().detach().numpy()
            outputs_numpy = max_indices.cpu().detach().numpy()
            batch_f1_score, batch_RMSE = eval_functions(label_numpy, outputs_numpy)
            all_f1_scores.append(batch_f1_score)
            all_RMSE.append(batch_RMSE)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0
    avg_rmse = np.mean(all_RMSE) if all_RMSE else 0

    return avg_loss, avg_f1, avg_rmse


def validate_epoch(model, dataloader, loss_fn, device, PCA_flag=False):
    model.eval()
    total_loss = 0
    all_f1_scores = []
    all_avg_error = []
    all_p_wave_error = []
    all_qrs_wave_error = []
    all_t_wave_error = []
    all_pr_error = []
    all_qt_error = []

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            radar = batch_data['radar'].to(device)
            label = batch_data['label'].to(device).squeeze(1)

            mean = torch.mean(radar, dim=(0, 2), keepdim=True)
            std = torch.std(radar, dim=(0, 2), keepdim=True)
            radar = (radar - mean) / std

            label_data_int = label.to(torch.int64)
            one_hot_labels = F.one_hot(label_data_int, num_classes=4).float().permute(0, 2, 1)

            if PCA_flag:
                outputs, mid_feature = model(radar)
            else:
                outputs = model(radar)

            loss = loss_fn(outputs, one_hot_labels)

            max_indices = torch.argmax(outputs, dim=1)
            label_numpy = label.cpu().detach().numpy()
            outputs_numpy = max_indices.cpu().detach().numpy()

            batch_f1_score, avg_error, p_wave_error, qrs_wave_error, t_wave_error, pr_error, qt_error = eval_functions_all(
                label_numpy, outputs_numpy
            )

            all_f1_scores.append(batch_f1_score)
            all_avg_error.append(avg_error)
            all_p_wave_error.append(p_wave_error)
            all_qrs_wave_error.append(qrs_wave_error)
            all_t_wave_error.append(t_wave_error)
            all_pr_error.append(pr_error)
            all_qt_error.append(qt_error)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0
    avg_avg_error = np.mean(all_avg_error) if all_avg_error else 0
    avg_p_wave_error = np.mean(all_p_wave_error) if all_p_wave_error else 0
    avg_qrs_wave_error = np.mean(all_qrs_wave_error) if all_qrs_wave_error else 0
    avg_t_wave_error = np.mean(all_t_wave_error) if all_t_wave_error else 0
    avg_pr_error = np.mean(all_pr_error) if all_pr_error else 0
    avg_qt_error = np.mean(all_qt_error) if all_qt_error else 0

    return (avg_loss, avg_f1, avg_avg_error, avg_p_wave_error,
            avg_qrs_wave_error, avg_t_wave_error, avg_pr_error, avg_qt_error)


def test_model(model, dataloader, loss_fn, device, save_dir, uid, visual_flag=False, PCA_flag=False):
    model.eval()
    total_loss = 0
    all_f1_scores = []
    all_avg_error = []
    all_p_wave_error = []
    all_qrs_wave_error = []
    all_t_wave_error = []
    all_pr_error = []
    all_qt_error = []

    mid_feature_list = []
    label_list = []

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            radar = batch_data['radar'].to(device)
            label = batch_data['label'].to(device).squeeze(1)
            ECG = batch_data['ECG'].to(device).squeeze(1)
            info = batch_data['info']

            mean = torch.mean(radar, dim=(0, 2), keepdim=True)
            std = torch.std(radar, dim=(0, 2), keepdim=True)
            radar = (radar - mean) / std

            label_data_int = label.to(torch.int64)
            one_hot_labels = F.one_hot(label_data_int, num_classes=4).float().permute(0, 2, 1)

            if PCA_flag:
                outputs, mid_feature = model(radar)
            else:
                outputs = model(radar)

            loss = loss_fn(outputs, one_hot_labels)

            max_indices = torch.argmax(outputs, dim=1)
            label_numpy = label.cpu().detach().numpy()
            outputs_numpy = max_indices.cpu().detach().numpy()
            ECG_numpy = ECG.cpu().detach().numpy()

            batch_f1_score, avg_error, p_wave_error, qrs_wave_error, t_wave_error, pr_error, qt_error = eval_functions_all(
                label_numpy, outputs_numpy, save_dir, info, uid, visual_flag, ECG_numpy
            )

            all_f1_scores.append(batch_f1_score)
            all_avg_error.append(avg_error)
            all_p_wave_error.append(p_wave_error)
            all_qrs_wave_error.append(qrs_wave_error)
            all_t_wave_error.append(t_wave_error)
            all_pr_error.append(pr_error)
            all_qt_error.append(qt_error)

            total_loss += loss.item()

            if PCA_flag:
                for j in range(outputs.shape[0]):
                    mid_fea = mid_feature[j]
                    if "BBB" in info['diagnosis'][j]:
                        label_list.append(1)
                    else:
                        label_list.append(0)

                    intermediate_features = mid_fea.cpu()
                    mid_feature_list.append(intermediate_features)

    if PCA_flag and mid_feature_list:
        intermediate_features = torch.stack(mid_feature_list)
        labels = np.array(label_list)

        intermediate_features_np = intermediate_features.numpy()
        intermediate_features_reshaped = intermediate_features_np.reshape(
            intermediate_features_np.shape[0], -1
        )

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(intermediate_features_reshaped)

        class_0_features = reduced_features[labels == 0]
        class_1_features = reduced_features[labels == 1]

        fig = plt.figure(figsize=(10, 6))
        plt.scatter(class_0_features[:, 0], class_0_features[:, 1],
                    label='not RBBB/LBBB', alpha=0.5, color='red')
        plt.scatter(class_1_features[:, 0], class_1_features[:, 1],
                    label='RBBB/LBBB', alpha=0.5, color='green')
        plt.title('PCA of Intermediate Features')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.savefig(os.path.join(save_dir, uid + '-PCA.png'))
        plt.close(fig)

    avg_loss = total_loss / len(dataloader)
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0
    avg_avg_error = np.mean(all_avg_error) if all_avg_error else 0
    avg_p_wave_error = np.mean(all_p_wave_error) if all_p_wave_error else 0
    avg_qrs_wave_error = np.mean(all_qrs_wave_error) if all_qrs_wave_error else 0
    avg_t_wave_error = np.mean(all_t_wave_error) if all_t_wave_error else 0
    avg_pr_error = np.mean(all_pr_error) if all_pr_error else 0
    avg_qt_error = np.mean(all_qt_error) if all_qt_error else 0

    return (avg_loss, avg_f1, avg_avg_error, avg_p_wave_error,
            avg_qrs_wave_error, avg_t_wave_error, avg_pr_error, avg_qt_error)


def main():
    parser = argparse.ArgumentParser(description='radar singal segmentation')
    parser.add_argument('--data_dir', type=str, default='./dataset_finetune/Dataset_4898_0318')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--data_set_len', type=int, default=640)
    parser.add_argument('--model', type=str, default='SwinTransformerFPN')
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--visual_flag', action='store_true')
    parser.add_argument('--pca_flag', action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    uid = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"ID: {uid}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, val_loader, test_loader, point_nums = prepare_dataloaders(
        args.data_dir, args.batch_size, args.data_set_len
    )

    model = select_model(args.model, point_nums, device)
    print(f"model: {type(model).__name__}")

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    print(f"optimizer: {type(optimizer).__name__}, lr: {args.lr}")

    best_model = None
    best_performance = 0
    patience = 10
    no_improvement = 0

    print("begin train...")
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')

        train_loss, train_f1, train_rmse = train_epoch(
            model, train_loader, loss_fn, optimizer, device, args.pca_flag
        )
        print(f"train loss: {train_loss:.4f}, F1: {train_f1:.4f}, RMSE: {train_rmse:.4f}")

        (val_loss, val_f1, val_avg_error, val_p_wave_error,
         val_qrs_wave_error, val_t_wave_error, val_pr_error, val_qt_error) = validate_epoch(
            model, val_loader, loss_fn, device, args.pca_flag
        )
        print(f"val loss: {val_loss:.4f}, F1: {val_f1:.4f}")
        print(f"val avg error: {val_avg_error:.4f}, P wave error: {val_p_wave_error:.4f}")
        print(f"QRS wave error: {val_qrs_wave_error:.4f}, T wave error: {val_t_wave_error:.4f}")
        print(f"PR interval error: {val_pr_error:.4f}, QT interval error: {val_qt_error:.4f}")

        if val_f1 > best_performance:
            best_model = copy.deepcopy(model)
            best_performance = val_f1
            no_improvement = 0
            save_path = os.path.join(args.output_dir, f'best_model_{uid}.pth')
            torch.save(best_model.state_dict(), save_path)
            print(f"save path: {save_path}")
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"early stop, patience: {patience}")
            break

    print("begin test...")
    (test_loss, test_f1, test_avg_error, test_p_wave_error,
     test_qrs_wave_error, test_t_wave_error, test_pr_error, test_qt_error) = test_model(
        best_model, test_loader, loss_fn, device, args.output_dir, uid,
        args.visual_flag, args.pca_flag
    )

    print("=" * 50)
    print("test result:")
    print(f"test loss: {test_loss:.4f}, F1: {test_f1:.4f}")
    print(f"test avg error: {test_avg_error:.4f}, P wave error: {test_p_wave_error:.4f}")
    print(f"QRS wave error: {test_qrs_wave_error:.4f}, T wave error: {test_t_wave_error:.4f}")
    print(f"PR interval error: {test_pr_error:.4f}, QT interval error: {test_qt_error:.4f}")
    print("=" * 50)

    results = {
        'uid': uid,
        'model': args.model,
        'test_loss': test_loss,
        'test_f1': test_f1,
        'test_avg_error': test_avg_error,
        'test_p_wave_error': test_p_wave_error,
        'test_qrs_wave_error': test_qrs_wave_error,
        'test_t_wave_error': test_t_wave_error,
        'test_pr_error': test_pr_error,
        'test_qt_error': test_qt_error
    }

    results_path = os.path.join(args.output_dir, f'results_{uid}.txt')
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(f"result save: {results_path}")
    print("finish")


if __name__ == '__main__':
    main()
