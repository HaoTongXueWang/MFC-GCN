import torch
import numpy as np
import warnings
import copy
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from config import Config
from models import MultiNetworkMoEModel
from train_utils import train, evaluate_with_predictions, calculate_pos_weight
from visualize import plot_cv_curves

warnings.filterwarnings("ignore")

def run_cross_validation(config):
    all_networks = [torch.load(path) for path in config.data_paths]
    num_nodes = all_networks[0].num_nodes
    labels = all_networks[0].y

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    splits = list(skf.split(range(num_nodes), labels.cpu().numpy()))

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n=== 正在处理第 {fold + 1}/{config.n_folds} 折 ===")

        config.network_weights = [1, 0.5, 0.5, 0.3, 0.1]

        current_networks = []
        for net in all_networks:
            net_copy = Data(
                x=net.x.clone(),
                edge_index=net.edge_index.clone(),
                y=net.y.clone()
            )
            net_copy.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            net_copy.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            net_copy.train_mask[train_idx] = True
            net_copy.test_mask[test_idx] = True
            current_networks.append(net_copy.to(config.device))

        labels = current_networks[0].y
        train_mask = current_networks[0].train_mask

        pos_weight = calculate_pos_weight(labels, train_mask).to(config.device)
        config.pos_weight = pos_weight
        print(f"当前折的pos_weight: {pos_weight.item():.2f}")

        config.in_channels = current_networks[0].num_node_features
        model = MultiNetworkMoEModel(config).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate,
                                     weight_decay=config.weight_decay)

        best_roc_auc = 0
        no_improve = 0
        best_test_roc = 0
        best_test_pr = 0
        best_gate_weights = None
        best_preds = None
        best_trues = None

        for epoch in range(1, config.num_epochs + 1):
            loss = train(
                model, optimizer, current_networks, labels, train_mask, epoch, pos_weight
            )

            if epoch % 10 == 0 or epoch == config.num_epochs:
                train_preds, train_trues, train_roc, train_pr = evaluate_with_predictions(
                    model, current_networks, labels, train_mask)
                test_preds, test_trues, test_roc, test_pr = evaluate_with_predictions(
                    model, current_networks, labels, current_networks[0].test_mask)

                with torch.no_grad():
                    _, _, _, _, gate_weights, _ = model(current_networks, train_mode=False)
                    current_gate_weights = gate_weights.mean(0).cpu().tolist()

                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
                print(f'  Train ROC-AUC: {train_roc:.4f}, PR-AUC: {train_pr:.4f}')
                print(f'  Test  ROC-AUC: {test_roc:.4f}, PR-AUC: {test_pr:.4f}')
                print(f'  Expert Weights: {[round(w, 3) for w in current_gate_weights]}')

                if test_roc > best_roc_auc:
                    best_roc_auc = test_roc
                    best_test_roc = test_roc
                    best_test_pr = test_pr
                    best_gate_weights = current_gate_weights
                    best_preds = test_preds
                    best_trues = test_trues
                    no_improve = 0
                    torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pt')
                    print(f"Best model saved for fold {fold + 1}")
                else:
                    no_improve += 1
                    if no_improve >= config.patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        fold_results.append({
            'train_roc': train_roc,
            'train_pr': train_pr,
            'test_roc': best_test_roc,
            'test_pr': best_test_pr,
            'final_weights': model.config.network_weights,
            'gate_weights': best_gate_weights,
            'preds': best_preds,
            'trues': best_trues
        })

    print("\n=== 交叉验证结果摘要 ===")
    test_rocs = [res['test_roc'] for res in fold_results]
    test_prs = [res['test_pr'] for res in fold_results]

    print("\n=== 每折详细结果 ===")
    for fold, res in enumerate(fold_results):
        print(f"折 {fold + 1}:")
        print(f"  训练 ROC-AUC: {res['train_roc']:.4f}, PR-AUC: {res['train_pr']:.4f}")
        print(f"  测试 ROC-AUC: {res['test_roc']:.4f}, PR-AUC: {res['test_pr']:.4f}")
        print(f"  最终网络权重: {res['final_weights']}")
        print(f"  专家门控权重: {[round(w, 3) for w in res['gate_weights']]}")

    print("\n=== 整体统计 ===")
    print(f"平均测试 ROC-AUC: {np.mean(test_rocs):.4f} ± {np.std(test_rocs):.4f}")
    print(f"平均测试 PR-AUC: {np.mean(test_prs):.4f} ± {np.std(test_prs):.4f}")

    all_gate_weights = np.array([res['gate_weights'] for res in fold_results])
    avg_gate_weights = all_gate_weights.mean(axis=0)
    std_gate_weights = all_gate_weights.std(axis=0)

    print("\n=== 专家使用分析 ===")
    for i, (avg, std) in enumerate(zip(avg_gate_weights, std_gate_weights)):
        print(f"专家 {i + 1}: {avg:.4f} ± {std:.4f}")

    # 绘制曲线，包含平均曲线
    plot_cv_curves(
        fold_results,
        save_path='cv_curves.png',
        plot_mean=True,
        mean_line_name="Average",
        mean_line_color="red",
        mean_line_style="--",
        mean_line_width=3
    )
    np.save('moe_cross_val_results.npy', np.array(fold_results, dtype=object))

if __name__ == "__main__":
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    run_cross_validation(config)