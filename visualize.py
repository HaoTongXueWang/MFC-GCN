import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_cv_curves(fold_results,
                   save_path=None,
                   curve_settings=None,
                   plot_mean=True,
                   mean_line_name="Average",
                   mean_line_color="red",
                   mean_line_style="--",
                   mean_line_width=3):
    plt.figure(figsize=(15, 6))

    # 默认设置
    default_settings = {
        'roc': {
            'line_names': [f'Fold {i + 1}' for i in range(len(fold_results))],
            'line_values': [res['test_roc'] for res in fold_results],
            'mean_value': np.mean([res['test_roc'] for res in fold_results]),
            'std_value': np.std([res['test_roc'] for res in fold_results]),
            'xlabel': 'False Positive Rate',
            'ylabel': 'True Positive Rate',
            'title': 'ROC Curves'
        },
        'pr': {
            'line_names': [f'Fold {i + 1}' for i in range(len(fold_results))],
            'line_values': [res['test_pr'] for res in fold_results],
            'mean_value': np.mean([res['test_pr'] for res in fold_results]),
            'std_value': np.std([res['test_pr'] for res in fold_results]),
            'xlabel': 'Recall',
            'ylabel': 'Precision',
            'title': 'PR Curves'
        }
    }

    # 合并用户自定义设置
    if curve_settings:
        for curve_type in ['roc', 'pr']:
            if curve_type in curve_settings:
                default_settings[curve_type].update(curve_settings[curve_type])

    settings = default_settings

    # ========== ROC曲线 ==========
    plt.subplot(1, 2, 1)

    # 计算平均ROC曲线
    if plot_mean:
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)

        for res in fold_results:
            fpr, tpr, _ = roc_curve(res['trues'], res['preds'])
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= len(fold_results)
        mean_auc = auc(all_fpr, mean_tpr)

        # 绘制平均曲线
        plt.plot(all_fpr, mean_tpr, color=mean_line_color, linestyle=mean_line_style,
                 lw=mean_line_width, label=f'{mean_line_name} ({mean_auc:.3f})')

    # 绘制各折曲线
    for i, res in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(res['trues'], res['preds'])
        plt.plot(fpr, tpr, lw=2, alpha=0.7,
                 label=f"{settings['roc']['line_names'][i]} ({settings['roc']['line_values'][i]:.3f})")

    # 添加平均值图例
    plt.plot([], [], ' ',
             label=f"Mean = {settings['roc']['mean_value']:.3f} ± {settings['roc']['std_value']:.3f}")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(settings['roc']['xlabel'])
    plt.ylabel(settings['roc']['ylabel'])
    plt.title(settings['roc']['title'])
    plt.legend(loc="lower right")

    # ========== PR曲线 ==========
    plt.subplot(1, 2, 2)

    # 计算平均PR曲线
    if plot_mean:
        all_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(all_recall)

        for res in fold_results:
            precision, recall, _ = precision_recall_curve(res['trues'], res['preds'])
            mean_precision += np.interp(all_recall, recall[::-1], precision[::-1])

        mean_precision /= len(fold_results)
        mean_auc = auc(all_recall, mean_precision)

        # 绘制平均曲线
        plt.plot(all_recall, mean_precision, color=mean_line_color, linestyle=mean_line_style,
                 lw=mean_line_width, label=f'{mean_line_name} ({mean_auc:.3f})')

    # 绘制各折曲线
    for i, res in enumerate(fold_results):
        precision, recall, _ = precision_recall_curve(res['trues'], res['preds'])
        plt.plot(recall, precision, lw=2, alpha=0.7,
                 label=f"{settings['pr']['line_names'][i]} ({settings['pr']['line_values'][i]:.3f})")

    # 添加平均值图例
    plt.plot([], [], ' ',
             label=f"Mean = {settings['pr']['mean_value']:.3f} ± {settings['pr']['std_value']:.3f}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(settings['pr']['xlabel'])
    plt.ylabel(settings['pr']['ylabel'])
    plt.title(settings['pr']['title'])
    plt.legend(loc="lower left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()