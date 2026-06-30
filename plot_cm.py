import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def score_to_sentiment_class(scores):
    """
    MOSI / MOSEI 连续情感分数 [-3, 3] 转为 7 类：
    -3,-2,-1,0,1,2,3 -> 0,1,2,3,4,5,6
    """
    scores = np.asarray(scores).reshape(-1)
    rounded = np.rint(scores).astype(int)
    rounded = np.clip(rounded, -3, 3)
    return rounded + 3


def plot_cm_like_paper(
    y_true_score=None,
    y_pred_score=None,
    cm_input=None,
    save_path="results/sentiment_confusion_matrix.png",
    normalize=False
):
    """
    两种用法：
    1. 有真实值和预测值：传入 y_true_score, y_pred_score
    2. 已经有混淆矩阵：传入 cm_input
    """

    class_names = ["HN", "N", "WN", "NT", "WP", "P", "HP"]

    if cm_input is not None:
        cm_show = np.asarray(cm_input, dtype=float)
    else:
        y_true_cls = score_to_sentiment_class(y_true_score)
        y_pred_cls = score_to_sentiment_class(y_pred_score)

        cm = confusion_matrix(
            y_true_cls,
            y_pred_cls,
            labels=list(range(7))
        )

        if normalize:
            row_sum = cm.sum(axis=1, keepdims=True)
            cm_show = np.divide(
                cm,
                row_sum,
                out=np.zeros_like(cm, dtype=float),
                where=row_sum != 0
            ) * 100
        else:
            cm_show = cm.astype(float)

    # 字体设置
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    # 图大小
    fig, ax = plt.subplots(figsize=(3.8, 3.5), dpi=600)

    # 关键：颜色和图中一致，使用 Blues，范围固定 0-80
    im = ax.imshow(
        cm_show,
        cmap="Blues",
        vmin=0,
        vmax=100,
        interpolation="nearest"
    )

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 20, 40, 60, 80,100])
    cbar.ax.tick_params(labelsize=12)

    # 坐标轴标签
    ax.set_xlabel("Predicted", fontsize=18)
    ax.set_ylabel("Label", fontsize=18)

    # 类别标签
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(7))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, rotation=45, va="center", fontsize=12)

    # 白色网格线，和示例图一致
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 写入每个格子的数字
    for i in range(7):
        for j in range(7):
            value = cm_show[i, j]

            # 深色背景用白字，浅色背景用黑字
            text_color = "white" if value >= 45 else "black"

            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11
            )

    ax.set_aspect("equal")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    cm_example = np.array([
        [4, 22, 14, 4, 2, 0, 0],
        [1, 82, 49, 18, 4, 2, 0],
        [1, 32, 70, 36, 2, 4, 0],
        [0, 8, 25, 48, 21, 4, 0],
        [0, 2, 11, 24, 47, 28, 1],
        [0, 1, 3, 8, 20, 65, 3],
        [0, 1, 0, 0, 2, 10, 7],
    ])

    plot_cm_like_paper(
        cm_input=cm_example,
        save_path="results/example_confusion_matrix.png"
    )








