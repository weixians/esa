import json
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from crowd_nav import global_util

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def read_json(filename):
    with open(filename) as f:
        policy_json = json.load(f)
        dynamic_result, static_result = policy_json["dynamic"], policy_json["static"]
        return dynamic_result, static_result


category_names = ["success", "collision", "timeout"]
category_colors = [[116, 209, 93, 255], [245, 165, 138, 255], [246, 249, 165, 255]]
for i in range(len(category_colors)):
    for j in range(4):
        category_colors[i][j] = category_colors[i][j] / 255


def stack_bar_chart(results, category_names, ax, title, show_extra_info):
    labels = list(results.keys())
    for i in range(len(labels)):
        if labels[i] == "ESA":
            labels[i] = "ESA (Ours)"
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    if show_extra_info:
        ax.yaxis.set_visible(True)
    else:
        ax.yaxis.set_visible(False)
    ax.set_xticklabels(labels=labels, rotation=0)
    ax.set_ylim(0, np.max(np.sum(data, axis=1)) + 5)

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        heights = data[:, i]
        starts = data_cum[:, i] - heights
        ax.bar(labels, heights, bottom=starts, width=0.5, label=colname, color=color)
        xcenters = starts + heights / 2
        r, g, b, _ = color
        text_color = "white" if r * g * b < 0.5 else "darkgrey"
        for y, (x, c) in enumerate(zip(xcenters, heights)):
            if c > 3:
                ax.text(
                    y,
                    x,
                    str(int(c)),
                    ha="center",
                    va="center",
                    color=text_color,
                    rotation=0,
                    fontsize=10,
                )
        ax.set_title(title, fontsize=12, color="black")

    if show_extra_info:
        ax.legend(loc="lower right")


def process_data(data: Dict, res: Dict, policy_name):
    for key, value in data.items():
        value = [round(v * 100) for v in value[2][:3]]
        if not key in res.keys():
            res[key] = {policy_name: value}
        else:
            res[key][policy_name] = value


if __name__ == "__main__":
    seed = 42
    policy_names = ["lstm_rl", "sarl", "esa"]
    folder = "test_result"

    base_dir = os.path.join(global_util.get_project_root(), folder)

    results_for_plot = {}
    length = len(policy_names)
    keys_remove = ["1", "2"]
    for policy in policy_names:
        dynamic_result, static_result = read_json(os.path.join(base_dir, "{}_seed{}.json".format(policy, seed)))
        # dynamic_result.pop("5")
        for key in keys_remove:
            if key in static_result.keys():
                static_result.pop(key)

        dynamic_result_new = {}
        static_result_new = {}
        for key in dynamic_result.keys():
            dynamic_result_new["dynamic_{}+static_{}".format(key, 0)] = dynamic_result[key]
        for key in static_result.keys():
            static_result_new["dynamic_{}+static_{}".format(5, key)] = static_result[key]

        process_data(dynamic_result_new, results_for_plot, policy.upper())
        process_data(static_result_new, results_for_plot, policy.upper())

    plt.rcParams["figure.figsize"] = (22.0, 4.0)
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["savefig.dpi"] = 120
    plt.rcParams["figure.dpi"] = 120

    # dynamic humans
    keys = list(results_for_plot.keys())
    for i in range(len(keys)):
        ax = plt.subplot(1, len(keys), i + 1)
        stack_bar_chart(results_for_plot[keys[i]], category_names, ax, keys[i], i == 0)
    plt.savefig(os.path.join(base_dir, "success_rates.png"))
    # out_format = 'pdf'
    # plt.savefig("{}.{}".format(os.path.join(base_dir, "success_rates"), out_format), bbox_inches='tight', dpi=200,
    #             format=out_format)
