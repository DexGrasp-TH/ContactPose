import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict

sys.path.append(".")
sys.path.append("..")
from utilities.import_open3d import *

from utilities.dataset import ContactPose


class GraspDataset:
    def __init__(
        self,
        object_list_file="data/object_names.txt",
        intents=("use", "handoff"),
        participants=range(1, 51),
        verbose=False,
    ):
        self.object_list_file = object_list_file
        self.intents = intents
        self.participants = participants
        self.verbose = verbose

        self.object_names = self._load_object_names()
        # self.object_names = ["apple"]

        self.all_grasps = self.collect()
        self.analysis()

    def _load_object_names(self):
        with open(self.object_list_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]

        # remove unwanted objects
        blacklist = {"hands", "palm_print"}
        names = [n for n in names if n not in blacklist]

        if self.verbose:
            print(f"[INFO] Loaded {len(names)} objects (filtered)")
        return names

    def iterate_grasps(self):
        """
        Generator that yields valid ContactPose instances
        """
        for p_num in tqdm(self.participants, desc="Participants"):
            for intent in self.intents:
                for obj_name in self.object_names:
                    try:
                        cp = ContactPose(p_num, intent, obj_name)

                        if not os.path.exists(cp.object_filename):
                            continue

                        yield {
                            "p_num": p_num,
                            "intent": intent,
                            "object_name": obj_name,
                            "contact_pose": cp,
                        }

                    except Exception as e:
                        if self.verbose:
                            print(f"[WARN] Skip p={p_num}, intent={intent}, obj={obj_name}: {e}")
                        continue

    def collect(self):
        """
        Load all grasps into a list (if you don't want lazy loading)
        """
        all_grasps = list(self.iterate_grasps())
        if self.verbose:
            print(f"[INFO] Collected {len(all_grasps)} valid grasps")
        return all_grasps

    def analysis(self):
        grasp_data_list = []

        for grasp in tqdm(self.all_grasps, desc="All grasps"):
            cp = grasp["contact_pose"]

            # Compute object scale
            mesh = o3dio.read_triangle_mesh(cp.object_filename)
            mesh.scale(0.001, center=(0, 0, 0))  # 转换单位: 从 mm 到 m (缩小 1000 倍)
            verts = np.asarray(mesh.vertices)
            v_extent = np.max(verts, axis=0) - np.min(verts, axis=0)
            diagonal_scale = np.linalg.norm(v_extent)

            # Compute grasp type
            joint_locs = cp.hand_joints()
            if joint_locs[0] is None and joint_locs[1] is not None:
                grasp_type = "right"
            elif joint_locs[0] is not None and joint_locs[1] is None:
                grasp_type = "left"
            elif joint_locs[0] is not None and joint_locs[1] is not None:
                grasp_type = "both"

            grasp_data_list.append({"scale": diagonal_scale, "type": grasp_type})

        print("\n========== Grasp Statistics ==========")
        print(f"Total grasps: {len(grasp_data_list)}\n")

        types = [g["type"] for g in grasp_data_list]
        type_counter = Counter(types)
        print("Grasp type distribution:")
        for k, v in type_counter.items():
            print(f"  {k:>5s}: {v:5d} ({v / len(types) * 100:.2f}%)")

        print("\n======================================")

        ############# Visualize grasp type distribution over object scale intervals #############
        df = pd.DataFrame(grasp_data_list)
        # 1. 动态生成 Bin 区间
        bin_size = 0.02
        max_scale = df["scale"].max()
        # 创建从 0 到 max_scale + bin_size 的范围
        bins = np.arange(0, max_scale + bin_size, bin_size)
        # 2. 使用 pd.cut 生成区间标签，格式为 [0.0, 0.02)
        df["scale_bin"] = pd.cut(df["scale"], bins=bins, right=False, precision=2, include_lowest=True)
        # 3. 统计表格
        pivot_table = df.groupby(["scale_bin", "type"]).size().unstack(fill_value=0)
        # 打印统计
        print("\n--- Grasp Type Distribution by Object Scale Intervals ---")
        print(pivot_table)
        self.visualize(pivot_table)

    def visualize(self, pivot_table):
        sns.set(style="whitegrid")

        # 绘制堆叠柱状图
        # ax 为绘图对象，方便后续微调标签
        ax = pivot_table.plot(
            kind="bar", stacked=True, figsize=(14, 7), color=["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"]
        )

        plt.title("Grasp Type Distribution vs Object Scale Interval", fontsize=15)
        plt.xlabel("Object Scale Interval [Start, End) / meters", fontsize=12)
        plt.ylabel("Number of Grasps", fontsize=12)

        # 优化 X 轴标签：pd.cut 生成的标签默认是 Interval 对象，转为字符串更美观
        xticklabels = [f"{str(label)}" for label in pivot_table.index]
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")

        plt.legend(title="Grasp Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # 保存并显示
        plt.show()


if __name__ == "__main__":
    dataset = GraspDataset(
        object_list_file="data/object_names.txt",
        intents=["use", "handoff"],
        participants=range(1, 51),
    )

    # grasps = dataset.collect()

    # # Example analysis hook
    # print("Example grasp entry:")
    # g = grasps[0]
    # print(f"p={g['p_num']}, intent={g['intent']}, object={g['object_name']}")

    # # Example: access hand joints
    # cp = g["contact_pose"]
    # joints = cp.hand_joints()
    # print("Hand joints loaded:", joints is not None)
