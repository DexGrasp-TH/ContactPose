import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict
import trimesh
from scipy.spatial.transform import Rotation as sciR

sys.path.append(".")
sys.path.append("..")
from utilities.import_open3d import *

from utilities.dataset import ContactPose

NEW_DATASET_PATH = "/data/dataset/AnyScaleGrasp/ContactPose"


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

        # self._copy_object_files()

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

    def _copy_object_files(self):
        source_dir = "data/object_models"
        for obj_name in self.object_names:
            source_file_path = os.path.join(source_dir, f"{obj_name}.ply")
            mesh = trimesh.load(source_file_path, force="mesh")

            save_file_path = os.path.join(NEW_DATASET_PATH, f"object/{obj_name}/{obj_name}.obj")
            os.makedirs(os.path.dirname(save_file_path))
            mesh.export(save_file_path)
            print(f"Successfully converted {obj_name}'sPLY to OBJ. Vertices: {len(mesh.vertices)}")

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

            new_data = {
                "object": {},
                "hand": {
                    "left": {},
                    "right": {},
                },
                "extra": {},
            }

            new_data["object"]["name"] = obj_name = cp.object_name
            new_data["object"]["path"] = os.path.join(NEW_DATASET_PATH, "object", obj_name, f"{obj_name}.obj")
            new_data["object"]["rel_scale"] = 0.001  # unit mm to m
            new_data["object"]["pose"] = np.eye(4)  # 4*4 matrix

            for hand_idx, side in enumerate(["left", "right"]):
                if cp.mano_params[hand_idx] is None:
                    continue

                mano_params = cp.mano_params[hand_idx]
                mano_meshes = cp.mano_meshes()[hand_idx]

                hand_rot = (
                    sciR.from_matrix(mano_params["hTm"][:3, :3]) * sciR.from_rotvec(mano_params["pose"][:3])
                ).as_rotvec()
                hand_trans = mano_meshes["joints"][0]

                new_data["hand"][side]["trans"] = hand_trans
                new_data["hand"][side]["rot"] = hand_rot
                new_data["hand"][side]["mano_pose"] = mano_params["pose"][3:]
                new_data["hand"][side]["mano_betas"] = mano_params["betas"]
                new_data["hand"][side]["scale"] = 1000.0  # unit mm to m
                # new_data["hand"][side]["contacts"] = (tip_dists < CONTACT_THRESHOLD).cpu().tolist()

            save_dir = os.path.join(NEW_DATASET_PATH, "grasp", obj_name)
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f"{cp.p_num}_{cp.intent}.npy")
            np.save(file_path, new_data)
            if self.verbose:
                print(f"Successfully saved grasp: {file_path}.")


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
