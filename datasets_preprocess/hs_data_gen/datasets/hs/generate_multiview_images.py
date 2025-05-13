# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os, sys
from tqdm import tqdm
import argparse
import PIL.Image
import numpy as np
import json
import imageio
sys.path.append("/home/zgtang/data_gen/croco/")
print(sys.path)
from datasets.hs.paths import list_scenes_available
import cv2
import quaternion
import shutil
from datasets.hs.multiview_habitat_sim_generator import MultiviewHabitatSimGenerator, NoNaviguableSpaceError
import torch
        

def generate_multiview_images_for_scene(scene_dataset_config_file,
                                        scene,
                                        navmesh,
                                        output_dir, 
                                        views_count,
                                        size, 
                                        exist_ok=False, 
                                        generate_depth=False,
                                        **kwargs):
    """
    Generate tuples of overlapping views for a given scene.
    generate_depth: generate depth images and camera parameters.
    """
    if os.path.exists(output_dir) and not exist_ok:
        print(f"Scene {scene}: data already generated. Ignoring generation.")
        return
    try:
        print(f"Scene {scene}: {size} multiview acquisitions to generate...")
        os.makedirs(output_dir, exist_ok=exist_ok)

        metadata_filename = os.path.join(output_dir, "metadata.json")

        metadata_template = dict(scene_dataset_config_file=scene_dataset_config_file,
            scene=scene, 
            navmesh=navmesh,
            views_count=views_count,
            size=size,
            generate_depth=generate_depth,
            **kwargs)
        metadata_template["multiviews"] = dict()

        if os.path.exists(metadata_filename):
            print("Metadata file already exists:", metadata_filename)
            print("Loading already generated metadata file...")
            with open(metadata_filename, "r") as f:
                metadata = json.load(f)

            # for key in metadata_template.keys():
            #     if key != "multiviews":
            #         assert metadata_template[key] == metadata[key], f"existing file is inconsistent with the input parameters:\nKey: {key}\nmetadata: {metadata[key]}\ntemplate: {metadata_template[key]}."
        else:
            print("No temporary file found. Starting generation from scratch...")
            metadata = metadata_template

        starting_id = len(metadata["multiviews"])
        print(f"Starting generation from index {starting_id}/{size}...")
        if starting_id >= size:
            print("Generation already done.")
            return

        generator = MultiviewHabitatSimGenerator(scene_dataset_config_file=scene_dataset_config_file,
                                                scene=scene,
                                                navmesh=navmesh,
                                                views_count = views_count,
                                                size = size,
                                                **kwargs)
        dir_name = "manifold://ondevice_ai_writedata/tree/zgtang/data/hs/"
        manifold_output_dir = output_dir.replace("/home/zgtang/data_gen/croco/hs/", dir_name)
        meta_output_dir = output_dir.replace("/home/zgtang/data_gen/croco/hs/", "/home/zgtang/data_gen/croco/hs_meta/")
        print(dir_name, manifold_output_dir, meta_output_dir)
        # input('aha')
        os.makedirs(meta_output_dir, exist_ok=True)
        for idx in tqdm(range(starting_id, size)):
            # Generate / re-generate the observations
            try:        
                data = generator[idx]
                print('uploading', idx, scene)
                observations = data["observations"]
                positions = data["positions"]
                orientations = data["orientations"]

                idx_label = f"{idx:08}"
                all_img = []
                all_depth = []
                pose_raw_list = []
                intrinsic_raw = None
                rgb_list = []
                depth_list = []
                meta_data_i = {'nv': len(observations)}
                for oidx, observation in enumerate(observations):
                    observation_label = f"{oidx + 1}" # Leonid is indexing starting from 1
                    # Color image saved using PIL
                    img = PIL.Image.fromarray(observation['color'][:,:,:3])
                    all_img.append(observation['color'][:,:,:3])
                    # filename = os.path.join(output_dir, f"{idx_label}_{observation_label}.png")
                    # manifold_filename = os.path.join(manifold_output_dir, f"{idx_label}_{observation_label}.png")
                    # img.save(filename)
                    # rgb_list.append(manifold_filename)
                    if generate_depth:
                        # Depth image as EXR file
                        # filename = os.path.join(output_dir, f"{idx_label}_{observation_label}_depth.exr")
                        # cv2.imwrite(filename, observation['depth'], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                        
                        # filename = os.path.join(output_dir, f"{idx_label}_{observation_label}_depth.npy")
                        # manifold_filename = os.path.join(manifold_output_dir, f"{idx_label}_{observation_label}_depth.npy")
                        # depth_list.append(manifold_filename)
                        # print(observation['depth'].shape, observation['depth'].max(), observation['depth'].min()) # (512, 512) 1.8684721 0.52613306
                        # np.save(filename, observation['depth'])
                        all_depth.append(observation['depth'])
                        # Camera parameters
                        # print(observation)
                        intrinsic_raw = observation['camera_intrinsics'].tolist()
                        P = np.eye(4)
                        P[:3,:3] = observation['R_cam2world']
                        P[:3,3] = observation['t_cam2world']
                        R = np.eye(4)
                        R[:3, :3] = [
                            [1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]
                        ]
                        P = np.matmul(R, P)
                        pose_raw_list.append(P.tolist())
                        # camera_params = dict([(key, observation[key].tolist()) for key in ("camera_intrinsics", "R_cam2world", "t_cam2world")])
                        # filename = os.path.join(output_dir, f"{idx_label}_{observation_label}_camera_params.json")
                        # with open(filename, "w") as f:
                        #     json.dump(camera_params, f)
                imageio.imwrite(os.path.join(output_dir, f"{idx_label}.png"), np.concatenate(all_img, axis=1))
                png_16bit = np.concatenate(all_depth, axis=1)
                png_16bit = (png_16bit * 1000)
                png_16bit[png_16bit > 65535] = 0
                png_16bit = png_16bit.astype(np.uint16)
                imageio.imwrite(os.path.join(output_dir, f"{idx_label}_depth.png"), png_16bit, format='png')
                # np.savez_compressed(os.path.join(output_dir, , png_16bit)
                rgb_list.append(os.path.join(manifold_output_dir, f"{idx_label}.png"))
                depth_list.append(os.path.join(manifold_output_dir, f"{idx_label}_depth.png"))
                meta_data_i['pose_raw_list'] = pose_raw_list
                meta_data_i['intrinsic_raw'] = intrinsic_raw
                meta_data_i['rgb_list'] = rgb_list
                meta_data_i['depth_list'] = depth_list
                meta_data_i['C'] = np.round(data["pairwise_visibility_ratios"], 2).tolist()
                
                with open(os.path.join(meta_output_dir, f"{idx_label}.json"), "w") as f:
                    json.dump(meta_data_i, f, indent=4)
                torch.save(meta_data_i, os.path.join(meta_output_dir, f"{idx_label}_extra.pt"))

                metadata["multiviews"][idx_label] = {"positions": positions.tolist(),
                                                    "orientations": orientations.tolist(),
                                                    "C": data["pairwise_visibility_ratios"].tolist(),
                                                    "valid_fractions": data["valid_fractions"].tolist(),
                                                    "pairwise_visibility_ratios": data["pairwise_visibility_ratios"].tolist()}
                # print('uploading done')
            except RecursionError:
                print("Recursion error: unable to sample observations for this scene. We will stop there.")
                while 1:
                    pass
                break

            # Regularly save a temporary metadata file, in case we need to restart the generation
            if idx % 10 == 0:
                with open(metadata_filename, "w") as f:
                    json.dump(metadata, f)

        # Save metadata
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f)

        generator.close()
    except NoNaviguableSpaceError:
        pass

def create_commandline(scene_data, generate_depth, exist_ok=False):
    """
    Create a commandline string to generate a scene.
    """
    def my_formatting(val):
        if val is None or val == "":
            return '""'
        else:
            return val
    commandline = f"""python {__file__} --scene {my_formatting(scene_data.scene)} 
    --scene_dataset_config_file {my_formatting(scene_data.scene_dataset_config_file)} 
    --navmesh {my_formatting(scene_data.navmesh)} 
    --output_dir {my_formatting(scene_data.output_dir)} 
    --generate_depth {int(generate_depth)} 
    --exist_ok {int(exist_ok)}
    """
    commandline = " ".join(commandline.split())
    return commandline

if __name__ == "__main__":
    os.umask(2)

    parser = argparse.ArgumentParser(description="""Example of use -- listing commands to generate data for scenes available:
    > python datasets/habitat_sim/generate_multiview_habitat_images.py --list_commands
    """)

    parser.add_argument("--output_dir", type=str, default = "")
    parser.add_argument("--list_commands", action='store_true', help="list commandlines to run if true")
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--scene_dataset_config_file", type=str, default="")
    parser.add_argument("--navmesh", type=str, default="")

    parser.add_argument("--generate_depth", type=int, default=1)
    parser.add_argument("--exist_ok", type=int, default=1)

    parser.add_argument("--div", type=int, default=1)
    parser.add_argument("--mod", type=int, default=0)

    kwargs = dict(resolution=[256,256], hfov=60, views_count = 2, size=1000)
    kwargs = dict(resolution=[512,512], hfov=60, views_count = 5, size=1000)
    kwargs = dict(resolution=[100,100], rerender_resolution=[512,512], hfov=58, views_count = 12, size=100, minimum_covisibility=0.3, maximum_covisibility=0.7)

    args = parser.parse_args()
    generate_depth=bool(args.generate_depth)
    exist_ok = bool(args.exist_ok)

    if args.list_commands:
        # Listing scenes available...
        scenes_data = list_scenes_available(base_output_dir=args.output_dir)
        torch.save(scenes_data, "./scenes_data.pth")
        scenes_data = torch.load("./scenes_data.pth")
        # print(scenes_data[0], len(scenes_data))
        # input()
        
        for scene_data in scenes_data:
            print('scene data', scene_data)
            print(create_commandline(scene_data, generate_depth=generate_depth, exist_ok=exist_ok))
    else:
        scenes_data = torch.load("./scenes_data.pth")
        for scene_id, scene_data in enumerate(scenes_data):
            if scene_id % args.div != args.mod:
                continue
            args.scene = scene_data.scene
            args.navmesh = scene_data.navmesh
            args.output_dir = scene_data.output_dir
            generate_multiview_images_for_scene(scene=args.scene,
                                                scene_dataset_config_file = args.scene_dataset_config_file,
                                                navmesh = args.navmesh,
                                                output_dir = args.output_dir,
                                                exist_ok=exist_ok,
                                                generate_depth=generate_depth,
                                                **kwargs)
