import os, sys
from tqdm import tqdm
import argparse
import PIL.Image
import numpy as np
import json
import imageio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from hs.paths import list_scenes_available
import cv2
import quaternion
import shutil
from hs.multiview_habitat_sim_generator import MultiviewHabitatSimGenerator, NoNaviguableSpaceError
import torch

import torch.distributed as dist

def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='env://')

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
        

def generate_multiview_images_for_scene(scene_dataset_config_file,
                                        scene,
                                        navmesh,
                                        output_dir, 
                                        views_count,
                                        size,
                                        exist_ok=False, 
                                        generate_depth=False,
                                        device = "cuda:0",
                                        data_name="hs",
                                        **kwargs):
    """
    Generate tuples of overlapping views for a given scene.
    generate_depth: generate depth images and camera parameters.
    """

    output_dir_original = output_dir

    dir_name = f"../../metadata/{data_name}/"
    manifold_output_dir = output_dir_original.replace("../../metadata/", f"data/{data_name}/") # the dir written in the metadata file for image location
    meta_output_dir = output_dir_original.replace("metadata/", f"metadata/{data_name}/") # the dir where the metadata are saved
    output_dir = output_dir_original.replace("metadata/", f"data/{data_name}/") # the dir where the images are saved

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
                                                device=device,
                                                **kwargs)


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
                    
                    if generate_depth:
                        all_depth.append(observation['depth'])
                        # Camera parameters
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
                imageio.imwrite(os.path.join(output_dir, f"{idx_label}.png"), np.concatenate(all_img, axis=1))
                png_16bit = np.concatenate(all_depth, axis=1)
                png_16bit = (png_16bit * 1000)
                png_16bit[png_16bit > 65535] = 0
                png_16bit = png_16bit.astype(np.uint16)
                imageio.imwrite(os.path.join(output_dir, f"{idx_label}_depth.png"), png_16bit, format='png')
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

mp3d_test_list = [
    "2t7WUuJeko7",
    "5ZKStnWn8Zo",
    "ARNzJeq3xxb",
    "fzynW3qQPVF",
    "jtcxE69GiFV",
    "pa4otMbVnkk",
    "q9vSo1VnCiC",
    "rqfALeAoiTq",
    "UwV83HsGsw3",
    "wc2JMjhGNzB",
    "WYY7iVyf5p8",
    "YFuZgdQ5vWj",
    "yqstnuAEVhm",
    "YVUC4YcDtcY",
    "gxdoqLR6rwA",
    "gYvKGZ5eRqb",
    "RPmz2sHmrrY",
    "Vt2qJdWjCF2",
]

def is_val(x):

    if r"/mp3d/" in x:
        for mp3d_name in mp3d_test_list:
            if mp3d_name in x:
                return True
        return False
    if r"/val/" in x:
        return True
    return False

def is_train(x):
    
    if r"/mp3d/" in x:
        return False # all MP3D scenes are not used for training
    if r"/val/" in x:
        return False
    return True

def is_hm3d(x):
    if r"/hm3d/" in x:
        return True
    return False

def is_mp3d(x):
    if r"/mp3d/" in x:
        return True
    return False

if __name__ == "__main__":
    os.umask(2)

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default = "")
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--scene_dataset_config_file", type=str, default="")
    parser.add_argument("--navmesh", type=str, default="")

    parser.add_argument("--generate_depth", type=int, default=1)
    parser.add_argument("--exist_ok", type=int, default=1)

    parser.add_argument("--scenes_data_path", type=str, default="../../data/hs_scene_data.json")

    parser.add_argument("--div", type=int, default=1)
    parser.add_argument("--start_rank_id", type=int, default=0)

    parser.add_argument("--data_name", type = str, default = "hs")    
    
    parser.add_argument("--n_tuple_per_scene", type=int, default=100)
    parser.add_argument("--h", type=int, default=100)
    parser.add_argument("--w", type=int, default=100)
    parser.add_argument("--h2", type=int, default=512)
    parser.add_argument("--w2", type=int, default=512)
    parser.add_argument("--hfov", type=int, default=58)
    parser.add_argument("--view_sample_type", type=str, default="first")
    parser.add_argument("--n_v", type=int, default=12)
    parser.add_argument("--n_render", type=int, default=0)
    parser.add_argument("--random_step_variance", type = float, default = 2.0)
    parser.add_argument("--render_overlap", type = float, default = 0.95)

    parser.add_argument("--minimum_covisibility", type = float, default = 0.3)
    parser.add_argument("--maximum_covisibility", type = float, default = 0.7)

    parser.add_argument("--split", type = str, default = "all")

    args = parser.parse_args()
    generate_depth=bool(args.generate_depth)
    exist_ok = bool(args.exist_ok)
    
    kwargs = dict(resolution=[args.h,args.w], rerender_resolution=[args.h2,args.w2], hfov=args.hfov, views_count = args.n_v, n_render = args.n_render, size=args.n_tuple_per_scene, minimum_covisibility=args.minimum_covisibility, maximum_covisibility=args.maximum_covisibility, sample_type = args.view_sample_type, random_step_variance = args.random_step_variance, render_overlap = args.render_overlap)
    
    # init_distributed() # for distributed generation
    rank = get_rank()
    
    cuda_id = int(rank) % 8
    device = f"cuda:{cuda_id}"
    scenes_data_json = json.load(open(args.scenes_data_path, "r"))
    for scene_id, scene_data in enumerate(scenes_data_json):
        scene_name = scene_data['scene']
        if "hm3d" in args.split:
            if not is_hm3d(scene_name):
                continue
        if "mp3d" in args.split:
            if not is_mp3d(scene_name):
                continue

        if "test" in args.split:
            if not is_val(scene_name):
                continue
        if "train" in args.split:
            if not is_train(scene_name):
                continue
        
        if scene_id % args.div != rank + args.start_rank_id:
            continue
        args.scene = scene_data['scene']
        args.navmesh = scene_data['navmesh']
        args.output_dir = scene_data['output_dir']
        generate_multiview_images_for_scene(scene=args.scene,
                                            scene_dataset_config_file = args.scene_dataset_config_file,
                                            navmesh = args.navmesh,
                                            output_dir = args.output_dir,
                                            exist_ok=exist_ok,
                                            generate_depth=generate_depth,
                                            device = device,
                                            data_name = args.data_name,
                                            **kwargs)
        # try:
        #     generate_multiview_images_for_scene(scene=args.scene,
        #                                         scene_dataset_config_file = args.scene_dataset_config_file,
        #                                         navmesh = args.navmesh,
        #                                         output_dir = args.output_dir,
        #                                         exist_ok=exist_ok,
        #                                         generate_depth=generate_depth,
        #                                         device = device,
        #                                         data_name = args.data_name,
        #                                         **kwargs)
        # except:
        #     print('whole scene failed', scene_name)