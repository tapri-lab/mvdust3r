# Data Extraction

## Instructions for ScanNet

Scannet is structured as folders `scenexxxx_xx`. create a folder `scannet` in this directory and put `scenexxxx_xx` inside it.

## Instructions for other Habitat-Sim (HM3D, Gibson, MP3D)

HM3D, Gibson, MP3D contains `*.glb` and `*.navmesh` for each scene. put them in `habitat-sim-data/hm3d/train/`, `habitat-sim-data/hm3d/val/`, `habitat-sim-data/gibson/` and `habitat-sim-data/mp3d` for hm3d train, hm3d validation, gibson and mp3d respectively.

Here, we have a file `hs_scene_data.json` as the scene location config.