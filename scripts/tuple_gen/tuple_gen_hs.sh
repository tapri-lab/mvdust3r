#!/bin/bash

n_v=15
n_render=3
data_name=habitatSim_tree_step_2.0
n_tuple_per_scene=110
view_sample_type=tree_diverse_first
random_step_variance=2.0
render_overlap=0.0
split=all
minimum_covisibility=0.3
maximum_covisibility=0.7

python hs/generate_multiview_images.py --div 1 --start_rank_id 0 --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance --render_overlap $render_overlap --n_render $n_render --split $split --minimum_covisibility $minimum_covisibility --maximum_covisibility $maximum_covisibility


n_v=15
n_render=3
data_name=habitatSim_tree_step_1.0
n_tuple_per_scene=110
view_sample_type=tree_diverse_first
random_step_variance=1.0
render_overlap=0.0
split=all
minimum_covisibility=0.3
maximum_covisibility=0.7

python hs/generate_multiview_images.py --div 1 --start_rank_id 0 --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance --render_overlap $render_overlap --n_render $n_render --split $split --minimum_covisibility $minimum_covisibility --maximum_covisibility $maximum_covisibility


n_v=15
n_render=3
data_name=habitatSim_star_step_2.0
n_tuple_per_scene=110
view_sample_type=first
random_step_variance=2.0
render_overlap=0.0
split=all
minimum_covisibility=0.3
maximum_covisibility=0.7

python hs/generate_multiview_images.py --div 1 --start_rank_id 0 --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance --render_overlap $render_overlap --n_render $n_render --split $split --minimum_covisibility $minimum_covisibility --maximum_covisibility $maximum_covisibility

n_v=30
n_render=6
data_name=habitatSim_hm3d_test
n_tuple_per_scene=110
view_sample_type=tree_diverse_first
random_step_variance=2.0
render_overlap=0.0
split=hm3d_test
minimum_covisibility=0.3
maximum_covisibility=0.7

python hs/generate_multiview_images.py --div 1 --start_rank_id 0 --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance --render_overlap $render_overlap --n_render $n_render --split $split --minimum_covisibility $minimum_covisibility --maximum_covisibility $maximum_covisibility

n_v=30
n_render=6
data_name=habitatSim_mp3d_test
n_tuple_per_scene=110
view_sample_type=tree_diverse_first
random_step_variance=2.0
render_overlap=0.0
split=mp3d_test
minimum_covisibility=0.3
maximum_covisibility=0.7

python hs/generate_multiview_images.py --div 1 --start_rank_id 0 --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance --render_overlap $render_overlap --n_render $n_render --split $split --minimum_covisibility $minimum_covisibility --maximum_covisibility $maximum_covisibility