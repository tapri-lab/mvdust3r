#!/bin/bash

# n_job=8
# n_v=5
# n_render=1
# data_type=scannetpp
# hardness=hard

# for ((i=0; i<n_job; i++))
# do
#   echo $((8 * n_job)) $i
#   torchx run -s mast -- -h zionex -j 1x8 -m datasets_preprocess/preproc_scannet2.py -- --div $((8 * n_job)) --node-no $i --n-v $n_v --n-render $n_render --data-type $data_type --hardness $hardness
# done
# #   torchx run -s mast -- -h zionex -j 1x8 -m datasets_preprocess/preproc_scannet_render.py -- --div 64 --node-no $i

################### FIRST RUN THIS TO GENERATE SCENE LIST

# python hs_data_gen/hs/generate_multiview_images.py --list_commands --output_dir hs

################### SECOND RUN THIS FOR RENDERING AND TUPLE GENERATION

# n_node=30
# n_process_per_node=8
# n_v=12
# data_name=hs
# n_tuple_per_scene=2000
# view_sample_type=first
# random_step_variance=2.0

# for ((i=0; i<n_node; i++))
# do
#   echo $((8 * n_node)) $i
#   torchx run -s mast_conda -- -h zion_2s -j 1x8 -m hs/generate_multiview_images.py -- --div $((n_process_per_node * n_node)) --start_rank_id $((n_process_per_node * i)) --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance
# done




# n_node=30
# n_process_per_node=8
# n_v=12
# data_name=hs_tree
# n_tuple_per_scene=2000
# view_sample_type=random
# random_step_variance=1.0

# for ((i=0; i<n_node; i++))
# do
#   echo $((8 * n_node)) $i
#   torchx run -s mast_conda -- -h zion_2s -j 1x8 -m hs/generate_multiview_images.py -- --div $((n_process_per_node * n_node)) --start_rank_id $((n_process_per_node * i)) --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance
# done

# n_node=30
# n_process_per_node=8
# n_v=12
# data_name=hs_tree_last
# n_tuple_per_scene=1000
# view_sample_type=tree
# random_step_variance=1.0

# for ((i=0; i<n_node; i++))
# do
#   echo $((8 * n_node)) $i
#   torchx run -s mast_conda -- -h zion_2s -j 1x8 -m hs/generate_multiview_images.py -- --div $((n_process_per_node * n_node)) --start_rank_id $((n_process_per_node * i)) --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance
# done

# n_node=30
# n_process_per_node=8
# n_v=12
# data_name=hs_chain
# n_tuple_per_scene=200
# view_sample_type=chain
# random_step_variance=1.0

# for ((i=0; i<n_node; i++))
# do
#   echo $((8 * n_node)) $i
#   torchx run -s mast_conda -- -h zion_2s -j 1x8 -m hs/generate_multiview_images.py -- --div $((n_process_per_node * n_node)) --start_rank_id $((n_process_per_node * i)) --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance
# done

# n_node=60
# n_process_per_node=8
# n_v=12
# data_name=hs_tree_diverse_first
# n_tuple_per_scene=2000
# view_sample_type=tree_diverse_first
# random_step_variance=1.0

# for ((i=0; i<n_node; i++))
# do
#   echo $((8 * n_node)) $i
#   torchx run -s mast_conda -- -h zion_2s -j 1x8 -m hs/generate_multiview_images.py -- --div $((n_process_per_node * n_node)) --start_rank_id $((n_process_per_node * i)) --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance
# done


# n_node=101
# n_process_per_node=8
n_v=12
data_name=hs_tree_diverse_first_2
n_tuple_per_scene=2000
view_sample_type=tree_diverse_first
random_step_variance=2.0

python hs/generate_multiview_images.py --div 1 --start_rank_id 0 --n_v $n_v --data_name $data_name --n_tuple_per_scene $n_tuple_per_scene --view_sample_type $view_sample_type --random_step_variance $random_step_variance


################### THIRD: RUN THIS TO KILL JOB LOCALLY IF USE "&"

# ps -eo pid,ppid,stat,tty,cmd | grep 'multiview'

# for i in {0..15}
# do
#   kill $((i + 3199766))
# done
