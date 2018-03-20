CUDA_VISIBLE_DEVICES=4 python run.py --agt 10 --usr 1 --max_turn 40 \
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p \
	      --dqn_hidden_size 80 \
	      --experience_replay_pool_size 1000 \
	      --episodes 500 \
	      --simulation_epoch_size 50 \
	      --write_model_dir ./deep_dialog/checkpoints/dropout/ \
	      --run_mode 3 \
	      --act_level 0 \
	      --slot_err_prob 0.00 \
	      --intent_err_prob 0.00 \
	      --batch_size 32 \
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
	      --warm_start 1 \
	      --warm_start_epochs 120 \

