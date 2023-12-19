Config(
    {
        'BASE_TASK_CONFIG_PATH': 'envs/habitat/configs/tasks/vln_mp3d.yaml', 
        'TASK_CONFIG': Config(
            {
                'SEED': 100, 
                'ENVIRONMENT': 
                    Config(
                        {
                            'MAX_EPISODE_STEPS': 500, 
                            'MAX_EPISODE_SECONDS': 10000000, 
                            'ITERATOR_OPTIONS': 
                                Config(
                                    {
                                        'CYCLE': True, 
                                        'SHUFFLE': True, 
                                        'GROUP_BY_SCENE': True, 
                                        'NUM_EPISODE_SAMPLE': -1, 
                                        'MAX_SCENE_REPEAT_EPISODES': -1, 
                                        'MAX_SCENE_REPEAT_STEPS': 10000, 
                                        'STEP_REPETITION_RANGE': 0.2
                                        }
                                    )
                                }
                        ), 
                'TASK': 
                    Config(
                        {
                            'TYPE': 'ObjectNav-v1', 
                            'SUCCESS_DISTANCE': 0.2, 
                            'SENSORS': ['GPS_SENSOR', 'COMPASS_SENSOR'], 
                            'MEASUREMENTS': ['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL'], 
                            'GOAL_SENSOR_UUID': 'pointgoal', 
                            'POSSIBLE_ACTIONS': ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN'], 
                            'ACTIONS': Config(
                                {
                                    'STOP': Config({'TYPE': 'StopAction'}), 
                                    'MOVE_FORWARD': Config({'TYPE': 'MoveForwardAction'}), 
                                    'TURN_LEFT': Config({'TYPE': 'TurnLeftAction'}), 
                                    'TURN_RIGHT': Config({'TYPE': 'TurnRightAction'}), 
                                    'LOOK_UP': Config({'TYPE': 'LookUpAction'}), 
                                    'LOOK_DOWN': Config({'TYPE': 'LookDownAction'}), 
                                    'TELEPORT': Config({'TYPE': 'TeleportAction'}), 
                                    'ANSWER': Config({'TYPE': 'AnswerAction'}), 
                                    'HIGHTOLOWINFERENCE': Config({'TYPE': 'MoveHighToLowActionInference'}), 
                                    'HIGHTOLOWEVAL': Config({'TYPE': 'MoveHighToLowActionEval'}), 
                                    'HIGHTOLOW': Config({'TYPE': 'MoveHighToLowAction'})
                                    }
                                ), 
                            'POINTGOAL_SENSOR': 
                                Config(
                                    {
                                        'TYPE': 'PointGoalSensor', 
                                        'GOAL_FORMAT': 'POLAR', 
                                        'DIMENSIONALITY': 2
                                        }
                                    ), 
                            'POINTGOAL_WITH_GPS_COMPASS_SENSOR': 
                                Config(
                                    {
                                    'TYPE': 'PointGoalWithGPSCompassSensor', 
                                    'GOAL_FORMAT': 'POLAR', 
                                    'DIMENSIONALITY': 2
                                    }
                                    ), 
                            'OBJECTGOAL_SENSOR': 
                                Config(
                                    {
                                        'TYPE': 'ObjectGoalSensor', 
                                        'GOAL_SPEC': 'TASK_CATEGORY_ID', 
                                        'GOAL_SPEC_MAX_VAL': 50
                                        }
                                    ), 
                            'IMAGEGOAL_SENSOR': 
                                Config(
                                    {
                                        'TYPE': 'ImageGoalSensor'
                                        }
                                    ), 
                            'HEADING_SENSOR': 
                                Config(
                                    {
                                        'TYPE': 'HeadingSensor'
                                        }
                                    ), 
                            'COMPASS_SENSOR': 
                                Config(
                                    {
                                        'TYPE': 'CompassSensor'
                                        }
                                    ), 
                            'GPS_SENSOR': 
                                Config(
                                    {
                                        'TYPE': 'GPSSensor', 
                                        'DIMENSIONALITY': 2
                                        }
                                    ), 
                            'PROXIMITY_SENSOR': 
                                Config(
                                    {
                                        'TYPE': 'ProximitySensor', 
                                        'MAX_DETECTION_RADIUS': 2.0
                                        }
                                    ), 
                            'SUCCESS': 
                                Config(
                                    {
                                        'TYPE': 'Success', 
                                        'SUCCESS_DISTANCE': 0.2
                                        }
                                    ), 
                            'SPL': Config({'TYPE': 'SPL'}), 
                            'SOFT_SPL': Config({'TYPE': 'SoftSPL'}), 
                            'TOP_DOWN_MAP': 
                                Config(
                                    {
                                        'TYPE': 'TopDownMap', 
                                        'MAX_EPISODE_STEPS': 1000, 
                                        'MAP_PADDING': 3, 
                                        'MAP_RESOLUTION': 1024, 
                                        'DRAW_SOURCE': True, 
                                        'DRAW_BORDER': True, 
                                        'DRAW_SHORTEST_PATH': True, 
                                        'FOG_OF_WAR': Config({'DRAW': True, 'VISIBILITY_DIST': 5.0, 'FOV': 90}), 
                                        'DRAW_VIEW_POINTS': True, 'DRAW_GOAL_POSITIONS': True, 'DRAW_GOAL_AABBS': True
                                        }
                                    ), 
                            'COLLISIONS': Config({'TYPE': 'Collisions'}), 
                            'QUESTION_SENSOR': Config({'TYPE': 'QuestionSensor'}), 
                            'CORRECT_ANSWER': Config({'TYPE': 'CorrectAnswer'}), 
                            'EPISODE_INFO': Config({'TYPE': 'EpisodeInfo'}), 
                            'INSTRUCTION_SENSOR': Config({'TYPE': 'InstructionSensor'}), 
                            'INSTRUCTION_SENSOR_UUID': 'rxr_instruction', 
                            'DISTANCE_TO_GOAL': Config({'TYPE': 'DistanceToGoal', 'DISTANCE_TO': 'POINT'}), 
                            'ANSWER_ACCURACY': Config({'TYPE': 'AnswerAccuracy'}), 
                            'GLOBAL_GPS_SENSOR': Config({'TYPE': 'GlobalGPSSensor', 'DIMENSIONALITY': 3}), 
                            'OREINTATION_SENSOR': Config({'TYPE': 'OrienSensor'}), 
                            'RXR_INSTRUCTION_SENSOR': 
                                Config(
                                    {
                                        'TYPE': 'RxRInstructionSensor', 
                                        'features_path': 'data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{id:06}_{lang}_text_features.npz', 
                                        'max_text_len': 512
                                        }
                                    ), 
                            'SHORTEST_PATH_SENSOR': 
                                Config(
                                    {
                                        'TYPE': 'ShortestPathSensor', 
                                        'GOAL_RADIUS': 0.5, 
                                        'USE_ORIGINAL_FOLLOWER': False
                                        }
                                    ), 
                            'VLN_ORACLE_PROGRESS_SENSOR': Config({'TYPE': 'VLNOracleProgressSensor'}), 
                            'NDTW': 
                                Config(
                                    {
                                        'TYPE': 'NDTW', 
                                        'SPLIT': 'val_seen', 
                                        'FDTW': True, 
                                        'GT_PATH': 'data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json', 
                                        'SUCCESS_DISTANCE': 3.0
                                        }
                                    ), 
                            'SDTW': Config({'TYPE': 'SDTW'}), 
                            'PATH_LENGTH': Config({'TYPE': 'PathLength'}), 
                            'ORACLE_NAVIGATION_ERROR': Config({'TYPE': 'OracleNavigationError'}), 
                            'ORACLE_SUCCESS': Config({'TYPE': 'OracleSuccess', 'SUCCESS_DISTANCE': 3.0}), 
                            'ORACLE_SPL': Config({'TYPE': 'OracleSPL'}), 
                            'STEPS_TAKEN': Config({'TYPE': 'StepsTaken'}), 
                            'POSITION': Config({'TYPE': 'Position'}), 
                            'POSITION_INFER': Config({'TYPE': 'PositionInfer'}), 
                            'TOP_DOWN_MAP_VLNCE': 
                                Config(
                                    {
                                        'TYPE': 'TopDownMapVLNCE', 
                                        'MAX_EPISODE_STEPS': 1000, 
                                        'MAP_RESOLUTION': 512, 
                                        'DRAW_SOURCE_AND_TARGET': True, 
                                        'DRAW_BORDER': False, 
                                        'DRAW_SHORTEST_PATH': False, 
                                        'DRAW_REFERENCE_PATH': False, 
                                        'DRAW_FIXED_WAYPOINTS': False, 
                                        'DRAW_MP3D_AGENT_PATH': False, 
                                        'GRAPHS_FILE': 'data/connectivity_graphs.pkl', 
                                        'FOG_OF_WAR': Config({'DRAW': False, 'FOV': 79, 'VISIBILITY_DIST': 5.0})
                                        }
                                    )
                        }), 
                'SIMULATOR': 
                    Config(
                        {
                            'TYPE': 'Sim-v0', 
                            'ACTION_SPACE_CONFIG': 'v1', 
                            'FORWARD_STEP_SIZE': 0.25, 
                            'SCENE': 'data/scene_datasets/habitat-test-scenes/van-gogh-room.glb', 
                            'SEED': 100, 
                            'TURN_ANGLE': 30, 
                            'TILT_ANGLE': 30, 
                            'DEFAULT_AGENT_ID': 0, 
                            'RGB_SENSOR': 
                                Config(
                                    {
                                        'HEIGHT': 480, 
                                        'WIDTH': 640, 
                                        'HFOV': 79, 
                                        'POSITION': [0, 0.88, 0], 
                                        'ORIENTATION': [0.0, 0.0, 0.0], 
                                        'TYPE': 'HabitatSimRGBSensor'
                                        }
                                    ), 
                            'DEPTH_SENSOR': 
                                Config(
                                    {
                                        'HEIGHT': 480, 
                                        'WIDTH': 640, 
                                        'HFOV': 79, 
                                        'POSITION': [0, 0.88, 0], 
                                        'ORIENTATION': [0.0, 0.0, 0.0], 
                                        'TYPE': 'HabitatSimDepthSensor', 
                                        'MIN_DEPTH': 0.5, 
                                        'MAX_DEPTH': 5.0, 
                                        'NORMALIZE_DEPTH': True
                                        }
                                    ), 
                            'SEMANTIC_SENSOR': 
                                Config(
                                    {
                                        'HEIGHT': 480, 
                                        'WIDTH': 640, 
                                        'HFOV': 79, 
                                        'POSITION': [0, 0.88, 0], 
                                        'ORIENTATION': [0.0, 0.0, 0.0], 
                                        'TYPE': 'HabitatSimSemanticSensor'
                                        }
                                    ), 
                            'AGENT_0': 
                                Config(
                                    {
                                        'HEIGHT': 0.88, 
                                        'RADIUS': 0.18, 
                                        'MASS': 32.0, 
                                        'LINEAR_ACCELERATION': 20.0, 
                                        'ANGULAR_ACCELERATION': 12.56, 
                                        'LINEAR_FRICTION': 0.5, 
                                        'ANGULAR_FRICTION': 1.0, 
                                        'COEFFICIENT_OF_RESTITUTION': 0.0, 
                                        'SENSORS': ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR'], 
                                        'IS_SET_START_STATE': False, 
                                        'START_POSITION': [0, 0, 0], 
                                        'START_ROTATION': [0, 0, 0, 1]
                                        }
                                    ), 
                            'AGENTS': ['AGENT_0'], 
                            'HABITAT_SIM_V0': 
                                Config(
                                    {
                                        'GPU_DEVICE_ID': 0, 
                                        'GPU_GPU': False, 
                                        'ALLOW_SLIDING': True, 
                                        'ENABLE_PHYSICS': False, 
                                        'PHYSICS_CONFIG_FILE': './data/default.physics_config.json'
                                        }
                                    )
                        }), 
                'PYROBOT': 
                    Config(
                        {
                            'ROBOTS': ['locobot'], 
                            'ROBOT': 'locobot', 
                            'SENSORS': ['RGB_SENSOR', 'DEPTH_SENSOR', 'BUMP_SENSOR'], 
                            'BASE_CONTROLLER': 'proportional', 
                            'BASE_PLANNER': 'none', 
                            'RGB_SENSOR': 
                                Config(
                                    {
                                        'HEIGHT': 480, 
                                        'WIDTH': 640, 
                                        'TYPE': 'PyRobotRGBSensor', 
                                        'CENTER_CROP': False
                                        }
                                    ), 
                            'DEPTH_SENSOR': 
                                Config(
                                    {
                                        'HEIGHT': 480, 
                                        'WIDTH': 640, 
                                        'TYPE': 'PyRobotDepthSensor', 
                                        'MIN_DEPTH': 0.0, 
                                        'MAX_DEPTH': 5.0, 
                                        'NORMALIZE_DEPTH': True, 
                                        'CENTER_CROP': False
                                        }
                                    ), 
                            'BUMP_SENSOR': Config({'TYPE': 'PyRobotBumpSensor'}), 
                            'LOCOBOT': Config(
                                {
                                    'ACTIONS': ['BASE_ACTIONS', 'CAMERA_ACTIONS'], 
                                    'BASE_ACTIONS': ['go_to_relative', 'go_to_absolute'], 
                                    'CAMERA_ACTIONS': ['set_pan', 'set_tilt', 'set_pan_tilt']
                                    }
                                )
                        }
                    ), 
                'DATASET': 
                    Config(
                        {
                            'TYPE': 'VLN-CE-v1', 
                            'SPLIT': 'val_unseen', 
                            'SCENES_DIR': 'data/scene_datasets/mp3d/', 
                            'CONTENT_SCENES': ['*'], 
                            'DATA_PATH': 'data/datasets/VLNCE/{split}/{split}.json.gz', 
                            'ROLES': ['guide'], 
                            'LANGUAGES': ['*'], 
                            'EPISODES_ALLOWED': None, 
                            'EPISODES_DIR': 'data/datasets/VLNCE/{split}/'
                            }
                        ), 
                'BASE_TASK_CONFIG_PATH': 
                    'envs/habitat/configs/tasks/vln_mp3d.yaml'
            }), 
        'CMD_TRAILING_OPTS': [], 
        'TRAINER_NAME': 'dagger', 
        'ENV_NAME': 'VLNCEDaggerEnv', 
        'TORCH_GPU_ID': 0, 
        'VIDEO_OPTION': [], 
        'TENSORBOARD_DIR': 'data/tensorboard_dirs/debug', 
        'VIDEO_DIR': 'videos/debug', 
        'EVAL_CKPT_PATH_DIR': 'data/checkpoints', 
        'NUM_ENVIRONMENTS': 16, 
        'NUM_PROCESSES': -1, 
        'SENSORS': ['RGB_SENSOR', 'DEPTH_SENSOR'], 
        'CHECKPOINT_FOLDER': 'data/checkpoints', 
        'NUM_UPDATES': 10000, 
        'NUM_CHECKPOINTS': 10, 
        'CHECKPOINT_INTERVAL': -1, 
        'TOTAL_NUM_STEPS': -1.0, 
        'LOG_INTERVAL': 10, 
        'LOG_FILE': 'train.log', 
        'FORCE_BLIND_POLICY': False, 
        'VERBOSE': True, 
        'EVAL': Config(
            {
                'SPLIT': 'val_seen', 
                'USE_CKPT_CONFIG': True, 
                'EPISODE_COUNT': -1, 
                'LANGUAGES': ['en-US', 'en-IN'], 
                'SAMPLE': False, 
                'SAVE_RESULTS': True, 
                'EVAL_NONLEARNING': False, 
                'NONLEARNING': Config({'AGENT': 'RandomAgent'})
                }), 
        'RL': Config(
            {
                'REWARD_MEASURE': 'distance_to_goal', 
                'SUCCESS_MEASURE': 'spl', 
                'SUCCESS_REWARD': 2.5, 
                'SLACK_REWARD': -0.01,
                'POLICY': Config(
                    {
                        'name': 'PointNavResNetPolicy', 
                        'OBS_TRANSFORMS': Config(
                            {
                                'ENABLED_TRANSFORMS': ('CenterCropperPerSensor',), 
                                'CENTER_CROPPER': Config({'HEIGHT': 256, 'WIDTH': 256}), 
                                'RESIZE_SHORTEST_EDGE': Config({'SIZE': 256}), 
                                'CUBE2EQ': Config({'HEIGHT': 256, 'WIDTH': 512, 'SENSOR_UUIDS': []}), 
                                'CUBE2FISH': Config({'HEIGHT': 256, 'WIDTH': 256, 'FOV': 180, 'PARAMS': (0.2, 0.2, 0.2), 'SENSOR_UUIDS': []}), 
                                'EQ2CUBE': Config({'HEIGHT': 256, 'WIDTH': 256, 'SENSOR_UUIDS': []}), 
                                'CENTER_CROPPER_PER_SENSOR': Config({'SENSOR_CROPS': [('rgb', (224, 224)), ('depth', (256, 256))]}), 
                                'RESIZER_PER_SENSOR': Config({'SIZES': [('rgb', (224, 298)), ('depth', (256, 341))]})
                                }
                            )
                        }
                    ), 
                'PPO': Config(
                    {
                        'clip_param': 0.2, 
                        'ppo_epoch': 4, 
                        'num_mini_batch': 2, 
                        'value_loss_coef': 0.5, 
                        'entropy_coef': 0.01, 
                        'lr': 0.00025, 
                        'eps': 1e-05, 
                        'max_grad_norm': 0.5, 
                        'num_steps': 5, 
                        'use_gae': True, 
                        'use_linear_lr_decay': False, 
                        'use_linear_clip_decay': False, 
                        'gamma': 0.99, 'tau': 0.95, 
                        'reward_window_size': 50, 
                        'use_normalized_advantage': False, 
                        'hidden_size': 512, 
                        'use_double_buffered_sampler': False
                        }
                    ), 
                'DDPPO': Config(
                    {
                        'sync_frac': 0.6, 
                        'distrib_backend': 'GLOO', 
                        'rnn_type': 'GRU', 
                        'num_recurrent_layers': 1, 
                        'backbone': 'resnet18', 
                        'pretrained_weights': 'data/ddppo-models/gibson-2plus-resnet50.pth', 
                        'pretrained': False, 
                        'pretrained_encoder': False, 
                        'train_encoder': True, 
                        'reset_critic': True, 
                        'force_distributed': False
                        }
                    )
            }), 
        'ORBSLAM2': Config(
            {
                'SLAM_VOCAB_PATH': 'habitat_baselines/slambased/data/ORBvoc.txt', 
                'SLAM_SETTINGS_PATH': 'habitat_baselines/slambased/data/mp3d3_small1k.yaml', 
                'MAP_CELL_SIZE': 0.1, 
                'MAP_SIZE': 40, 
                'CAMERA_HEIGHT': 1.25, 
                'BETA': 100, 
                'H_OBSTACLE_MIN': 0.375, 
                'H_OBSTACLE_MAX': 1.25, 
                'D_OBSTACLE_MIN': 0.1, 
                'D_OBSTACLE_MAX': 4.0, 
                'PREPROCESS_MAP': True, 
                'MIN_PTS_IN_OBSTACLE': 320.0, 
                'ANGLE_TH': 0.2617993877991494, 
                'DIST_REACHED_TH': 0.15, 
                'NEXT_WAYPOINT_TH': 0.5, 
                'NUM_ACTIONS': 3, 
                'DIST_TO_STOP': 0.05, 
                'PLANNER_MAX_STEPS': 500, 
                'DEPTH_DENORM': 10.0
                }), 
        'PROFILING': Config(
            {
                'CAPTURE_START_STEP': -1, 
                'NUM_STEPS_TO_CAPTURE': -1
                }), 
        'SIMULATOR_GPU_IDS': [0], 
        'RESULTS_DIR': 'data/checkpoints/pretrained/evals', 
        'INFERENCE': Config(
            {
                'SPLIT': 'test', 
                'LANGUAGES': ['en-US', 'en-IN'], 
                'SAMPLE': False, 
                'USE_CKPT_CONFIG': True, 
                'CKPT_PATH': 'data/checkpoints/CMA_PM_DA_Aug.pth', 
                'PREDICTIONS_FILE': 'predictions.json', 
                'INFERENCE_NONLEARNING': False, 
                'NONLEARNING': Config({'AGENT': 'RandomAgent'}), 
                'FORMAT': 'rxr'
                }
            ), 
        'IL': Config(
            {
                'lr': 0.00025, 
                'batch_size': 5, 
                'epochs': 4, 
                'use_iw': True, 
                'inflection_weight_coef': 3.2, 
                'waypoint_aug': False, 
                'load_from_ckpt': False, 
                'ckpt_to_load': 'data/checkpoints/ckpt.0.pth', 
                'is_requeue': False, 
                'RECOLLECT_TRAINER': Config(
                    {
                        'preload_trajectories_file': True, 
                        'trajectories_file': 'data/trajectories_dirs/debug/trajectories.json.gz', 
                        'max_traj_len': -1, 
                        'effective_batch_size': -1, 
                        'preload_size': 30, 
                        'use_iw': True, 
                        'gt_file': 'data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz'
                        }
                    ), 
                'DAGGER': Config(
                    {
                        'iterations': 10, 
                        'update_size': 5000, 
                        'p': 0.75, 
                        'expert_policy_sensor': 'SHORTEST_PATH_SENSOR', 
                        'expert_policy_sensor_uuid': 'shortest_path_sensor', 
                        'load_space': False, 
                        'lmdb_map_size': 1000000000000.0, 
                        'lmdb_fp16': False, 
                        'lmdb_commit_frequency': 500, 
                        'preload_lmdb_features': False, 
                        'lmdb_features_dir': 'data/trajectories_dirs/debug/trajectories.lmdb'
                        }
                    )
                }
            ), 
        'MODEL': Config(
            {
                'policy_name': 'CMAPolicy', 
                'ablate_depth': False, 
                'ablate_rgb': False, 
                'ablate_instruction': False, 
                'INSTRUCTION_ENCODER': Config(
                    {
                        'sensor_uuid': 'instruction', 
                        'vocab_size': 2504, 
                        'use_pretrained_embeddings': True, 
                        'embedding_file': 'data/datasets/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz', 
                        'dataset_vocab': 'data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz', 
                        'fine_tune_embeddings': False, 
                        'embedding_size': 50, 
                        'hidden_size': 128, 
                        'rnn_type': 'LSTM', 
                        'final_state_only': True, 
                        'bidirectional': False
                        }
                    ), 
                'spatial_output': True, 
                'RGB_ENCODER': Config(
                    {
                        'cnn_type': 'TorchVisionResNet50', 
                        'output_size': 256
                        }
                    ), 
                'DEPTH_ENCODER': Config(
                    {
                        'cnn_type': 'VlnResnetDepthEncoder', 
                        'output_size': 128, 
                        'backbone': 'resnet50', 
                        'ddppo_checkpoint': 'data/ddppo-models/gibson-2plus-resnet50.pth'
                        }
                    ), 
                'STATE_ENCODER': Config(
                    {
                        'hidden_size': 512, 
                        'rnn_type': 'GRU'
                        }
                    ), 
                'SEQ2SEQ': Config(
                    {'use_prev_action': False}), 
                'PROGRESS_MONITOR': Config({'use': False, 'alpha': 1.0})
                }
            ), 
        'ENVIRONMENT': Config({'MAX_EPISODE_STEPS': 500}), 
        'SIMULATOR': Config(
            {
                'TURN_ANGLE': 30, 
                'TILT_ANGLE': 30, 
                'ACTION_SPACE_CONFIG': 'v1', 
                'AGENT_0': Config(
                    {
                        'SENSORS': ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR'], 
                        'HEIGHT': 0.88, 'RADIUS': 0.18
                        }
                    ), 
                'HABITAT_SIM_V0': Config({'GPU_DEVICE_ID': 0, 'ALLOW_SLIDING': True}), 
                'SEMANTIC_SENSOR': Config({'WIDTH': 640, 'HEIGHT': 480, 'HFOV': 79, 'POSITION': [0, 0.88, 0]}), 
                'RGB_SENSOR': Config({'WIDTH': 640, 'HEIGHT': 480, 'HFOV': 79, 'POSITION': [0, 0.88, 0]}), 
                'DEPTH_SENSOR': Config({'WIDTH': 640, 'HEIGHT': 480, 'HFOV': 79, 'MIN_DEPTH': 0.5, 'MAX_DEPTH': 5.0, 'POSITION': [0, 0.88, 0]})
                }
            ), 
        'TASK': Config(
            {
                'TYPE': 'ObjectNav-v1', 
                'POSSIBLE_ACTIONS': ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN'], 
                'SENSORS': ['GPS_SENSOR', 'COMPASS_SENSOR'], 
                'MEASUREMENTS': ['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL'], 
                'SUCCESS': Config({'SUCCESS_DISTANCE': 0.2})
                }
            ), 
        'DATASET': Config(
            {
                'TYPE': 'VLN-CE-v1', 
                'SPLIT': 'val_unseen', 
                'DATA_PATH': 'data/datasets/VLNCE/{split}/{split}.json.gz', 
                'EPISODES_DIR': 'data/datasets/VLNCE/{split}/', 
                'SCENES_DIR': 'data/scene_datasets/mp3d/'
                }
            )
        }
    )