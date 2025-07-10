exp_configuration={
    578:{
         ####### dataset setting #######
        'dataset':'ImageNet',
        'image_size':224,
        'class_num':1000, 
        'eval_path':'eval_data/clean_data/',
        'tg_imgs_path':'eval_data/tg_data/',
        ####### attack setting #######
        'targeted':True,
        'epsilon':16,
        'lr':1.6,  # step size
        'steps':100,  # max_iterations
        'di_prob':0.7,  # prob for DI and RE
        'resize_rate':0.9,  # resize rate for DI
        'DI_type':'g',
        'kn_name':'gaussian',  # kernel name
        'ks':5,  # kernel size for TI
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'visualize':False,
        'mu':1,  # norm for grad
        'TI':False,
        'DI':True,
        ####################################
        # LMM
        'Logit':'TopkLoss',  # 'LogitLoss'  'CELoss' 'TopkLoss'   'Topkdis'  
        'alpha':1,
        'p':2,
        'noise_std':0.2,
        'weight':1,
        ####################################
        # ITDS
        'itds_proportion':0.5,  # beta
        'mimix':False,
        'signmix':False,
        'norm':False,
        'untg_steps':5,
        'tg_samples':3,  # n
        'top_k':10,
        'tg_imgs':'all',    #  'all'  'val'
        ####################################
        # Admix
        'admix_portion':0.2, # 'Admix portion for the mixed image'
        'num_mix_samples':3, #  'Number of randomly sampled images'
        'extract_path':'eval_data/ImageNet_crt2000',
        'beta':1.5,
        # 'samples':200,
         # ODI params
        'shininess':0.5,
        'source_3d_models':['pack','pillow','book'],
        'rand_elev':(-35,35),
        'rand_azim':(-35,35),
        'rand_angle':(-35,35),
        'min_dist':0.8, 'rand_dist':0.4,
        'light_location':[0.0, 0.0,4.0],
        'rand_light_location':4,
        'rand_ambient_color':0.3,
        'ambient_color':0.6,
        'rand_diffuse_color':0.5,
        'diffuse_color':0.0,
        'specular_color':0.0,
        'background_type':'random_pixel',
        'texture_type':'random_solid',
        #####################################
        # CFM
        'attack_type': 'CDM',
        'mixed_image_type_feature':'C', # 'C': Clean image / 'A': Current Batch image
        'shuffle_image_feature':'SelfShuffle', # 'None': Without shuffle, 'SelfShuffle': With shuffle
        'blending_mode_feature':'M', # 'M': Convex interpolation, 'A': Addition
        'mix_lower_bound_feature':0., # mix ratio is sampled from [mix_lower_bound_feature, mix_upper_bound_feature]
        'mix_upper_bound_feature':0.75,
        'mix_prob':0.1,
        'divisor':4,
        'channelwise':True,
        'mixup_layer':'conv_linear_include_last', 
        'every_layer':False,
        #####################################
        # Randmix
        'fix_mode_input':'M',   
        'k':5, # topk for faraway
        'zeta':0.5, # mix for randmix
        'nor':2,
        'drop_ratio':0.1,
        #####################################
    }
