from image_quality.train.train import train_main


if __name__ == '__main__':
    args = {}
    args['multi_gpu'] = 1
    args['gpu'] = 0

    # Choose between 'resnet50', 'densnet121', 'vgg16', 'efficientnetb0', 'efficientnetb4'
    args['backbone'] = 'resnet50'
    # args['backbone'] = 'densenet121'
    # args['backbone'] = 'vgg16'
    # args['backbone'] = 'efficientnetb0'
    # args['backbone'] = 'efficientnetb4'

    # Depending on which backbone is used, choose the corresponding ImageNet pretrained weights file, set to None is no pretrained weights to be used.
    # args['weights'] = r'..\pretrained_weights\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # args['weights'] = r'..\pretrained_weights\efficientnetb0_notop.h5'
    # args['weights'] = r'..\pretrained_weights\efficientnetb4_notop.h5'
    # args['weights'] = r'..\pretrained_weights\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    args['weights'] = r'C:\pretrained_weights_files\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # args['weights'] = r'..\pretrained_weights\densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # args['weights'] = None

    database = 'koniq' # 'spaq'
    if database == 'koniq':
        # Image and score must be provided
        args['images_scores_file'] = r'..\databases\train_val_test_koniq.pkl'
        args['image_folder'] = r'..\databases\koniq_all'
        args['n_quality_levels'] = 5 # 1

    elif database == 'spaq':
        # Image and score must be provided
        args['images_scores_file'] = r'..\databases\train_val_test_spaq.pkl'
        args['image_folder'] = r'..\databases\koniq_all'
        args['n_quality_levels'] = 1

    args['result_folder'] = r'..\results\{}_koniq_sca_{}'.format(
        database, args['backbone'])

    args['initial_epoch'] = 0

    args['lr_base'] = 1e-4/2  # 1e-4/2
    args['lr_schedule'] = True
    args['batch_size'] = 8
    args['epochs'] = 100

    args['image_aug'] = True

    args['do_finetune'] = True

    train_main(args)
