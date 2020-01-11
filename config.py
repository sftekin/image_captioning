model_params = {
        'drop_prob': 0.3,
        'n_layers': 2,
        'lstm_dim': 512,
        'conv_dim': 512,
        'att_dim': 512,
        'transfer_learn': True,
        'pre_train': True,
    }

train_params = {
    'n_epoch': 50,
    'clip': 5,
    'lr': 0.008,
    'seq_len': 17,
    'eval_every': 200,
    'show_image': False
}

batch_params = {
    'batch_size': 128,
    'num_works': 0,
    'shuffle': True,
    'use_transform': True,
    "input_size": (224, 224)
}