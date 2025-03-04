sweep_config = {
    'pretrained_path': './weights',
    'batch_size': 128,
    'lr': 1e-4,
    'warmup': 1,
    'max_epochs': 1226446, 
    'extract_layers': [11],
    'function_layers': 'mean',
    'pool': None,
    'dim_output': 512,
    'temperature': 1.0,
    'without_context': True,
    'margin': 0.5,
    'p': 2,
    'eps': 1e-6,
    }

spot_config = {
    'pretrained_path': None,
    'dim_feedforward': 1024,
    'nheads': 16,
    'nlayers': 12,
    'dropout': 0.0,
    'dim_model': 512,
    'batch_first': True,
    'n_tokens': 20340, 
    'context_length': 1500,
    'autoregressive': False,
    'pool': None,
    'learnable_pe': True,
    }

visual_config = {
    'pretrained_path': "/home/hmh/weights/conch/pytorch_model.bin",
    'model_name': 'conch',
    }