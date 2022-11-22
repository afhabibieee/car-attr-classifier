import argparse
import configs

def parse_args():
    parser = argparse.ArgumentParser(description='training few-shot model')

    parser.add_argument('--img_size',       type=int,   default=configs.IMAGE_SIZE,             help='set image size')
    parser.add_argument('--n_way',          type=int,   default=configs.N_WAY,                  help='class num to classify')
    parser.add_argument('--n_shot',         type=int,   default=configs.N_SHOT,                 help='number of labeled data in each class')
    parser.add_argument('--n_query',        type=int,   default=configs.N_SHOT,                 help='number of query image each task')
    parser.add_argument('--n_train_task',   type=int,   default=configs.N_TRAINING_EPISODES,    help='number of task episodes during meta training')
    parser.add_argument('--n_val_task',     type=int,   default=configs.N_TRAINING_EPISODES,    help='number of task for meta validation')
    parser.add_argument('--n_worker',       type=int,   default=configs.N_WORKERS,              help='number of concurrent processing')
    parser.add_argument('--epochs',         type=int,   default=configs.EPOCHS,                 help='number of passes of the entire training')
    parser.add_argument('--lr',             type=float, default=configs.LEARNING_RATE,          help='step size at each iteration while moving toward a minimum of a loss function')
    parser.add_argument('--wd',             type=float, default=configs.WEIGHT_DECAY,           help='decreasing the learning rate during training')
    parser.add_argument('--backbone_name',  default='convnet',                      help='convnet/effinet/resnet')
    parser.add_argument('--variant_depth',  default=None,                           help='depth variation of the selected backbone (ex: \'s\' for effinet)')

    return parser.parse_args()