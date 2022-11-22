from configs import DAGSHUB_USER_NAME, DAGSHUB_REPO_NAME

import os, torch
from torch import optim
#from torch.optim.lr_scheduler import MultiStepLR

from fewshotdataloader import generate_loader

from protonet import PrototypicalNetworks

from utils import train_per_epoch
from utils import get_experiment_id, get_last_run_id

from io_utils import parse_args

import mlflow

mlflow.set_tracking_uri(f'https://dagshub.com/{DAGSHUB_USER_NAME}/{DAGSHUB_REPO_NAME}.mlflow')
mlflow.set_experiment('car attribute')

if __name__=='__main__':

    params = parse_args()

    mlflow.end_run()
    with mlflow.start_run(experiment_id=get_experiment_id('car attribute')):

        train_loader = generate_loader(
            'train', 
            image_size=params.img_size,
            n_way=params.n_way,
            n_shot=params.n_shot,
            n_query=params.n_query,
            n_task=params.n_train_task,
            n_workers=params.n_worker
        )
        val_loader = generate_loader(
            'val',
            image_size=params.img_size,
            n_way=params.n_way,
            n_shot=params.n_shot,
            n_query=params.n_query,
            n_task=params.n_val_task,
            n_workers=params.n_worker
        )

        model = PrototypicalNetworks(params.backbone_name, params.variant_depth)

        epochs = params.epochs
        learning_rate = params.lr
        weight_decay = params.wd

        mlflow.log_params(
            {
                'img_size':         params.img_size,
                'n_way':            params.n_way,
                'n_shot':           params.n_shot,
                'n_query':          params.n_query,
                'n_train_task':     params.n_train_task,
                'n_worker':         params.n_worker,
                'epochs':           epochs,
                'learning_rate':    learning_rate,
                'weight_decay':     weight_decay,
                'backbone':         params.backbone_name,
                'varian_depth':     params.variant_depth
            }
        )

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_state = model.state_dict()
        best_validation_accuracy = 0.0
        at_epoch = 1

        for epoch in range(1, epochs+1):
            train_loss, val_loss, train_acc, val_acc = train_per_epoch(
                model, train_loader, val_loader, criterion, optimizer
            )

            print(
                    'Epoch: {}/{}\n'.format(epoch, epochs),
                    'loss: {:.3f} - '.format(train_loss),
                    'val_loss: {:.3f} - '.format(val_loss),
                    'accuracy: {:.3f} - '.format(train_acc),
                    'val_accuracy: {:.3f}'.format(val_acc)
                )
            print('\n')

            if val_acc > best_validation_accuracy:
                at_epoch = epoch
                best_validation_accuracy = val_acc
                mlflow.pytorch.log_model(model, "model")
                print("Yeay! we found a new best model :')\n")
            
            mlflow.log_metrics(
                {
                    'train_loss':   train_loss,
                    'val_loss':     val_loss,
                    'train_acc':    train_acc,
                    'val_acc':      val_acc
                },
                step=epoch
            )