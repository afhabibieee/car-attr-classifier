import time, torch
import backbones
from tqdm import tqdm
from configs import DEVICE

import mlflow

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision

def get_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id

def get_last_run_id(name):
    exp = get_experiment_id(name)
    client = mlflow.MlflowClient()
    runs = client.search_runs(experiment_ids=exp)
    if len(runs) == 0:
        return None
    last_run = runs[0]
    return last_run.info.run_id

def get_backbone(backbone_name, variant_depth, dropout):
    if backbone_name=='effinet':
        return backbones.effinet(
            variant=variant_depth,
            dropout=float(dropout)
        )
    elif backbone_name=='resnet':
        return backbones.resnet(
            variant=variant_depth,
            dropout=float(dropout)
        )
    elif backbone_name=='convnet':
        return backbones.ConvNet(
            depth=int(variant_depth),
            dropout=float(dropout)
        )
    else:
        ValueError('{} as the selected backbone was not found'.format(backbone_name))

def compute_prototypes(support_features, support_labels):
    n_way = len(torch.unique(support_labels))
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )

def cosine_distance_to_prototypes(samples, prototypes):
    """
    Compute prediction logits from their cosine distance to support set prototypes.
    """
    return (
        torch.nn.functional.normalize(samples, dim=1)
        @ torch.nn.functional.normalize(prototypes, dim=1).T
    )

def rectify_prototypes(prototypes, support_features, support_labels, query_features):
    """
    Updates prototypes with label propagation and feature shifting.
    """
    n_classes = support_labels.unique().size(0)
    one_hot_support_labels = torch.nn.functional.one_hot(support_labels, n_classes)

    average_support_query_shift = support_features.mean(0, keepdim=True) - query_features.mean(0, keepdim=True)
    query_features = query_features + average_support_query_shift

    support_logits = cosine_distance_to_prototypes(support_features, prototypes).exp()
    query_logits = cosine_distance_to_prototypes(query_features, prototypes).exp()

    one_hot_query_prediction = torch.nn.functional.one_hot(
        query_logits.argmax(-1), n_classes
    )

    normalization_vector = (
        (one_hot_support_labels * support_logits).sum(0)
        + (one_hot_query_prediction * query_logits).sum(0)
    ).unsqueeze(0)  # [1, K]

    support_reweighting = (
        one_hot_support_labels * support_logits
    ) / normalization_vector  # [shot_s, K]

    query_reweighting = (
        one_hot_query_prediction * query_logits
    ) / normalization_vector  # [shot_q, K]

    prototypes = (support_reweighting * one_hot_support_labels).t().matmul(support_features) +\
        (query_reweighting * one_hot_query_prediction).t().matmul(query_features)
    
    return prototypes

def softmax_if_specified(output):
    """
    If the option is chosen when the classifier is initialized, we perform a softmax on the
    output in order to return soft probabilities.
    """
    return output.softmax(-1)
    
def train_per_epoch(model, train_loader, test_loader, criterion, optimizer):
    train_losses, test_losses = [], []
    train_correct, test_correct = 0, 0
    train_total, test_total = 0, 0
    
    model.train()
    for support_images, support_labels, query_images, query_labels, _ in tqdm(train_loader, total=len(train_loader)):
        optimizer.zero_grad()
        classification_scores, correct, total = evaluate_per_task(
            model,
            support_images.to(DEVICE),
            support_labels.to(DEVICE),
            query_images.to(DEVICE),
            query_labels.to(DEVICE)
        )
        train_loss = criterion(classification_scores, query_labels.to(DEVICE))
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())
        train_correct += correct
        train_total += total
    
    else:
        with torch.no_grad():
            model.eval()
            for support_images, support_labels, query_images, query_labels, _ in test_loader:
                classification_scores, correct, total = evaluate_per_task(
                    model,
                    support_images.to(DEVICE),
                    support_labels.to(DEVICE),
                    query_images.to(DEVICE),
                    query_labels.to(DEVICE)
                )
                test_loss = criterion(classification_scores, query_labels.to(DEVICE))

                test_losses.append(test_loss.item())
                test_correct += correct
                test_total += total

    avg_train_loss = torch.mean(torch.Tensor(train_losses).detach().data).item()
    avg_test_loss = torch.mean(torch.Tensor(test_losses).detach().data).item()
    train_acc = train_correct/train_total
    test_acc = test_correct/test_total
    
    return (
        avg_train_loss, 
        avg_test_loss,
        train_acc, 
        test_acc
    )
    
def evaluate_per_task(
    model,
    support_images,
    support_labels,
    query_images,
    query_labels
):
    classification_scores = model(
        support_images, support_labels, query_images
    )
    correct = (torch.max(classification_scores.detach().data, 1)[1] == query_labels).sum().item()
    total = len(query_labels)
    return classification_scores, correct, total 
    
def evaluate(model, data_loader, datasets):
    total_pred = 0
    correct_pred = 0
    total_recall = 0
    total_precision = 0

    num_classes = data_loader.batch_sampler.n_way
    recall = MulticlassRecall(num_classes=num_classes)
    precision = MulticlassPrecision(num_classes=num_classes)

    results = []
    results.append(
        'Ground truth,Predicted,1st prob,2nd prob,Top distance score'
    )
    times = 0
    model.eval()
    with torch.no_grad():
        for support_images, support_labels, query_images, query_labels, class_id in data_loader:
            start = time.time()
            classification_scores, correct, total = evaluate_per_task(
                model,
                support_images.to(DEVICE),
                support_labels.to(DEVICE),
                query_images.to(DEVICE),
                query_labels.to(DEVICE)
            )
            end = time.time()
            times += (end - start)
            correct_pred += correct
            total_pred += total

            top_score, pred_labels = torch.max(
                classification_scores.data, 1
            )

            total_recall += recall(pred_labels, query_labels).item()
            total_precision += precision(pred_labels, query_labels).item()
            
            first_prob, second_prob = zip(*torch.topk(classification_scores.softmax(-1).data, 2, 1)[0])
            first_prob = torch.stack(first_prob, dim=0)
            second_prob = torch.stack(second_prob, dim=0)

            for i in range(len(query_labels)):
                result = '{},{},{},{},{}'.format(
                    datasets.class_names[class_id[query_labels[i]]],
                    datasets.class_names[class_id[pred_labels[i]]],
                    first_prob[i],
                    second_prob[i],
                    top_score[i]
                )
                results.append(result)

    avg_accuracy = correct_pred / total_pred
    avg_recall = total_recall / len(data_loader)
    avg_precision = total_precision / len(data_loader)
    avg_time = times / len(data_loader)
    return avg_accuracy, avg_recall, avg_precision, results, avg_time

def precision_recall_curve(path, thresholds):
    results = pd.read_csv(path)

    y_true = ["known" if value else "unknown" for value in results['Ground truth'] == results['Predicted']]
    pred_scores = results['1st prob'].values

    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["known" if score >= threshold else "unknown" for score in pred_scores]

        precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label="known")
        recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label="known")
        
        precisions.append(precision)
        recalls.append(recall)

    f1 = 2 * ((np.array(precisions) * np.array(recalls)) / (np.array(precisions) + np.array(recalls)))

    return precisions, recalls, f1