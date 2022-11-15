import torch
from tqdm import tqdm
from configs import DEVICE

def compute_prototypes(support_features, support_labels):
    n_way = len(torch.unique(support_labels))
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )

def train_per_epoch(model, train_loader, test_loader, criterion, optimizer):
    train_losses, test_losses = [], []
    train_correct, test_correct = 0, 0
    train_total, test_total = 0, 0

    model.to(DEVICE)
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

def evaluate(model, data_loader):
    total_pred = 0
    correct_pred = 0

    model.eval()
    with torch.no_grad():
        for support_images, support_labels, query_images, query_labels, _ in data_loader:
            classification_scores, correct, total = evaluate_per_task(
                model,
                support_images.to(DEVICE),
                support_labels.to(DEVICE),
                query_images.to(DEVICE),
                query_labels.to(DEVICE)
            )
            correct_pred += correct
            total_pred += total
    return correct_pred/total_pred