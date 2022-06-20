from pennylane import numpy as np

# PyTorch
import torch

import time


def digits2position(vec_of_digits, n_positions):
    """One-hot encoding of a batch of vectors."""
    return np.eye(n_positions)[vec_of_digits]


def train(
    model, X_train, Y_train, loss_fn, optimizer, num_iterations, n_classes, batch_size
):

    n_train = len(Y_train)

    # prepare data and label format
    X_train = torch.Tensor(X_train)

    Y_train_onehot = torch.from_numpy(
        digits2position(Y_train, n_classes)
    )  # [0] -> [1,0],   [1] -> [0,1]

    # TRAINING
    model.train()

    offset = 0
    for i in range(num_iterations):
        start_it_time = time.time()
        # reshuffle data when all of it was used
        if offset > n_train - 1:
            indices = np.arange(n_train)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train_onehot = Y_train_onehot[indices]
            offset = 0

        # batch data
        train_data_batch = X_train[offset : offset + batch_size]
        # train_data_batch = train_data_batch.to(device)
        train_label_batch = Y_train_onehot[offset : offset + batch_size]
        # train_label_batch = train_label_batch.to(device)

        # zero gradients
        optimizer.zero_grad()

        # get model predictions
        predictions = model(train_data_batch)

        # calculate loss
        loss = loss_fn(predictions, train_label_batch)

        # backpropagation, adjust weights
        loss.backward()
        optimizer.step()

        # accuracy
        _, predicted_class = torch.max(predictions, 1)
        _, labels = torch.max(train_label_batch, 1)
        batch_accuracy = float(torch.sum(predicted_class == labels).item()) / len(labels)

        # time
        total_it_time = time.time() - start_it_time
        minutes_it = total_it_time // 60
        seconds_it = round(total_it_time - minutes_it * 60)

        # print loss, accuracy and time of this iteration
        print(
            "Iter: {}/{} Time: {:.4f} min {:.4f} sec with loss: {:.4f} and accuracy: {:.4f} on the training data".format(
                i + 1, num_iterations, minutes_it, seconds_it, loss.item(), batch_accuracy
            ),
            end="\r",
            flush=True,
        )
        offset += batch_size


def test(model, X_test, Y_test, loss_fn, n_classes):
    # TEST
    model.eval()
    # TODO disable gradient calculation

    X_test = torch.Tensor(X_test)
    Y_test_onehot = torch.from_numpy(digits2position(Y_test, n_classes))
    Y_test = torch.from_numpy(Y_test)

    # TODO batches??

    # feed test data into network and get predictions
    predictions = model(X_test)

    # compute loss
    test_loss = loss_fn(predictions, Y_test_onehot)

    # accuracy
    _, predicted_class = torch.max(predictions, 1)
    test_accuracy = float(torch.sum(predicted_class == Y_test).item()) / len(Y_test)

    # print loss and accuracy for test data
    print("")
    print(
        "Test loss: {:.4f} and accuracy: {:.4f} on the test data".format(
            test_loss.item(), test_accuracy
        )
    )
    return test_accuracy
