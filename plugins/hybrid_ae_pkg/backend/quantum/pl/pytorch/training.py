from typing import Optional, List

import torch
from torch.utils.data import TensorDataset, DataLoader


def training_loop(
    model: torch.nn.Module,
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    train_label: Optional[torch.Tensor],
    test_input: Optional[torch.Tensor],
    test_target: Optional[torch.Tensor],
    test_label: Optional[torch.Tensor],
    optis: List,
    steps: int,
    batch_size: int,
):
    loss_func = torch.nn.MSELoss()
    train_dataset = TensorDataset(train_input, train_target)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data_available = test_input is not None and test_target is not None

    if test_data_available:
        test_dataset = TensorDataset(test_input, test_target)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    batch_cnt_overall = 0

    for i in range(steps):
        # training on the training dataset
        error_sum = 0
        batch_cnt = 0

        for batch_input, batch_target in train_dataloader:
            for opt in optis:
                opt.zero_grad()

            loss = loss_func(model(batch_input), batch_target)
            loss.backward()

            for opt in optis:
                opt.step()

            error_sum += loss.item()

            batch_cnt += 1
            batch_cnt_overall += 1

        error_mean = error_sum / batch_cnt
        print("Step:", i, "Training MSE:", error_mean)

        if test_data_available:
            # calculating the error on the test dataset
            error_sum = 0
            batch_cnt = 0

            for batch_input, batch_target in test_dataloader:
                with torch.no_grad():
                    loss = loss_func(model(batch_input), batch_target)
                    error_sum += loss.item()
                    batch_cnt += 1

            error_mean = error_sum / batch_cnt
            print("Step:", i, "Test MSE:", error_mean)

        # test if the model has an "embed" function and use it to generate embeddings and save them
        embed_func = getattr(model, "embed", None)

        if callable(embed_func):
            if (
                train_label is not None
                and test_input is not None
                and test_label is not None
            ):
                embeddings = torch.cat(
                    (model.embed(train_input), model.embed(test_input)), dim=0
                )
                labels = torch.cat((train_label, test_label), dim=0)
