import os
import time
from typing import Optional, List, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .quantum.qiskit.hybrid import ClassicalAutoEncoder


# TODO: dont train in the 0th epoch
def train_nn(
    net: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    criterion: nn.Module,
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[ReduceLROnPlateau] = None,
    input_callback: Callable[[torch.Tensor], torch.Tensor] = None,
    end_of_epoch_callbacks: List[
        Callable[[int, float, float, nn.Module, Optimizer, ReduceLROnPlateau], None]
    ] = None,
    add_loss_callbacks: List[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
):
    if optimizer is None:
        optimizer = optim.Adam(net.parameters())

    if end_of_epoch_callbacks is None:
        end_of_epoch_callbacks = []

    start_epoch = 0

    for i in range(start_epoch, epochs):
        loss_sum = 0
        start_time = time.time()

        # training for one epoch
        for j, batch in enumerate(iter(train_loader)):
            optimizer.zero_grad()

            if isinstance(batch, list):
                batch = batch[0]

            input_batch: torch.Tensor = batch

            if input_callback is not None:
                input_batch = input_callback(input_batch)

            if i == 0:
                # no training in the 0th epoch
                with torch.no_grad():
                    output: torch.Tensor = net(input_batch)
            else:
                output: torch.Tensor = net(input_batch)

            loss = criterion(output, batch)

            if add_loss_callbacks is not None:
                if i == 0:
                    # no training in the 0th epoch
                    with torch.no_grad():
                        for callback in add_loss_callbacks:
                            loss += callback(net, input_batch)
                else:
                    for callback in add_loss_callbacks:
                        loss += callback(net, input_batch)

            loss_sum += loss.item()

            if i != 0:
                # no training in the 0th epoch
                loss.backward()

                for param in net.parameters():
                    if torch.isnan(param.grad).any():
                        test = 0

                optimizer.step()

            print("batch complete, loss: " + str(loss.item()))

        end_time = time.time()
        # walltime += end_time - start_time
        print("training time: ", end_time - start_time)

        train_avg_loss = loss_sum / len(train_loader)

        print("training loss: ", train_avg_loss, flush=True)

        loss_sum = 0
        start_time = time.time()

        # calculating the validation error
        val_avg_loss = 0

        if val_loader is not None:
            for batch in iter(val_loader):
                with torch.no_grad():
                    output = net(batch)

                    loss = criterion(output, batch)
                    loss_sum += loss.item()

            end_time = time.time()

            val_avg_loss = loss_sum / len(val_loader)

            if lr_scheduler is not None:
                lr_scheduler.step(val_avg_loss)
                print(lr_scheduler.state_dict())

            print("validation loss: ", val_avg_loss, flush=True)
            print("validation loss calculation time: ", end_time - start_time)
            print(flush=True)

        for callback in end_of_epoch_callbacks:
            callback(i, train_avg_loss, val_avg_loss, net, optimizer, lr_scheduler)


def load_model_parameters(
    net: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: Optional[ReduceLROnPlateau],
    save_path: str,
) -> int:
    files = sorted(os.listdir(save_path))

    newest_file = files[-1]
    checkpoint = torch.load(os.path.join(save_path, newest_file))
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if lr_scheduler is not None and "lr_scheduler_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_dict"])

    start_epoch = checkpoint["epoch"] + 1

    return start_epoch


def add_noise_to_input_callback(factor: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def callback(batch: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(batch)  # sample random values uniformly from [0, 1)
        noise = (
            noise * 2 * factor
        ) - factor  # scale the noise to be in [-factor, factor]

        return torch.tensor(batch) + noise

    return callback


def first_order_contractive_loss(
    cae: ClassicalAutoEncoder, factor: float
) -> Callable[[nn.Module, torch.Tensor], torch.Tensor]:
    def callback(net: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        # gradient calculation has to be enabled for the jacobian
        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(
                lambda x: cae.get_embeddings(x, True), batch, create_graph=True
            )

        frobenius: torch.Tensor = ((jacobian * jacobian).sum() + 1e-8).sqrt()

        return factor * frobenius

    return callback


def second_order_contractive_loss(
    cae: ClassicalAutoEncoder, factor: float, corrupted_num: int, noise_variance: float
) -> Callable[[nn.Module, torch.Tensor], torch.Tensor]:
    def callback(net: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            jacobian_x = torch.autograd.functional.jacobian(
                lambda x: cae.get_embeddings(x, True), batch, create_graph=True
            )
            frobenius_sum = 0

            for i in range(corrupted_num):
                noise = torch.randn_like(batch) * noise_variance
                jacobian_sample = torch.autograd.functional.jacobian(
                    lambda x: cae.get_embeddings(x, True),
                    batch + noise,
                    create_graph=True,
                )

                difference = jacobian_x - jacobian_sample
                frobenius = (difference * difference).sum()
                frobenius_sum += frobenius

            frobenius = frobenius_sum / corrupted_num

            return factor * frobenius

    return callback
