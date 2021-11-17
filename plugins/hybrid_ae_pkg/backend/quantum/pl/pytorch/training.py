from os import PathLike
from tempfile import SpooledTemporaryFile
from typing import Optional, List, Callable, Dict, Any, Union, BinaryIO, IO

import torch
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader

from plugins.hybrid_ae_pkg.backend.helpers import save_optim_state
from qhana_plugin_runner.storage import STORE


class Checkpoint:
    def __init__(
        self,
        epoch: int,
        model_params_file: Union[str, PathLike, BinaryIO, IO[bytes]],
        optim_params_files: List[Union[str, PathLike, BinaryIO, IO[bytes]]],
    ):
        self.epoch = epoch
        self.model_params_file = model_params_file
        self.optim_params_files = optim_params_files


def training_loop(
    model: torch.nn.Module,
    shadow_model: Optional[torch.nn.Module],
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    train_label: Optional[torch.Tensor],
    test_input: Optional[torch.Tensor],
    test_target: Optional[torch.Tensor],
    test_label: Optional[torch.Tensor],
    grad_optis: Optional[List[Optimizer]],
    grad_free_opt: Optional[Callable],
    grad_free_opt_args: Optional[Dict[str, Any]],
    steps: int,
    batch_size: int,
    checkpoint: Optional[Checkpoint],
    save_checkpoints: bool,
    db_id: Optional[int],
):
    """

    @param model: PyTorch model that will be trained
    @param shadow_model: second PyTorch model that also gets evaluated but not trained
    @param train_input:
    @param train_target:
    @param train_label:
    @param test_input:
    @param test_target:
    @param test_label:
    @param grad_optis: List of gradient-based optimizers, CANNOT be used together with a gradient-free optimizer
    @param grad_free_opt: Gradient-free optimizer, CANNOT be used together with gradient-based optimizers
    @param grad_free_opt_args:
    @param steps:
    @param batch_size:
    @param checkpoint:
    @param save_checkpoints:
    @param db_id: index of the database entry for the task, only needs to be specified if save_checkpoints is true
    """
    loss_func = torch.nn.MSELoss()
    train_dataset = TensorDataset(train_input, train_target)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data_available = test_input is not None and test_target is not None

    if test_data_available:
        test_dataset = TensorDataset(test_input, test_target)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    batch_cnt_overall = 0
    start_epoch = 0

    # load model and optimizer parameters / state from the checkpoint
    if checkpoint is not None:
        start_epoch = checkpoint.epoch + 1
        model.load_state_dict(torch.load(checkpoint.model_params_file))

        for optim, file in zip(grad_optis, checkpoint.optim_params_files):
            optim.load_state_dict(torch.load(file))

    for i in range(start_epoch, steps):
        # training on the training dataset
        error_sum = 0
        batch_cnt = 0

        for batch_input, batch_target in train_dataloader:
            # TODO: check if gradient-based or gradient-free optimizers are available

            for opt in grad_optis:
                opt.zero_grad()

            loss = loss_func(model(batch_input), batch_target)
            loss.backward()

            with torch.no_grad():
                # execute shadow model
                shadow_model_loss = loss_func(
                    shadow_model(batch_input), batch_target
                )  # TODO: return this value

            for opt in grad_optis:
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

                    shadow_model_loss = loss_func(
                        shadow_model(batch_input), batch_target
                    )  # TODO: do something with this value

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
                # TODO: save these embeddings and labels?
                embeddings = torch.cat(
                    (model.embed(train_input), model.embed(test_input)), dim=0
                )
                labels = torch.cat((train_label, test_label), dim=0)

        if save_checkpoints:
            with SpooledTemporaryFile(mode="wb") as output:
                model.save_model_parameters(output)
                STORE.persist_task_result(
                    db_id,
                    output,
                    "model_" + str(i) + ".pth",
                    "autoencoder-params",
                    "application/octet-stream",
                )

            for j, optim in enumerate(grad_optis):
                with SpooledTemporaryFile(mode="wb") as output:
                    save_optim_state(optim, output)
                    STORE.persist_task_result(
                        db_id,
                        output,
                        "optim" + str(j) + "_" + str(j) + ".pth",
                        "c-optim-params",
                        "application/octet-stream",
                    )
