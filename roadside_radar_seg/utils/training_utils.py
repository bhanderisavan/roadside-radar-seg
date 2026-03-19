import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from yacs.config import CfgNode
from typing import List, Dict, Any, Union
import numpy as np
from roadside_radar_seg.data import metadata
from tabulate import tabulate
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

_VALID_TYPES = {tuple, list, str, int, float, bool}


def convert_cfgnode_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary"""
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                "Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfgnode_to_dict(v, key_list + [k])
        return cfg_dict


def combine_input_embeddings_with_global_fvs(
    batch_input_embeddings_packed: PackedSequence,  # [batch, points, ...]
    global_fvs: Tensor,  # [batch, ...]
    padding_value: int,
) -> PackedSequence:

    input_embeddings_padded, lengths = pad_packed_sequence(
        batch_input_embeddings_packed,
        batch_first=True,
        padding_value=padding_value,  # 0.0,
        total_length=None,
    )  # ip_padded [4,N,64]
    features = torch.cat(
        (
            input_embeddings_padded,  # [4, N, 64]
            global_fvs.unsqueeze(1).repeat(
                1, input_embeddings_padded.shape[1], 1
            ),  # [4, N, 1024]
        ),
        dim=2,
    )  # [BATCH_SIZE, N, self.num_cat_features]
    # [4, N, 1088]

    # boolean mask of shape [batch_size, ip_padded.shape[1](1024)]
    # True for padded points false for valid points.
    mask = torch.arange(input_embeddings_padded.shape[1]).unsqueeze(
        0
    ) >= lengths.unsqueeze(1)

    # Apply the mask to zero out the padded portions
    features[mask] = padding_value  # 0.0

    packed_features = pack_padded_sequence(
        features,  # shape = [BATCH_SIZE, MAX_POINTS, NUM_FEATURES]
        lengths=lengths,
        batch_first=True,  # because the previous arg has `BATCH` as the first dimension.
        # enforce_sorted=False,  # If this is True, the batch must be sorted descending according to the `actual_lengths`.
    )

    return packed_features, features


def generate_cls_loss_targets_padded(
    input_point_indices: torch.Tensor,  # [Batch, N_max]
    batched_targets: List[dict],
    input_padding_value: int,
):
    """_summary_

    Args:
        input_point_indices (torch.Tensor): _description_
        input_padding_value (int): _description_

    gt_cls_ids: 0 means bg class, input_padding_value means padded point.
    """
    gt_cls_ids_padded = input_point_indices.clone()

    gt_cls_ids_padded[gt_cls_ids_padded != input_padding_value] = 0

    for gt_idx, gt_dict in enumerate(batched_targets):
        for gt_points, gt_cls_label in zip(gt_dict["points"], gt_dict["labels"]):
            # find out the index location of the current gt points in ip_indices_np and replcae it with gt_class_label.
            gt_indices = torch.isin(
                input_point_indices[gt_idx, :],
                torch.from_numpy(gt_points["index"].astype(np.float32)).to(
                    input_point_indices
                ),
            )
            gt_cls_ids_padded[gt_idx, gt_indices] = gt_cls_label

    return gt_cls_ids_padded  # [Batch, N_max]


def get_epoch_stats_dict(
    sim_losses_batchwise: List[float],
    cls_losses_batchwise: List[float],
    lerning_rate: float,
    epoch_idx: int,
    map_dict: Dict[str, torch.Tensor],
    cm: torch.Tensor,
) -> dict:
    """_summary_

    Args:
        sim_losses_batchwise (List[float]): similarity loss for each batch.
        cls_losses_batchwise (List[float]): cls loss for each batch.
        map_dict (Dict[str, torch.Tensor]): metric output dict. see evaluator.RadarMeanAveragePrecision.compute()
        lerning_rate (float): model learning rate
        epoch_idx (int): current epoch number
        cm  (torch.Tensor): confusion matrix

    Returns:
        dict: dictionary with current epoch stats.

    The return dict has the following keys:
    'epoch'             : current epoch number.
    'similarity_loss'   : similarity loss for the current epoch. (mean over batch losses)
    'cls_loss'          : cls loss for the current epoch. (mean over batch losses)
    'total_loss'        : total loss (similarity + cls) for the current epoch.
    'lr'                : learning rate for the current epoch.
    'mAP'               : mean average precision the current epoch.
    'mAP_50'            : mean average precision @ iou = 0.5 the current epoch.
    'mAP_75'            : mean average precision @ iou = 0.75 the current epoch.
    'mAP_classwise'     : mean average precision class wise the current epoch. Dict with class_name: mAP pair.
    'cm'                : confusion matrix

    """

    cls_wise_map = {
        metadata.CLASS_ID_TO_NAMES_WITHOUT_GROUP[str(cls_idx.item()).zfill(2)]: round(
            map_dict["map_per_class"][iter_idx].item(), 4
        )
        for iter_idx, cls_idx in enumerate(map_dict["classes"])
    }
    total_losses_batchwise = [
        s + c for s, c in zip(sim_losses_batchwise, cls_losses_batchwise)
    ]
    total_batches = len(sim_losses_batchwise)

    assert (
        len(total_losses_batchwise)
        == len(sim_losses_batchwise)
        == len(cls_losses_batchwise)
        == total_batches
    )

    train_map_50 = round(map_dict["map_50"].item(), 4)
    current_epoch_stats = dict(
        epoch=epoch_idx,
        similarity_loss=sum(sim_losses_batchwise) / total_batches,
        cls_loss=sum(cls_losses_batchwise) / total_batches,
        total_loss=sum(total_losses_batchwise) / total_batches,
        lr=lerning_rate,
        mAP_classwise=cls_wise_map,
        mAP_50=train_map_50,
        mAP=map_dict["map"].item(),
        mAP_75=map_dict["map_75"].item(),
        cm=cm.cpu().tolist(),
    )

    if map_dict.__contains__("map_30"):
        current_epoch_stats["mAP_30"] = round(map_dict["map_30"].item(), 4)

    return current_epoch_stats


def create_table(dictionary: Dict[str, float]):
    k, v = tuple(zip(*dictionary.items()))
    table = tabulate(
        [v],
        headers=k,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )

    return table


def get_map_tables(epoch_dict: Dict[str, Any], mode: str):
    eval_dict = {
        f"{mode}_mAP": epoch_dict["mAP"] * 100,
        f"{mode}_AP50": epoch_dict["mAP_50"] * 100,
        f"{mode}_AP75": epoch_dict["mAP_75"] * 100,
    }

    if epoch_dict.__contains__("mAP_30"):
        eval_dict[f"{mode}_AP30"] = (epoch_dict["mAP_30"] * 100,)

    eval_table = create_table(eval_dict)
    cls_wise_map = {k: v * 100 for k, v in epoch_dict["mAP_classwise"].items()}
    cls_wise_table = create_table(cls_wise_map)

    return eval_table, cls_wise_table


def print_epoch_results(
    train_epoch_dict: Dict[str, Any], val_epoch_dict: Dict[str, Any]
):

    assert train_epoch_dict["epoch"] == val_epoch_dict["epoch"]

    epoch_idx = train_epoch_dict["epoch"]
    print(
        f"Train Epoch:\t{epoch_idx}.\tcls loss: {train_epoch_dict['cls_loss']:.3f}.\tsim loss: {train_epoch_dict['similarity_loss']:.3f}.\ttotal loss: {train_epoch_dict['total_loss']:.3f}."
    )
    print(
        f"Val Epoch:\t{epoch_idx}.\tcls loss: {val_epoch_dict['cls_loss']:.3f}.\tsim loss: {val_epoch_dict['similarity_loss']:.3f}.\ttotal loss: {val_epoch_dict['total_loss']:.3f}."
    )
    train_eval_table, train_cls_wise_map_table = get_map_tables(
        train_epoch_dict, mode="train"
    )

    class_names = list(metadata.CLASS_NAMES_TO_ID_WITHOUT_GROUP.keys())

    if "background" not in class_names:
        class_names.insert(0, "backgournd")

    train_df = pd.DataFrame(
        train_epoch_dict["cm"], columns=class_names, index=class_names
    )
    train_cm_table = tabulate(
        train_df,
        tablefmt="fancy_grid",
        headers="keys",
        numalign="center",
        stralign="center",
    )

    val_df = pd.DataFrame(val_epoch_dict["cm"], columns=class_names, index=class_names)
    val_cm_table = tabulate(
        val_df,
        tablefmt="fancy_grid",
        headers="keys",
        numalign="center",
        stralign="center",
    )

    val_eval_table, val_cls_wise_map_table = get_map_tables(val_epoch_dict, mode="val")

    print(f"\n{'-'*25} EPOCH {epoch_idx} EVALUATION {'-'*25}\n")
    print("TRAINING\n")
    print(train_eval_table, "\n")
    print(train_cls_wise_map_table, "\n")
    print(f"Train Confusion Matrix [True Labels, Pred Labels]:\n")
    print(train_cm_table, "\n\n")
    print("VALIDATION\n")
    print(val_eval_table, "\n")
    print(val_cls_wise_map_table, "\n")
    print(f"Validation Confusion Matrix [True Labels, Pred Labels]:\n")
    print(val_cm_table, "\n")


def _log_dict_to_tensorboard(
    epoch_stats_dict: Dict[str, Any], mode: str, writer: SummaryWriter, epoch_idx: int
):
    for k, v in epoch_stats_dict.items():
        if isinstance(v, dict):
            _log_dict_to_tensorboard(v, mode=mode, writer=writer, epoch_idx=epoch_idx)
            continue
        writer.add_scalar(f"{mode}/{k}", v, epoch_idx)


def log_training_dict_to_tensorboard(
    training_epoch_dict: Dict[str, Any], writer: SummaryWriter
):

    training_dict = deepcopy(training_epoch_dict)

    if "cm" in training_dict.keys():
        del training_dict["cm"]

    _log_dict_to_tensorboard(
        training_dict,
        mode="train",
        writer=writer,
        epoch_idx=training_epoch_dict["epoch"],
    )


def log_validation_dict_to_tensorboard(
    validation_epoch_dict: Dict[str, Any], writer: SummaryWriter
):
    validation_dict = deepcopy(validation_epoch_dict)

    if "cm" in validation_dict.keys():
        del validation_dict["cm"]
    # do not log learning rate in validation logs.
    if "lr" in validation_dict.keys():
        del validation_dict["lr"]
    _log_dict_to_tensorboard(
        validation_dict,
        mode="val",
        writer=writer,
        epoch_idx=validation_dict["epoch"],
    )


def get_model_summary(model):
    table = PrettyTable(["Modules", "Trainable_Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            params = 0  # parameter.numel()
            table.add_row([name, params])
            # total_params += params
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    table.add_row(["total", total_params])
    return table


def index_recarray_by_column(
    cloud: np.recarray, column: str, indices: Union[int, list], invert: bool = False
) -> np.recarray:
    """
    index a cloud/numpy array based on values of a specific column.
    e.g instead of indexing based on element location [0 ... n]

    Args:
        cloud: a numpy array to index
        column: which column to use for indexing
        indices: list of values for indexing
        invert: if True, return cloud that does not contain indices
    returns:
        cloud_indexed: cloud indexed by indices in column.


    EXAMPLE
    -------

    >>> select_by_column(cloud, column = "index", indices=3, invert= True)

    This will index the cloud array with 'index' column.
    It will check where the value of clud["index"] =3 and return the cloud except that because invert is True.

    """
    assert column in cloud.dtype.names

    assert isinstance(
        indices, (int, list)
    ), f"Unsupported type of value for indices : {indices.__class__.__name__}"

    if isinstance(indices, int):
        indices = [indices]

    # query the indices based on column
    try:
        indices_based_on_column = [
            np.where(cloud[column] == int(i))[0].item() for i in indices
        ]
    except ValueError:
        indices_based_on_column = []

    if invert:
        return np.delete(cloud, indices_based_on_column)

    cloud_indexed = cloud[indices_based_on_column]

    return cloud_indexed


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )

    plt.show()
