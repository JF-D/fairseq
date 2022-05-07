#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer


def grad_statistics(cfg, trainer):
    nhooked_iters, niters = 0, 100
    hook_modules = {}
    def backward_hook(module, grad_in, grad_out):
        nonlocal nhooked_iters, niters
        assert len(grad_out) == 1
        module.hook_grads.append(grad_out[0].clone().detach().cpu().numpy())

        if module.hook_id == 0:
            nhooked_iters += 1

        if nhooked_iters >= niters:
            if torch.distributed.get_rank() == 0:
                print('Train finish, begin to plot...')
            for idx, hook_module in hook_modules.items():
                ep = cfg.checkpoint.restore_file.split('/')[-1][10:].split('.')[0]
                if torch.distributed.get_rank() == 0:
                    print(f'  plot {idx}...')
                assert len(hook_module.hook_grads) >= niters
                import numpy as np
                import matplotlib.pyplot as plt
                from matplotlib import cm

                if torch.distributed.get_rank() == 0:
                    # plot 3d value surf
                    for g in hook_module.hook_grads:
                        print(g.shape)
                    hook_grads = [g[:16, :16, :16].reshape(-1) for g in hook_module.hook_grads[1:]]

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    iterations = np.arange(len(hook_grads[1:]))
                    positions = np.arange(hook_grads[1:][0].size)
                    # iterations, positions = np.meshgrid(iterations, positions)
                    positions, iterations = np.meshgrid(positions, iterations)
                    grads = np.vstack(hook_grads[1:])

                    surf = ax.plot_surface(positions, iterations, grads, rstride=1, cstride=1, cmap=cm.viridis)
                    fig.colorbar(surf, shrink=1, aspect=30)
                    fig.savefig(f'log/3dfigs/grad_value_ep{ep}_l{idx}.pdf')

                # box plot of values and errors
                hook_module.hook_grads = [g.reshape(-1) for g in hook_module.hook_grads[1:]]
                if torch.distributed.get_rank() == 0:
                    fig, ax = plt.subplots()
                    grads = []
                    for grad in hook_module.hook_grads[1:]:
                        grads.append(grad[grad != 0])
                    ax.boxplot(grads, whis=1.5, showfliers=False, showmeans=True)
                    fig.savefig(f'log/figs/grad_value_ep{ep}_l{idx}.pdf')

                if torch.distributed.get_rank() == 0:
                    relative_errors = []
                    for i in range(len(hook_module.hook_grads) - 1):
                        delayed, current = hook_module.hook_grads[i], hook_module.hook_grads[i + 1]
                        delayed, current = delayed[current != 0], current[current != 0]
                        relative_errors.append(np.abs((current - delayed) / (current + 1e-9)))
                    fig, ax = plt.subplots()
                    ax.boxplot(relative_errors, showfliers=False, showmeans=True)
                    fig.savefig(f'log/figs/grad_error_ep{ep}_l{idx}.pdf')
            exit()
        # if torch.distributed.get_rank() == 0 and module.hook_id == 0:
        #     print(module.hook_id)
        #     # if grad_in is not None:
        #     #     print('  grad in:', len(grad_in), [g.size() for g in grad_in])
        #     if grad_out is not None:
        #         print('  grad out:', len(grad_out), [g.size() for g in grad_out])

    module_lists = list(trainer.model.module.module.encoder.layers.named_children())
    module_lists += list(trainer.model.module.module.decoder.layers.named_children())
    for idx, (name, module) in enumerate(module_lists):
        if idx in [0, 2, 5, 7, 9]:
            module.hook_id = idx
            module.hook_grads = []
            module.register_backward_hook(backward_hook)

            hook_modules[idx] = module

#apply torch.autograd.Function that calls a backward_function to tensors in output
def _apply_to_tensors_only(module, functional, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(module,
                                                    functional,
                                                    backward_function,
                                                    output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        return functional.apply(module, backward_function, outputs)
    else:
        return outputs

class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        if not hasattr(module, 'applied_pre_backward_ref_cnt'):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1
        # print(f"After Forward: {ctx.module.__class__.__name__}")
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # print(f"Before Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module, *args)
        return (None, None) + args


def replace_gradients_with_last_iter(trainer):

    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module, *args):
            # some models (e.g. Albert) may run multiple forwards on the same layer in a loop
            # before doing backwards, so each backward will need a pre-fetch - using reference
            # counting to support this scenario
            # print(f"COUNTER before: {sub_module.applied_pre_backward_ref_cnt}")
            if sub_module.applied_pre_backward_ref_cnt > 0:

                if not hasattr(sub_module, 'delayed_gradient'):
                    sub_module.delayed_gradient = args[0]
                try:
                    bs = args[0].size(1)
                    out_grad = sub_module.delayed_gradient[:, :bs] * (args[0] != 0)
                except:
                    out_grad = args[0]
                    print('[Normal] a normal iteration')

                sub_module.delayed_gradient = args[0]
                args = (out_grad, )

                sub_module.applied_pre_backward_ref_cnt -= 1
                return (None, None) + args
            #print(f"COUNTER after: {sub_module.applied_pre_backward_ref_cnt}")

        return _apply_to_tensors_only(module,
                                      PreBackwardFunction,
                                      _run_before_backward_function,
                                      output)

    module_lists = list(trainer.model.module.module.encoder.layers.named_children())
    module_lists += list(trainer.model.module.module.decoder.layers.named_children())
    for idx, (name, module) in enumerate(module_lists):
        if idx in [5]:
            module.hook_id = idx
            module.register_forward_hook(_pre_backward_module_hook)

def parameter_statistics(cfg, trainer, task):
    track_modules = {}
    parameters = {}
    module_lists = list(trainer.model.module.module.encoder.layers.named_children())
    module_lists += list(trainer.model.module.module.decoder.layers.named_children())
    for idx, (name, module) in enumerate(module_lists):
        if idx in [0, 2, 5, 7, 9]:
            track_modules[idx] = module
            parameters[idx-0.1] = []
            parameters[idx] = []

    for ep in range(0, 100):
        cfg.checkpoint['restore_file'] = f'checkpoints/en-de-base/checkpoint{ep}.pt'
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
            cfg.checkpoint,
            trainer,
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
        for idx, module in track_modules.items():
            w = list(module.parameters())[-2][:16].clone().detach().cpu().numpy().reshape(-1)
            parameters[idx-0.1].append(w)
            w = list(module.parameters())[-4][:4, :16].clone().detach().cpu().numpy().reshape(-1)
            parameters[idx].append(w)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import cm

    for idx, params in parameters.items():
        if torch.distributed.get_rank() == 0:
            # plot 3d value surf
            print(f'plot layer {idx}...')
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            iterations = np.arange(len(params))
            positions = np.arange(params[0].size)
            positions, iterations = np.meshgrid(positions, iterations)
            grads = np.vstack(params)

            surf = ax.plot_surface(positions, iterations, grads, rstride=1, cstride=1, cmap=cm.viridis)
            # ax.contourf(iterations, positions, grads, zdir='z', offset=-0.1)
            # ax.set_zlim3d(*lims[idx])
            cbar = fig.colorbar(surf, shrink=1, aspect=30)
            # cbar.mappable.set_clim(vmin=lims[idx][0], vmax=lims[idx][1])
            # fig.savefig(f'log/figs_param/param_value_l{idx}.pdf')
            def data_gen():
                for angle in range(0, 360, 10):
                    yield angle
            def run(angle):
                ax.view_init(30, angle)
            ani = animation.FuncAnimation(fig, run, data_gen, repeat=True)
            ani.save(f'log/figs_param_gif/param_value_l{idx}.gif', writer='pillow')
    exit()

disable_shuffle = False
current_sample, first_epoch, epoch_info = None, True, [0, 0, False]

def replace_gradients_with_last_sample(trainer):
    global current_sample, first_epoch, disable_shuffle
    disable_shuffle = True
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module, *args):
            if sub_module.applied_pre_backward_ref_cnt > 0:

                key = tuple(list(current_sample[0]['id'].cpu().numpy()))
                if not hasattr(sub_module, 'delayed_gradient'):
                    sub_module.delayed_gradient = {}
                    sub_module.delayed_stream = torch.cuda.Stream()
                try:
                    delayed = sub_module.delayed_gradient[key]
                    assert args[0].shape == delayed.shape
                    out_grad = delayed.cuda()
                except:
                    out_grad = args[0]
                    if torch.distributed.get_rank() == 0 and not first_epoch:
                        print('[Normal] a normal iteration')

                with torch.cuda.stream(sub_module.delayed_stream):
                    sub_module.delayed_gradient[key] = args[0].cpu()
                args = (out_grad, )

                sub_module.applied_pre_backward_ref_cnt -= 1
                return (None, None) + args

        return _apply_to_tensors_only(module,
                                      PreBackwardFunction,
                                      _run_before_backward_function,
                                      output)

    module_lists = list(trainer.model.module.module.encoder.layers.named_children())
    module_lists += list(trainer.model.module.module.decoder.layers.named_children())
    for idx, (name, module) in enumerate(module_lists):
        if idx in [5]:
            module.hook_id = idx
            module.register_forward_hook(_pre_backward_module_hook)

def myplot_animation(data_list, name, ylabel='Iteration'):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import cm

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    datas = [act.reshape(-1) for act in data_list]
    iterations = np.arange(len(data_list))
    positions = np.arange(datas[0].size)
    positions, iterations = np.meshgrid(positions, iterations)
    acts = np.vstack(datas)
    surf = ax.plot_surface(positions, iterations, acts, rstride=1, cstride=1, cmap=cm.viridis)
    ax.set_xlabel('Tensor')
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(surf, shrink=1, aspect=30)
    def data_gen():
        for angle in range(0, 360, 10):
            yield angle
    def run(angle):
        ax.view_init(30, angle)
    ani = animation.FuncAnimation(fig, run, data_gen, interval=350, repeat=True)
    ani.save(name, writer='pillow')

class RecordFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        if not hasattr(module, 'applied_pre_backward_ref_cnt'):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1
        # print(f"After Forward: {ctx.module.__class__.__name__}")
        global epoch_info
        if epoch_info[1] == 1:
            module.fw_stats[epoch_info[0]] = []
        module.fw_stats[epoch_info[0]].append(outputs[:4, :4, :4].clone().detach().cpu().numpy())
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # print(f"Before Backward: {ctx.module.__class__.__name__}")
        global epoch_info
        if epoch_info[1] == 1:
            ctx.module.bw_stats[epoch_info[0]] = []
        ctx.module.bw_stats[epoch_info[0]].append(args[0][:4, :4, :4].clone().detach().cpu().numpy())
        ctx.pre_backward_function(ctx.module, *args)
        return (None, None) + args

RECORD_LAST_SAMPLE = False
def record_last_sample_statistics(trainer):
    global disable_shuffle, epoch_info, RECORD_LAST_SAMPLE
    RECORD_LAST_SAMPLE = True
    disable_shuffle = True
    track_modules = {}
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module, *args):
            if sub_module.applied_pre_backward_ref_cnt > 0:
                sub_module.applied_pre_backward_ref_cnt -= 1
            if epoch_info[1] >= 100 and module.hook_id == 0:
                print('finish hook!!!')
                for idx, hook_module in track_modules.items():
                    if torch.distributed.get_rank() == 0:
                        myplot_animation(hook_module.fw_stats[0], f'log/figs_act/act_value_l{idx}.gif', ylabel='Epoch')
                        myplot_animation(hook_module.bw_stats[0], f'log/figs_act/grad_value_l{idx}.gif', ylabel='Epoch')
                exit()

        return _apply_to_tensors_only(module,
                                      RecordFunction,
                                      _run_before_backward_function,
                                      output)

    module_lists = list(trainer.model.module.module.encoder.layers.named_children())
    module_lists += list(trainer.model.module.module.decoder.layers.named_children())
    for idx, (name, module) in enumerate(module_lists):
        if idx in [0, 2, 5, 7, 9]:
            module.hook_id = idx
            module.fw_stats = {}
            module.bw_stats = {}
            module.register_forward_hook(_pre_backward_module_hook)

            track_modules[idx] = module


class SyntheticFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        if not hasattr(module, 'applied_pre_backward_ref_cnt'):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1

        if not hasattr(module, 'last_output'):
            module.last_output = outputs.clone().detach()

        if module.vary_input:
            outputs = outputs.detach() * 1.001
        else:
            outputs = module.last_output.clone().detach()
        if torch.distributed.get_rank() == 0:
            print(outputs[:2, :2, :2])
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # if not hasattr(ctx.module, 'last_gradient'):
        #     ctx.module.last_gradient = args[0].clone().detach()
        ctx.pre_backward_function(ctx.module, *args)
        return (None, None) + args

KEEP_FIRST_SAMPLE = False
def synthetic_input_module(cfg, trainer, module_id, vary_input=False):
    global KEEP_FIRST_SAMPLE
    KEEP_FIRST_SAMPLE = True
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module, *args):
            if sub_module.applied_pre_backward_ref_cnt > 0:
                module.grad_stats.append(args[0][:4, :4, :4].clone().detach().cpu().numpy())
                sub_module.applied_pre_backward_ref_cnt -= 1

            if len(module.grad_stats) >= 100:
                ep = cfg.checkpoint.restore_file.split('/')[-1][10:].split('.')[0]
                if torch.distributed.get_rank() == 0:
                    if vary_input:
                        fig_name = f'log/figs_sim/sim_l{module_id}_ep{ep}.gif'
                    else:
                        fig_name = f'log/figs_sim/eq_l{module_id}_ep{ep}.gif'
                    myplot_animation(module.grad_stats, fig_name)
                exit()

        return _apply_to_tensors_only(module,
                                      SyntheticFunction,
                                      _run_before_backward_function,
                                      output)

    module_lists = list(trainer.model.module.module.encoder.layers.named_children())
    module_lists += list(trainer.model.module.module.decoder.layers.named_children())
    for idx, (name, module) in enumerate(module_lists):
        if idx == module_id:
            module.hook_id = idx
            module.vary_input = vary_input
            module.grad_stats = []
            module.register_forward_hook(_pre_backward_module_hook)


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    # >>> hook to get gradients statistics
    # grad_statistics(cfg, trainer)
    # >>> hook to replace output gradients with delayed gradients
    # replace_gradients_with_last_iter(trainer)
    # >>> plot parameter statistics
    # parameter_statistics(cfg, trainer, task)
    # >>> replace gradients with last epoch sample
    # if torch.distributed.get_rank() == 0:
    #     replace_gradients_with_last_sample(trainer)
    # >>> record last sample statistics
    # record_last_sample_statistics(trainer)
    # >>> synchetic gradient module
    # synthetic_input_module(cfg, trainer, 9, vary_input=False)

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break
        global epoch_info, RECORD_LAST_SAMPLE
        if RECORD_LAST_SAMPLE:
            epoch_info[1] += 1
            cfg.checkpoint['restore_file'] = f'checkpoints/en-de-base/checkpoint{epoch_info[1]}.pt'
            cfg.reset_dataloader = True
            checkpoint_utils.load_checkpoint(
                cfg.checkpoint,
                trainer,
                # don't cache epoch iterators for sharded datasets
                disable_iterator_cache=task.has_sharded_data("train"),
            )

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        global first_epoch
        first_epoch = False
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    global disable_shuffle
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(not disable_shuffle) and (epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        aim_repo=(
            cfg.common.aim_repo
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_run_hash=(
            cfg.common.aim_run_hash
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    global current_sample, epoch_info, RECORD_LAST_SAMPLE, KEEP_FIRST_SAMPLE
    epoch_info[2] = False
    for i, samples in enumerate(progress):
        if KEEP_FIRST_SAMPLE:
            if current_sample is None:
                current_sample = samples
            samples = current_sample
        else:
            current_sample = samples
        if torch.distributed.get_rank() == 0:
            print(samples[0]['id'][:10])
        if RECORD_LAST_SAMPLE:
            epoch_info = [i, epoch_info[1], epoch_info[2]]
            epoch_info[2] = (epoch_info[0] >= 1)
            if epoch_info[2]:
                valid_losses, should_stop = [0], False
                break
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
            or should_stop
            or (
                cfg.dataset.validate_interval_updates > 0
                and num_updates > 0
                and num_updates % cfg.dataset.validate_interval_updates == 0
            )
        )
        and not cfg.dataset.disable_validation
        and num_updates >= cfg.dataset.validate_after_updates
    )

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset_idx, subset in enumerate(subsets):
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            aim_repo=(
                cfg.common.aim_repo
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_run_hash=(
                cfg.common.aim_run_hash
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if (
                    cfg.dataset.max_valid_steps is not None
                    and i > cfg.dataset.max_valid_steps
                ):
                    break
                trainer.valid_step(sample)

        # log validation stats
        # only tracking the best metric on the 1st validation subset
        tracking_best = subset_idx == 0
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values(), tracking_best)

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig,
    trainer: Trainer,
    stats: Dict[str, Any],
    tracking_best: bool,
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if tracking_best and hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
