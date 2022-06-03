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
import random
import pickle
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

from myutils import _apply_to_tensors_only, PreBackwardFunction


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
    track_modules, track_params = {}, {}
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module, *args):
            if sub_module.applied_pre_backward_ref_cnt > 0:
                sub_module.applied_pre_backward_ref_cnt -= 1

        return _apply_to_tensors_only(module,
                                      RecordFunction,
                                      _run_before_backward_function,
                                      output)

    stats = {}
    def track_param_gradient_direction():
        if epoch_info[0] == 0 and torch.distributed.get_rank() == 0:
            for idx, param_list in track_params.items():
                for i, param in enumerate(param_list):
                    stats[idx][i].append(param.grad.clone().detach().cpu().numpy())
        if epoch_info[1] >= 100:
            if torch.distributed.get_rank() != 0:
                exit()
            print('finish hook!!!')
            import matplotlib.pyplot as plt
            from myutils import myplot_animation
            for idx, hook_module in track_modules.items():
                if torch.distributed.get_rank() == 0:
                    myplot_animation(hook_module.fw_stats[0], f'log/figs_act/act_value_l{idx}.gif', ylabel='Epoch')
                    myplot_animation(hook_module.bw_stats[0], f'log/figs_act/grad_value_l{idx}.gif', ylabel='Epoch')
            for idx, grads_list in stats.items():
                directions, norms = {}, {}
                for i, grads in enumerate(grads_list):
                    directions[i], norms[i] = [], []
                    for j in range(len(grads)):
                        norms[i].append(np.linalg.norm(grads[j]))
                        if j == 0:
                            directions[i].append(0)
                        else:
                            sim = np.dot(grads[j].reshape(-1), grads[j - 1].reshape(-1)) / (norms[i][j] * norms[i][j - 1])
                            directions[i].append(sim)
                x = np.arange(len(norms[0]))
                fig, ax = plt.subplots(dpi=300)
                ax.bar(x - 0.2, norms[0], width=0.4, color='#67AB9F', label='Linear Grad Norm')
                ax.bar(x + 0.2, norms[1], width=0.4, color='#A680B8', label='LayerNorm grad Norm')
                rax = ax.twinx()
                rax.plot(x, directions[0], label='Linear Grad Sim', marker='o')
                rax.plot(x, directions[1], label='LayerNorm Grad Sim', marker='x')

                ax.set_ylabel('Norms')
                rax.set_ylabel('Similarity (cos value)')
                ax.set_xlabel('Epoch')
                ax.legend()
                rax.legend()
                ax.set_title(f'param_direction_module_{idx}')
                fig.savefig(f'log/figs_act/param_direction_l{idx}.pdf')
            exit()

    trainer.hook_before_grad_reduce = track_param_gradient_direction
    module_lists = list(trainer.model.module.module.encoder.layers.named_children())
    module_lists += list(trainer.model.module.module.decoder.layers.named_children())
    for idx, (name, module) in enumerate(module_lists):
        if idx in [0, 2, 5, 7, 9]:
            module.hook_id = idx
            module.fw_stats = {}
            module.bw_stats = {}
            module.register_forward_hook(_pre_backward_module_hook)

            track_modules[idx] = module
            track_params[idx] = [list(module.parameters())[-4], list(module.parameters())[-2]]
            stats[idx] = [[], []]


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
                if torch.distributed.get_rank() == 0:
                    module.grad_stats.append(args[0].clone().detach().cpu().numpy())
                else:
                    module.grad_stats.append(args[0][:1,:1,:1].clone().detach().cpu().numpy())
                sub_module.applied_pre_backward_ref_cnt -= 1

        return _apply_to_tensors_only(module,
                                      SyntheticFunction,
                                      _run_before_backward_function,
                                      output)

    stats, track_module, track_params = {}, None, {}
    def track_param_gradient_direction():
        if torch.distributed.get_rank() == 0:
            for idx, param_list in track_params.items():
                for i, param in enumerate(param_list):
                    stats[idx][i].append(param.grad.clone().detach().cpu().numpy())

        if len(track_module.grad_stats) >= 60:
            if torch.distributed.get_rank() != 0:
                exit()
            import matplotlib.pyplot as plt
            from myutils import myplot_animation
            ep = cfg.checkpoint.restore_file.split('/')[-1][10:].split('.')[0]
            if vary_input:
                fig_name = f'log/figs_sim/sim_l{module_id}_ep{ep}.gif'
            else:
                fig_name = f'log/figs_sim/eq_l{module_id}_ep{ep}.gif'
            grad_stats = [g[:4, :4, :4] for g in track_module.grad_stats]
            myplot_animation(grad_stats, fig_name)

            stats[module_id + 1].append(track_module.grad_stats)
            for idx, grads_list in stats.items():
                directions, norms = {}, {}
                for i, grads in enumerate(grads_list):
                    directions[i], norms[i] = [], []
                    for j in range(len(grads)):
                        norms[i].append(np.linalg.norm(grads[j]))
                        if j == 0:
                            directions[i].append(0)
                        else:
                            sim = np.dot(grads[j].reshape(-1), grads[j - 1].reshape(-1)) / (norms[i][j] * norms[i][j - 1])
                            directions[i].append(sim)
                x = np.arange(len(norms[0]))
                fig, ax = plt.subplots(dpi=300)
                ax.bar(x - 0.3, norms[0], width=0.3, color='#67AB9F', label='Linear Grad Norm')
                ax.bar(x, norms[1], width=0.3, color='#A680B8', label='LayerNorm grad Norm')
                ax.bar(x + 0.3, norms[2], width=0.3, color='#7EA6E0', label='Activation grad Norm')
                rax = ax.twinx()
                rax.plot(x, directions[0], label='Linear Grad Sim', marker='o')
                rax.plot(x, directions[1], label='LayerNorm Grad Sim', marker='x')
                rax.plot(x, directions[2], label='Activation Grad Sim', marker='+')

                ax.set_ylabel('Norms')
                rax.set_ylabel('Similarity (cos value)')
                ax.set_xlabel('Iteration')
                ax.legend()
                rax.legend()
                ax.set_title(f'param_direction_module_{idx}_ep{ep}')
                fig.savefig(f'log/figs_sim/param_direction_l{idx}_ep{ep}.pdf')
            exit()

    trainer.hook_before_grad_reduce = track_param_gradient_direction
    module_lists = list(trainer.model.module.module.encoder.layers.named_children())
    module_lists += list(trainer.model.module.module.decoder.layers.named_children())
    for idx, (name, module) in enumerate(module_lists):
        if idx == module_id:
            module.hook_id = idx
            module.vary_input = vary_input
            module.grad_stats = []
            module.register_forward_hook(_pre_backward_module_hook)
            track_module = module
        elif idx == module_id + 1:
            track_params[idx] = [list(module.parameters())[-4], list(module.parameters())[-2]]
            stats[idx] = [[], []]


class MimicDelayedGrad:
    """ Mimic preemption with delayed gradients (last iteration parameter gradient).

    default case: Mimic preemption with delayed gradients.
    drop_grad=True: drop the gradient before preemption node.
        drop_dp=True: use with drop_grad, drop a data parallel process with preempted nodes.
        drop_preemption_only=True: use with drop_grad, drop the gradient of preemption node only.
    """
    def __init__(self, trainer, p=0.05, max_preempt_iters=20, drop_grad=False, drop_dp=False, drop_preemption_only=False):
        self.max_preempt_iters = max_preempt_iters
        self.drop_grad = drop_grad
        self.drop_dp = drop_dp
        self.drop_preemption_only = drop_preemption_only

        if self.drop_preemption_only:
            assert not self.drop_dp and self.drop_grad
        elif self.drop_dp:
            assert self.drop_grad

        self.parameter_map = {}
        self.gradient_hist = {}
        self.partition = {}

        last_param_name = ['encoder.layers.3', 'decoder.layers.0', 'decoder.layers.3']

        cur_part = 0
        self.partition[cur_part] = []
        for name, param in trainer.model.module.named_parameters():
            if len(last_param_name) > cur_part and last_param_name[cur_part] in name:
                cur_part += 1
                self.partition[cur_part] = []
            self.partition[cur_part].append(name)
            self.parameter_map[name] = param
            if not self.drop_grad:
                self.gradient_hist[name] = torch.zeros_like(param)

        self.update = 0
        self.preemption_iters = [0, 0, 0, 0]

        # random.seed(torch.distributed.get_rank())
        self.prob = p
        print(f'[Mimic Preemption] prob: {self.prob}')

    def run(self):
        self.update += 1

        preemptions = [0] * len(self.preemption_iters)
        preemption_depth = 0
        for i in range(len(self.preemption_iters)):
            if self.preemption_iters[i] > 0:
                preemptions[i] = 1
                preemption_depth = max(preemption_depth, i + 1)
                self.preemption_iters[i] -= 1
            if self.preemption_iters[i] <= 0 and self.update >= 5 and random.random() < self.prob:
                self.preemption_iters[i] = random.randint(0, self.max_preempt_iters)

        if self.drop_dp and preemption_depth > 0:
            preemption_depth = len(self.preemption_iters)

        if not self.drop_preemption_only:
            preemptions = [1 if i < preemption_depth else 0 for i in range(len(self.preemption_iters))]

        last_preemptions = torch.tensor(preemptions).cuda()
        distributed_utils.all_reduce(last_preemptions, distributed_utils.get_data_parallel_group())
        self.last_preemptions = last_preemptions

        for i in range(len(self.preemption_iters)):
            if preemptions[i] > 0:
                for name in self.partition[i]:
                    if self.drop_grad:
                        self.parameter_map[name].grad = torch.zeros_like(self.parameter_map[name].grad)
                    else:
                        self.parameter_map[name].grad = self.gradient_hist[name].clone().detach()

    def post_scale(self):
        if self.drop_grad:
            for i in range(len(self.preemption_iters)):
                if self.last_preemptions[i] == 8:
                    for name in self.partition[i]:
                        self.parameter_map[name].grad = None
                elif self.last_preemptions[i] > 0:
                    for name in self.partition[i]:
                        self.parameter_map[name].grad.data.mul_(self.last_preemptions[i] / 8)

def mimic_delay_gradient(trainer, p=0.05, max_preempt_iters=20, drop_grad=False, drop_dp=True, drop_preemption_only=False):
    hook = MimicDelayedGrad(trainer, p=p, max_preempt_iters=max_preempt_iters, drop_grad=drop_grad, drop_dp=drop_dp, drop_preemption_only=drop_preemption_only)
    trainer.hook_before_grad_reduce = hook.run
    trainer.hook_after_grad_reduce = hook.post_scale


class MimicPreemption:
    def __init__(self, trainer, prob=0.1, iters_per_preemption=10, sec_per_iter=0.2, delayed_grad=False, recovery_grad=False, seed=12345):
        self.delayed_grad = delayed_grad
        self.recovery_grad = recovery_grad

        self.parameter_map = {}
        self.gradient_hist = {}
        self.partition = {}

        last_param_name = ['encoder.layers.3', 'decoder.layers.0', 'decoder.layers.3']

        cur_part = 0
        self.partition[cur_part] = []
        for name, param in trainer.model.module.named_parameters():
            if len(last_param_name) > cur_part and last_param_name[cur_part] in name:
                cur_part += 1
                self.partition[cur_part] = []
            self.partition[cur_part].append(name)
            self.parameter_map[name] = param
            if self.delayed_grad:
                self.gradient_hist[name] = torch.zeros_like(param)

        from myutils import GenPreemptionTrace
        self.trace_generators = {}
        world_size = distributed_utils.get_world_size(distributed_utils.get_data_parallel_group())
        for i in range(4):
            rank = world_size * i + distributed_utils.get_rank(distributed_utils.get_data_parallel_group())
            self.trace_generators[i] = GenPreemptionTrace(world_size * 4, rank, prob=prob,
                                            iters_per_preemption=iters_per_preemption,
                                            sec_per_iter=sec_per_iter, seed=seed)

        self.update = 0
        print(f'[Mimic Preemption with Activation Gradient] prob: {prob}, iters_per_preemption: {iters_per_preemption}, '
              f'delayed_grad: {delayed_grad}, recovery_grad: {recovery_grad}, seed: {seed}, sec_per_iter: {sec_per_iter}')

    def run(self):
        if self.recovery_grad:
            self.update += 1
            return

        for i in range(len(self.trace_generators)):
            if self.trace_generators[i].check_preemption(self.update):
                for name in self.partition[i]:
                    if self.delayed_grad:
                        self.parameter_map[name].grad = self.gradient_hist[name].clone().detach()
                    else:
                        self.parameter_map[name].grad = torch.zeros_like(self.parameter_map[name].grad)
            else:
                if self.delayed_grad:
                    for name in self.partition[i]:
                        self.gradient_hist[name] = self.parameter_map[name].grad.clone().detach()

        self.update += 1

class SimiFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        if not hasattr(module, 'applied_pre_backward_ref_cnt'):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1

        module.cur_act = outputs.clone().detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        ctx.pre_backward_function(ctx.module, *args)
        module = ctx.module
        outputs = ctx.module.cur_act
        preemption = ctx.module.preemption

        with torch.no_grad():
            if preemption:
                synthetic_out = torch.zeros_like(args[0])

            act = outputs.clone().detach()
            l = min(act.size(0), module.sim_sample.size(0))
            if module.sim_func == 'cosine':
                act_norm = torch.sum(act * act, dim=(0, 2), keepdim=True)
                act = act / (act_norm + 1e-7)
                sim = torch.einsum('ijk,imk->jm', [module.sim_sample[:l], act[:l]])
                sim_idx = torch.argmax(sim, dim=0).cpu().numpy()
            elif module.sim_func == 'l2norm':
                act_ = act[:l].unsqueeze(1)
                sample_ = module.sim_sample[:l].unsqueeze(2)
                extra_act_ = torch.norm(act[l:].squeeze(1), dim=[0, -1]).unsqueeze(0)
                extra_sample_ = torch.norm(module.sim_sample[l:].squeeze(2), dim=[0, -1]).unsqueeze(1)

                sim = torch.zeros(module.nsamples, act.size(1)).cuda()
                nparts = (act.size(1) + 31) // 32
                extent = module.nsamples // nparts
                for i in range(nparts):
                    st = extent * i
                    if module.nsamples % nparts > i:
                        st += i
                        ed = st + extent + 1
                    else:
                        st += module.nsamples % nparts
                        ed = st + extent
                    sim[st:ed] = torch.norm(torch.abs(act_ - sample_[:, st:ed]), dim=[0, -1])
                sim = sim + extra_act_ + extra_sample_
                sim_idx = torch.argmin(sim, dim=0).cpu().numpy()
                del act_, sample_, extra_act_, extra_sample_

            for k in range(outputs.size(1)):
                most_sim = sim[sim_idx[k], k]
                if module.sim_func == 'cosine':
                    sim_threshold = 0.8
                elif module.sim_func == 'l2norm':
                    sim_threshold = 10

                if not preemption:
                    if not module.update_buffer:
                        break
                    if module.init + 1 <= module.nsamples and most_sim > sim_threshold:
                        module.sim_gradient[module.init] = args[0][:, k].clone().detach()
                        module.sim_sample[:l, module.init] = act[:l, k]
                        if module.sim_func == 'cosine':
                            module.sim_norm[module.init] = act_norm[0, k, 0]
                        module.init += 1
                    else:
                        module.sim_sample[:l, sim_idx[k]] = act[:l, k]
                        module.sim_gradient[sim_idx[k]] = args[0][:, k].clone().detach()
                        if module.sim_func == 'cosine':
                            module.sim_norm[sim_idx[k]] = act_norm[0, k, 0]
                else:
                    grad = module.sim_gradient[sim_idx[k]]
                    l = min(args[0].size(0), grad.size(0))
                    if module.sim_func == 'cosine':
                        synthetic_out[:l, k] = grad[:l] * act_norm[0, k, 0] / (module.sim_norm[sim_idx[k]] + 1e-7)
                    else:
                        synthetic_out[:l, k] = grad[:l] + 1e-7
            del sim, act

        if preemption:
            if hasattr(module, 'my_grad_sim') and module.init >= module.nsamples:
                first_preemption_in_cur_iter = True
                for i in range(module.hook_id + 1, 4):
                    if module.my_hook.trace_generators[i].check_preemption(module.my_hook.update):
                        first_preemption_in_cur_iter = False
                if first_preemption_in_cur_iter:
                    if len(module.my_grad_sim) == 0:
                        module.my_grad_sim = [[] for _ in range(9)]
                    real_norm = torch.norm(args[0], dim=[0, -1])
                    synthetic_norm = torch.norm(synthetic_out, dim=[0, -1])
                    sim_norm = torch.norm(synthetic_out - args[0], dim=[0, -1])
                    for i, norm in enumerate([real_norm, synthetic_norm, sim_norm]):
                        module.my_grad_sim[3 * i].append(torch.max(norm).item())
                        module.my_grad_sim[3 * i + 1].append(torch.mean(norm).item())
                        module.my_grad_sim[3 * i + 2].append(torch.min(norm).item())
                    # module.my_grad_sim.append(sim_norm.cpu().numpy())
                    module.my_grad_sim_update.append(module.my_hook.update)
                    if torch.distributed.get_rank() == 0 and len(module.my_grad_sim_update) > 200:
                        labels = []
                        for name1 in ['real', 'synthetic', 'simi']:
                            for name2 in ['max', 'mean', 'min']:
                                labels.append(name1 + '_' + name2)
                        from myutils import curve_line
                        ep = 20
                        curve_line(list(range(len(module.my_grad_sim_update))), module.my_grad_sim[:6], labels[:6],
                                   xlabel='Step', ylabel='Norm', title=f'module{module.hook_id - 1}_ep{ep}',
                                   figname=f'log/trace/stats_ep{ep}.pdf')
                        curve_line(list(range(len(module.my_grad_sim_update))), module.my_grad_sim[6:], labels[6:],
                                   xlabel='Step', ylabel='Norm', title=f'module{module.hook_id - 1}_ep{ep}',
                                   figname=f'log/trace/simi_ep{ep}.pdf')
                        print('Finish...')
                        exit()
            args = (synthetic_out.detach(), )
        return (None, None) + args

def mimic_sample_similarity(trainer, p=0.05, iters_per_preemption=10, sec_per_iter=0.2,
                            delayed_grad=False, recovery_grad=False, nsamples=1024,
                            sim_func='l2norm', record_grad_sim_stats=False):
    # generate seed
    seed = torch.tensor(random.randint(0, 123)).cuda()
    distributed_utils.all_reduce(seed, distributed_utils.get_data_parallel_group())
    seed = seed.item()

    hook = MimicPreemption(trainer, prob=p, iters_per_preemption=iters_per_preemption,
                           sec_per_iter=sec_per_iter, delayed_grad=delayed_grad, recovery_grad=recovery_grad, seed=seed)
    trainer.hook_before_grad_reduce = hook.run

    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module, *args):
            if sub_module.applied_pre_backward_ref_cnt > 0:
                sub_module.preemption = (hook.trace_generators[sub_module.hook_id].check_preemption(hook.update))
                if sub_module.hook_id + 1 < len(hook.trace_generators):
                    # TBD: stop update buffer for all preemption
                    sub_module.update_buffer = True
                else:
                    sub_module.update_buffer = True
                sub_module.applied_pre_backward_ref_cnt -= 1

        return _apply_to_tensors_only(module,
                                      SimiFunction,
                                      _run_before_backward_function,
                                      output)

    module_lists = list(trainer.model.module.module.encoder.layers.named_children())
    module_lists += list(trainer.model.module.module.decoder.layers.named_children())
    for idx, (name, module) in enumerate(module_lists):
        if idx in [2, 5, 8]:
            module.hook_id = [2, 5, 8].index(idx) + 1
            module.init = 0
            module.sim_norm = {}
            module.sim_sample = torch.zeros(128, nsamples, 512).cuda() + 1e-6
            module.sim_gradient = {k: torch.zeros(128, 512).cuda() for k in range(nsamples)}
            module.nsamples = nsamples
            module.sim_func = sim_func
            module.register_forward_hook(_pre_backward_module_hook)

            if record_grad_sim_stats:
                module.my_grad_sim = []
                module.my_grad_sim_update = []
                module.my_hook = hook


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
    # from myutils import grad_statistics
    # grad_statistics(cfg, trainer)
    # >>> hook to replace output gradients with delayed gradients
    # from myutils import replace_gradients_with_last_iter
    # replace_gradients_with_last_iter(trainer)
    # >>> plot parameter statistics
    # from myutils import parameter_statistics
    # parameter_statistics(cfg, trainer, task)
    # >>> A1: replace gradients with last epoch sample
    # if torch.distributed.get_rank() == 0:
    #     replace_gradients_with_last_sample(trainer)
    # >>> record last sample statistics
    # record_last_sample_statistics(trainer)
    # >>> synchetic gradient module
    # synthetic_input_module(cfg, trainer, 0, vary_input=False)
    # >>> A3: mimic delay gradient, drop_grad=True can be used as a baseline of drop preemptions
    # mimic_delay_gradient(trainer, p=0.3, max_preempt_iters=10, drop_grad=True, drop_dp=False, drop_preemption_only=True)
    # >>> combine A2 and A1: mimic last sample preemption
    # on v100, sec_per_iter=0.2, on A100, sec_per_iter=0.1 (just an estimation!!!)
    # save_dir form: 'en-de-mimic-hybrid-ep10-p30-i10-*'
    # save_dir = cfg.checkpoint['save_dir'].split('/')[-1]
    # p = int(save_dir.split('-')[5][1:]) / 100
    # iters_per_preemption = int(save_dir.split('-')[6][1:])
    p = 0.4
    iters_per_preemption = 10
    mimic_sample_similarity(trainer, p=p, iters_per_preemption=iters_per_preemption, sec_per_iter=0.2,
                            delayed_grad=False, recovery_grad=False, nsamples=1024, sim_func='l2norm',
                            record_grad_sim_stats=False)

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
