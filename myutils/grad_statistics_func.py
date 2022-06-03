import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from fairseq import checkpoint_utils
from .layers import _apply_to_tensors_only, PreBackwardFunction


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
