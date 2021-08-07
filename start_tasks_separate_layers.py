import itertools

from clearml import Task

PROJECT_NAME = 'Algonauts separate layers'
BASE_TASK = 'task template'

task = Task.init(project_name=PROJECT_NAME,
                 task_name='Task Manager',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

template_task = Task.get_task(project_name=PROJECT_NAME,
                              task_name=BASE_TASK)

available_devices = {
    '16': [0, 1],
}

queue_names = []
for k, vs in available_devices.items():
    for v in vs:
        queue_names.append(f'{k}-{v}')
queues_buffer = itertools.cycle(queue_names)

task_ids = []


def start_tasks_spp(rois, layers, ps, freeze_bns, pooling_modes, batch_size=32):
    for roi in rois:
        for layer in layers:
            for p in ps:
                for freeze_bn in freeze_bns:
                    for pooling_mode in pooling_modes:
                        assert pooling_mode in ['max', 'avg']
                        queue = next(queues_buffer)

                        p_text = '-'.join([str(i) for i in p])
                        freeze_text = 'f_bn' if freeze_bn else 'nof_bn'
                        pooling_text = f'spp_{p_text}_{pooling_mode}'

                        tags = [roi, layer, pooling_text, freeze_text]
                        cloned_task = Task.clone(source_task=template_task,
                                                 name=','.join(tags),
                                                 parent=template_task.id)

                        cloned_task.add_tags(tags)

                        cloned_task_parameters = cloned_task.get_parameters()
                        # cloned_task_parameters['rois'] = [roi]
                        cloned_task_parameters['Args/rois'] = roi
                        cloned_task_parameters['Args/track'] = 'mini_track'
                        # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
                        cloned_task_parameters['Args/learning_rate'] = 1e-4
                        cloned_task_parameters['Args/step_lr_epochs'] = [4]
                        cloned_task_parameters['Args/step_lr_ratio'] = 0.5
                        cloned_task_parameters['Args/batch_size'] = batch_size if not freeze_bn else 8
                        cloned_task_parameters['Args/accumulate_grad_batches'] = 1 if not freeze_bn else int(
                            batch_size / 8)
                        cloned_task_parameters['Args/num_layers'] = 1
                        cloned_task_parameters['Args/conv_size'] = 256
                        cloned_task_parameters['Args/first_layer_hidden'] = 2048
                        cloned_task_parameters['Args/layer_hidden'] = 2048
                        cloned_task_parameters['Args/debug'] = False
                        cloned_task_parameters['Args/freeze_bn'] = freeze_bn
                        cloned_task_parameters['Args/old_mix'] = True
                        cloned_task_parameters['Args/early_stop_epochs'] = 5
                        cloned_task_parameters['Args/max_epochs'] = 100
                        cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
                        cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                        cloned_task_parameters['Args/pooling_mode'] = pooling_mode
                        for l in layer.split(','):
                            cloned_task_parameters[f'Args/{l}_pooling_mode'] = 'spp'
                            cloned_task_parameters[f'Args/spp_size_{l}'] = p
                            cloned_task_parameters[f'Args/spp_size_t_{l}'] = [1 for _ in p]
                        cloned_task_parameters['Args/backbone_type'] = 'i3d_rgb'
                        cloned_task_parameters['Args/final_fusion'] = 'concat'
                        cloned_task_parameters['Args/pyramid_layers'] = layer
                        cloned_task_parameters['Args/pathways'] = 'none'
                        cloned_task_parameters['Args/val_check_interval'] = 1.0
                        cloned_task_parameters['Args/val_ratio'] = 0.1
                        cloned_task_parameters['Args/save_checkpoints'] = True
                        cloned_task_parameters[
                            'Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'

                        cloned_task.set_parameters(cloned_task_parameters)
                        print('Experiment set with parameters {}'.format(cloned_task_parameters))

                        # enqueue the task for execution
                        Task.enqueue(cloned_task.id, queue_name=queue)
                        print('Experiment id={} enqueue for execution'.format(cloned_task.id))

                        task_ids.append(cloned_task.id)


def start_tasks_adaptive_pooling(rois, layers, rfs, rf_ts, freeze_bns, pooling_modes, batch_size=32):
    for roi in rois:
        for layer in layers:
            for rf in rfs:
                for rf_t in rf_ts:
                    for freeze_bn in freeze_bns:
                        for pooling_mode in pooling_modes:
                            assert pooling_mode in ['adaptive_max', 'adaptive_avg']
                            queue = next(queues_buffer)

                            p_text = f'{rf}-{rf_t}'
                            freeze_text = 'f_bn' if freeze_bn else 'nof_bn'

                            tags = [roi, layer, p_text, pooling_mode, freeze_text]
                            cloned_task = Task.clone(source_task=template_task,
                                                     name=','.join(tags),
                                                     parent=template_task.id)

                            cloned_task.add_tags(tags)

                            cloned_task_parameters = cloned_task.get_parameters()
                            # cloned_task_parameters['rois'] = [roi]
                            cloned_task_parameters['Args/rois'] = roi
                            cloned_task_parameters['Args/track'] = 'mini_track'
                            # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
                            cloned_task_parameters['Args/learning_rate'] = 1e-4
                            cloned_task_parameters['Args/step_lr_epochs'] = [4]
                            cloned_task_parameters['Args/step_lr_ratio'] = 0.5
                            cloned_task_parameters['Args/batch_size'] = batch_size if not freeze_bn else 8
                            cloned_task_parameters['Args/accumulate_grad_batches'] = 1 if not freeze_bn else int(
                                batch_size / 8)
                            cloned_task_parameters['Args/num_layers'] = 1
                            cloned_task_parameters['Args/conv_size'] = 256
                            cloned_task_parameters['Args/first_layer_hidden'] = 2048
                            cloned_task_parameters['Args/layer_hidden'] = 2048
                            cloned_task_parameters['Args/debug'] = False
                            cloned_task_parameters['Args/freeze_bn'] = freeze_bn
                            cloned_task_parameters['Args/old_mix'] = True
                            cloned_task_parameters['Args/early_stop_epochs'] = 5
                            cloned_task_parameters['Args/max_epochs'] = 100
                            cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
                            cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                            # cloned_task_parameters['Args/pooling_mode'] = pooling_mode
                            for l in layer.split(','):
                                cloned_task_parameters[f'Args/{l}_pooling_mode'] = pooling_mode
                                cloned_task_parameters[f'Args/pooling_size_{l}'] = rf
                                cloned_task_parameters[f'Args/pooling_size_t_{l}'] = rf_t
                            cloned_task_parameters['Args/backbone_type'] = 'i3d_rgb'
                            cloned_task_parameters['Args/final_fusion'] = 'concat'
                            cloned_task_parameters['Args/pyramid_layers'] = layer
                            cloned_task_parameters['Args/pathways'] = 'none'
                            cloned_task_parameters['Args/val_check_interval'] = 1.0
                            cloned_task_parameters['Args/val_ratio'] = 0.1
                            cloned_task_parameters['Args/save_checkpoints'] = True
                            cloned_task_parameters[
                                'Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'

                            cloned_task.set_parameters(cloned_task_parameters)
                            print('Experiment set with parameters {}'.format(cloned_task_parameters))

                            # enqueue the task for execution
                            Task.enqueue(cloned_task.id, queue_name=queue)
                            print('Experiment id={} enqueue for execution'.format(cloned_task.id))

                            task_ids.append(cloned_task.id)


def start_tasks_no_pooling(rois, layers, freeze_bns, pooling_modes, batch_size=32):
    for roi in rois:
        for layer in layers:
            for freeze_bn in freeze_bns:
                for pooling_mode in pooling_modes:
                    assert pooling_mode == 'no'
                    queue = next(queues_buffer)

                    freeze_text = 'f_bn' if freeze_bn else 'nof_bn'

                    tags = [roi, layer, pooling_mode, freeze_text]
                    cloned_task = Task.clone(source_task=template_task,
                                             name=','.join(tags),
                                             parent=template_task.id)

                    cloned_task.add_tags(tags)

                    cloned_task_parameters = cloned_task.get_parameters()
                    # cloned_task_parameters['rois'] = [roi]
                    cloned_task_parameters['Args/rois'] = roi
                    cloned_task_parameters['Args/track'] = 'mini_track'
                    # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
                    cloned_task_parameters['Args/learning_rate'] = 1e-4
                    cloned_task_parameters['Args/step_lr_epochs'] = [4]
                    cloned_task_parameters['Args/step_lr_ratio'] = 0.5
                    cloned_task_parameters['Args/batch_size'] = batch_size if not freeze_bn else 8
                    cloned_task_parameters['Args/accumulate_grad_batches'] = 1 if not freeze_bn else int(
                        batch_size / 8)
                    cloned_task_parameters['Args/num_layers'] = 1
                    cloned_task_parameters['Args/conv_size'] = 256
                    cloned_task_parameters['Args/first_layer_hidden'] = 2048
                    cloned_task_parameters['Args/layer_hidden'] = 2048
                    cloned_task_parameters['Args/debug'] = False
                    cloned_task_parameters['Args/freeze_bn'] = freeze_bn
                    cloned_task_parameters['Args/old_mix'] = True
                    cloned_task_parameters['Args/early_stop_epochs'] = 5
                    cloned_task_parameters['Args/max_epochs'] = 100
                    cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
                    cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                    # cloned_task_parameters['Args/pooling_mode'] = pooling_mode
                    for l in layer.split(','):
                        cloned_task_parameters[f'Args/{l}_pooling_mode'] = pooling_mode
                    cloned_task_parameters['Args/backbone_type'] = 'i3d_rgb'
                    cloned_task_parameters['Args/final_fusion'] = 'concat'
                    cloned_task_parameters['Args/pyramid_layers'] = layer
                    cloned_task_parameters['Args/pathways'] = 'none'
                    cloned_task_parameters['Args/val_check_interval'] = 1.0
                    cloned_task_parameters['Args/val_ratio'] = 0.1
                    cloned_task_parameters['Args/save_checkpoints'] = True
                    cloned_task_parameters[
                        'Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'

                    cloned_task.set_parameters(cloned_task_parameters)
                    print('Experiment set with parameters {}'.format(cloned_task_parameters))

                    # enqueue the task for execution
                    Task.enqueue(cloned_task.id, queue_name=queue)
                    print('Experiment id={} enqueue for execution'.format(cloned_task.id))

                    task_ids.append(cloned_task.id)


# ## Freeze BN part 1
# # 16
# start_tasks_no_pooling(
#     rois=['EBA', 'LOC', 'PPA', 'FFA', 'V1', 'V2', 'V3', 'V4'],
#     layers=['x3'],
#     freeze_bns=[False],
#     pooling_modes=['no'],
#     batch_size=24
# )
#
# # 2
# start_tasks_no_pooling(
#     rois=['STS'],
#     layers=['x3'],
#     freeze_bns=[False],
#     pooling_modes=['no'],
#     batch_size=20
# )
#
# # 18
# start_tasks_adaptive_pooling(
#     rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS', 'V1', 'V2', 'V3', 'V4'],
#     layers=['x4'],
#     rfs=[1],
#     rf_ts=[1],
#     freeze_bns=[False],
#     pooling_modes=['adaptive_avg'],
#     batch_size=32
# )
#
# # 12
# start_tasks_spp(
#     rois=['V1', 'V2', 'V3', 'V4'],
#     layers=['x1'],
#     ps=[
#         [2, 4, 7],
#         [3, 6, 9],
#         [3, 5, 11],
#     ],
#     freeze_bns=[False],
#     pooling_modes=['max'],
#     batch_size=24
# )
#
# # 24
# start_tasks_spp(
#     rois=['V1', 'V2', 'V3', 'V4'],
#     layers=['x2', 'x3'],
#     ps=[
#         [2, 4, 7],
#         [3, 6, 9],
#         [3, 5, 11],
#     ],
#     freeze_bns=[False],
#     pooling_modes=['avg'],
#     batch_size=24
# )
#
# # 15
# start_tasks_spp(
#     rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS'],
#     layers=['x1'],
#     ps=[
#         [1, 2, 3],
#         [1, 3, 5],
#         [2, 3, 7],
#     ],
#     freeze_bns=[False],
#     pooling_modes=['max'],
#     batch_size=24
# )
#
# # 30
# start_tasks_spp(
#     rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS'],
#     layers=['x2', 'x3'],
#     ps=[
#         [1, 2, 3],
#         [1, 3, 5],
#         [2, 3, 7],
#     ],
#     freeze_bns=[False],
#     pooling_modes=['avg'],
#     batch_size=24
# )
#
# ## Freeze BN part 2
#
# # 16
# start_tasks_no_pooling(
#     rois=['EBA', 'LOC', 'PPA', 'FFA', 'V1', 'V2', 'V3', 'V4'],
#     layers=['x3'],
#     freeze_bns=[True],
#     pooling_modes=['no'],
#     batch_size=32
# )
#
# # 2
# start_tasks_no_pooling(
#     rois=['STS'],
#     layers=['x3'],
#     freeze_bns=[True],
#     pooling_modes=['no'],
#     batch_size=32
# )
#
# # 18
# start_tasks_adaptive_pooling(
#     rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS', 'V1', 'V2', 'V3', 'V4'],
#     layers=['x4'],
#     rfs=[1],
#     rf_ts=[1],
#     freeze_bns=[True],
#     pooling_modes=['adaptive_avg'],
#     batch_size=32
# )
#
# # 12
# start_tasks_spp(
#     rois=['V1', 'V2', 'V3', 'V4'],
#     layers=['x1'],
#     ps=[
#         [2, 4, 7],
#         [3, 6, 9],
#         [3, 5, 11],
#     ],
#     freeze_bns=[True],
#     pooling_modes=['max'],
#     batch_size=32
# )
#
# # 24
# start_tasks_spp(
#     rois=['V1', 'V2', 'V3', 'V4'],
#     layers=['x2', 'x3'],
#     ps=[
#         [2, 4, 7],
#         [3, 6, 9],
#         [3, 5, 11],
#     ],
#     freeze_bns=[True],
#     pooling_modes=['avg'],
#     batch_size=32
# )
#
# # 15
# start_tasks_spp(
#     rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS'],
#     layers=['x1'],
#     ps=[
#         [1, 2, 3],
#         [1, 3, 5],
#         [2, 3, 7],
#     ],
#     freeze_bns=[True],
#     pooling_modes=['max'],
#     batch_size=32
# )
#
# # 30
# start_tasks_spp(
#     rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS'],
#     layers=['x2', 'x3'],
#     ps=[
#         [1, 2, 3],
#         [1, 3, 5],
#         [2, 3, 7],
#     ],
#     freeze_bns=[True],
#     pooling_modes=['avg'],
#     batch_size=32
# )


## Freeze BN part 1
# 9
start_tasks_no_pooling(
    rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS', 'V1', 'V2', 'V3', 'V4'],
    layers=['x4'],
    freeze_bns=[False],
    pooling_modes=['no'],
    batch_size=32
)

# 18
start_tasks_adaptive_pooling(
    rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS', 'V1', 'V2', 'V3', 'V4'],
    layers=['x4'],
    rfs=[2, 3, 4, 5, 6],
    rf_ts=[1],
    freeze_bns=[False],
    pooling_modes=['adaptive_avg'],
    batch_size=32
)

# 4
start_tasks_spp(
    rois=['V1', 'V2', 'V3', 'V4'],
    layers=['x1'],
    ps=[
        [1, 2, 3],
    ],
    freeze_bns=[False],
    pooling_modes=['max'],
    batch_size=32
)

# 4
start_tasks_spp(
    rois=['V1', 'V2', 'V3', 'V4'],
    layers=['x2', 'x3'],
    ps=[
        [1, 2, 3],
    ],
    freeze_bns=[False],
    pooling_modes=['avg'],
    batch_size=32
)

# 15
start_tasks_spp(
    rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS'],
    layers=['x1'],
    ps=[
        [2, 4, 7],
        [3, 6, 9],
        [5, 7, 11],
    ],
    freeze_bns=[False],
    pooling_modes=['max'],
    batch_size=24
)

# 30
start_tasks_spp(
    rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS'],
    layers=['x2', 'x3'],
    ps=[
        [2, 4, 7],
        [3, 6, 9],
        [5, 7, 11],
    ],
    freeze_bns=[False],
    pooling_modes=['avg'],
    batch_size=24
)

## Freeze BN part 2

# 9
start_tasks_no_pooling(
    rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS', 'V1', 'V2', 'V3', 'V4'],
    layers=['x4'],
    freeze_bns=[True],
    pooling_modes=['no'],
    batch_size=32
)

# 18
start_tasks_adaptive_pooling(
    rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS', 'V1', 'V2', 'V3', 'V4'],
    layers=['x4'],
    rfs=[2, 3, 4, 5, 6],
    rf_ts=[1],
    freeze_bns=[True],
    pooling_modes=['adaptive_avg'],
    batch_size=32
)

# 4
start_tasks_spp(
    rois=['V1', 'V2', 'V3', 'V4'],
    layers=['x1'],
    ps=[
        [1, 2, 3],
    ],
    freeze_bns=[True],
    pooling_modes=['max'],
    batch_size=32
)

# 4
start_tasks_spp(
    rois=['V1', 'V2', 'V3', 'V4'],
    layers=['x2', 'x3'],
    ps=[
        [1, 2, 3],
    ],
    freeze_bns=[True],
    pooling_modes=['avg'],
    batch_size=32
)

# 15
start_tasks_spp(
    rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS'],
    layers=['x1'],
    ps=[
        [2, 4, 7],
        [3, 6, 9],
        [5, 7, 11],
    ],
    freeze_bns=[True],
    pooling_modes=['max'],
    batch_size=24
)

# 30
start_tasks_spp(
    rois=['EBA', 'LOC', 'PPA', 'FFA', 'STS'],
    layers=['x2', 'x3'],
    ps=[
        [2, 4, 7],
        [3, 6, 9],
        [5, 7, 11],
    ],
    freeze_bns=[True],
    pooling_modes=['avg'],
    batch_size=24
)


print(task_ids)
