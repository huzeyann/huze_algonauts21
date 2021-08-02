import itertools
from time import sleep

from clearml import Task

PROJECT_NAME = 'ROI LAYER search RF'
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

rois = ['V1', 'V2', 'V3', 'V4', 'EBA', 'LOC', 'PPA', 'FFA', 'STS']
layers = ['x1', 'x2', 'x3', 'x4']
combs = list(itertools.product(rois, layers))

step1_search_space = {('V1', 'x1'): [],
                      ('V1', 'x2'): [],
                      ('V1', 'x3'): [],
                      ('V1', 'x4'): [],
                      ('V2', 'x1'): [],
                      ('V2', 'x2'): [],
                      ('V2', 'x3'): [],
                      ('V2', 'x4'): [],
                      ('V3', 'x1'): [],
                      ('V3', 'x2'): [],
                      ('V3', 'x3'): [],
                      ('V3', 'x4'): [],
                      ('V4', 'x1'): [],
                      ('V4', 'x2'): [],
                      ('V4', 'x3'): [],
                      ('V4', 'x4'): [],
                      ('EBA', 'x1'): [],
                      ('EBA', 'x2'): [],
                      ('EBA', 'x3'): [],
                      ('EBA', 'x4'): [],
                      ('LOC', 'x1'): [],
                      ('LOC', 'x2'): [],
                      ('LOC', 'x3'): [],
                      ('LOC', 'x4'): [],
                      ('PPA', 'x1'): [6, 12],
                      ('PPA', 'x2'): [3, 6, 12],
                      ('PPA', 'x3'): [3, 6, 12],
                      ('PPA', 'x4'): [3, 9],
                      ('FFA', 'x1'): [6, 9, 12],
                      ('FFA', 'x2'): [3, 6, 9, 12],
                      ('FFA', 'x3'): [3, 12],
                      ('FFA', 'x4'): [3, 6, 9],
                      ('STS', 'x1'): [3, 6, 9, 12],
                      ('STS', 'x2'): [3, 6, 9],
                      ('STS', 'x3'): [6, 9],
                      ('STS', 'x4'): [3, 6]}

step2_search_space = {('V1', 'x1'): [1, 7, 11],
                      ('V1', 'x2'): [5, 7, 11, 13],
                      ('V1', 'x3'): [1, 5, 7, 11, 13],
                      ('V1', 'x4'): [1, 5],
                      ('V2', 'x1'): [7, 11, 13],
                      ('V2', 'x2'): [1, 5, 7, 11, 13],
                      ('V2', 'x3'): [1, 7, 11],
                      ('V2', 'x4'): [1, 5, 7],
                      ('V3', 'x1'): [1, 5, 7, 11, 13],
                      ('V3', 'x2'): [5, 11, 13],
                      ('V3', 'x3'): [1, 5, 13],
                      ('V3', 'x4'): [1, 5],
                      ('V4', 'x1'): [5, 7, 11, 13],
                      ('V4', 'x2'): [1, 7, 11],
                      ('V4', 'x3'): [1, 5, 11],
                      ('V4', 'x4'): [1, 5, 7],
                      ('EBA', 'x1'): [7, 11, 13],
                      ('EBA', 'x2'): [1, 13],
                      ('EBA', 'x3'): [1, 7],
                      ('EBA', 'x4'): [5, 7],
                      ('LOC', 'x1'): [1, 5, 7, 11],
                      ('LOC', 'x2'): [1, 5, 7, 13],
                      ('LOC', 'x3'): [5, 13],
                      ('LOC', 'x4'): [7],
                      ('PPA', 'x1'): [5, 7, 13],
                      ('PPA', 'x2'): [5, 7, 11, 13],
                      ('PPA', 'x3'): [1, 5, 7, 11, 13],
                      ('PPA', 'x4'): [5],
                      ('FFA', 'x1'): [1, 5, 7],
                      ('FFA', 'x2'): [7, 11, 13],
                      ('FFA', 'x3'): [1, 5, 7, 11],
                      ('FFA', 'x4'): [1, 7],
                      ('STS', 'x1'): [7, 11, 13],
                      ('STS', 'x2'): [1, 5, 7, 11],
                      ('STS', 'x3'): [1, 5, 11, 13],
                      ('STS', 'x4'): [1, 5]}

step3_search_space = {('V1', 'x1'): [2, 4, 8, 10, 14],
                      ('V1', 'x2'): [4, 8, 10, 14],
                      ('V1', 'x3'): [4, 8, 10, 14],
                      ('V1', 'x4'): [4, 8],
                      ('V2', 'x1'): [2, 4, 8],
                      ('V2', 'x2'): [4, 14],
                      ('V2', 'x3'): [2, 4, 8, 10, 14],
                      ('V2', 'x4'): [2, 4, 8],
                      ('V3', 'x1'): [2, 4],
                      ('V3', 'x2'): [2, 8, 10, 14],
                      ('V3', 'x3'): [2, 8, 10, 14],
                      ('V3', 'x4'): [2, 8],
                      ('V4', 'x1'): [8, 10, 14],
                      ('V4', 'x2'): [2, 8, 10, 14],
                      ('V4', 'x3'): [2, 4, 8, 10, 14],
                      ('V4', 'x4'): [4],
                      ('EBA', 'x1'): [2, 4, 8, 10, 14],
                      ('EBA', 'x2'): [2, 4, 8, 10, 14],
                      ('EBA', 'x3'): [2, 4, 8, 10],
                      ('EBA', 'x4'): [2, 8],
                      ('LOC', 'x1'): [8, 14],
                      ('LOC', 'x2'): [2, 8, 10, 14],
                      ('LOC', 'x3'): [2, 8, 10, 14],
                      ('LOC', 'x4'): [2, 4],
                      ('PPA', 'x1'): [2, 4, 8, 14],
                      ('PPA', 'x2'): [4, 8, 10, 14],
                      ('PPA', 'x3'): [2, 4, 10],
                      ('PPA', 'x4'): [4, 8],
                      ('FFA', 'x1'): [2, 8, 10, 14],
                      ('FFA', 'x2'): [4, 8, 10],
                      ('FFA', 'x3'): [2, 4, 8, 10, 14],
                      ('FFA', 'x4'): [2, 4],
                      ('STS', 'x1'): [2, 4, 8, 14],
                      ('STS', 'x2'): [2, 4, 10, 14],
                      ('STS', 'x3'): [2, 4, 8, 10, 14],
                      ('STS', 'x4'): [8]}
steps = [step1_search_space, step2_search_space, step3_search_space]
first_call = True
for step in steps:
    for (roi, layer), rfs in step.items():
        for rf in rfs:
            queue = next(queues_buffer)

            cloned_task = Task.clone(source_task=template_task,
                                     name=f'{roi},{layer},{rf}',
                                     parent=template_task.id)

            cloned_task.add_tags([roi, layer, str(rf)])

            cloned_task_parameters = cloned_task.get_parameters()
            # cloned_task_parameters['rois'] = [roi]
            cloned_task_parameters['Args/rois'] = roi
            cloned_task_parameters['Args/track'] = 'mini_track'
            # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
            cloned_task_parameters['Args/batch_size'] = 24
            cloned_task_parameters['Args/accumulate_grad_batches'] = 1
            cloned_task_parameters['Args/num_layers'] = 1
            cloned_task_parameters['Args/conv_size'] = 256
            cloned_task_parameters['Args/first_layer_hidden'] = 2048
            cloned_task_parameters['Args/layer_hidden'] = 2048
            cloned_task_parameters['Args/debug'] = False
            cloned_task_parameters['Args/freeze_bn'] = False
            cloned_task_parameters['Args/early_stop_epochs'] = 5
            cloned_task_parameters['Args/max_epochs'] = 100
            cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
            cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
            # cloned_task_parameters['Args/x1_pooling_mode'] = 'spp'
            cloned_task_parameters[f'Args/{layer}_pooling_mode'] = 'adaptive_max' if layer != 'x4' else 'adaptive_avg'
            cloned_task_parameters[f'Args/pooling_size_{layer}'] = rf
            cloned_task_parameters['Args/backbone_type'] = 'i3d_rgb'
            cloned_task_parameters['Args/final_fusion'] = 'concat'
            cloned_task_parameters['Args/pyramid_layers'] = layer
            cloned_task_parameters['Args/pathways'] = 'none'
            cloned_task_parameters['Args/aux_loss_weight'] = 0.0
            cloned_task_parameters['Args/val_check_interval'] = 0.5
            cloned_task_parameters['Args/val_ratio'] = 0.1
            cloned_task_parameters['Args/save_checkpoints'] = False
            cloned_task_parameters['Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'

            cloned_task.set_parameters(cloned_task_parameters)
            print('Experiment set with parameters {}'.format(cloned_task_parameters))

            # enqueue the task for execution
            Task.enqueue(cloned_task.id, queue_name=queue)
            print('Experiment id={} enqueue for execution'.format(cloned_task.id))

            task_ids.append(cloned_task.id)

            if first_call:
                sleep(10)
                first_call = False
    #         break
    #     break
    # break

print(task_ids)
