import itertools

from clearml import Task

PROJECT_NAME = 'Algonauts Mini V2'
BASE_TASK = 'task template'

task = Task.init(project_name=PROJECT_NAME,
                 task_name='Task Manager',
                 task_type=Task.TaskTypes.optimizer)

template_task = Task.get_task(project_name=PROJECT_NAME,
                              task_name=BASE_TASK)

rois = ['LOC', 'FFA', 'STS', 'EBA', 'PPA', 'V1', 'V2', 'V3', 'V4']

available_devices = {
    '16': [0, 1],
}

queue_names = []
for k, vs in available_devices.items():
    for v in vs:
        queue_names.append(f'{k}-{v}')
queues_buffer = itertools.cycle(queue_names)

task_ids = []

pooling_schs = ['spp', 'avg']
global_pooling = False
for pooling_sch in pooling_schs:
    for roi in rois:
        queue = next(queues_buffer)

        cloned_task = Task.clone(source_task=template_task,
                                 name=template_task.name + f' {roi} {pooling_sch}',
                                 parent=template_task.id)

        cloned_task.add_tags([roi, pooling_sch, 'pyramid'])

        cloned_task_parameters = cloned_task.get_parameters()
        # cloned_task_parameters['rois'] = [roi]
        cloned_task_parameters['Args/roi'] = roi
        # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
        cloned_task_parameters['Args/batch_size'] = 24
        cloned_task_parameters['Args/conv_size'] = 256
        cloned_task_parameters['Args/layer_hidden'] = 2048
        cloned_task_parameters['Args/debug'] = False
        cloned_task_parameters['Args/early_stop_epochs'] = 15
        cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
        cloned_task_parameters['Args/x1_pooling_mode'] = pooling_sch
        cloned_task_parameters['Args/x2_pooling_mode'] = pooling_sch
        cloned_task_parameters['Args/x3_pooling_mode'] = pooling_sch
        cloned_task_parameters['Args/x4_pooling_mode'] = pooling_sch
        cloned_task_parameters['Args/backbone_type'] = 'all'
        cloned_task_parameters['Args/fc_fusion'] = 'concat'
        cloned_task_parameters['Args/pyramid_layers'] = 'x1,x2,x3,x4'
        cloned_task_parameters['Args/pathways'] = 'topdown'
        cloned_task_parameters['Args/aux_loss_weight'] = 0.5
        cloned_task_parameters['Args/val_check_interval'] = 0.25
        cloned_task_parameters['Args/save_checkpoints'] = True
        cloned_task_parameters['Args/predictions_dir'] = f'/home/huze/.cache/predictions/v2_pyramid_{pooling_sch}/'
        # cloned_task_parameters['Args/predictions_dir'] = f'/home/huze/.cache/predictions/v1_global_pool/'

        # put back into the new cloned task
        cloned_task.set_parameters(cloned_task_parameters)
        print('Experiment set with parameters {}'.format(cloned_task_parameters))

        # enqueue the task for execution
        Task.enqueue(cloned_task.id, queue_name=queue)
        print('Experiment id={} enqueue for execution'.format(cloned_task.id))

        task_ids.append(cloned_task.id)

print(task_ids)


# pooling_sch = ''
# global_pooling = True
# for roi in rois:
#     queue = next(queues_buffer)
#
#     cloned_task = Task.clone(source_task=template_task,
#                              name=template_task.name + f' {roi} {pooling_sch}',
#                              parent=template_task.id)
#
#     cloned_task.add_tags([roi, 'global_pool', 'x2'])
#
#     cloned_task_parameters = cloned_task.get_parameters()
#     # cloned_task_parameters['rois'] = [roi]
#     cloned_task_parameters['Args/roi'] = roi
#     # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
#     cloned_task_parameters['Args/batch_size'] = 32
#     cloned_task_parameters['Args/conv_size'] = 512
#     cloned_task_parameters['Args/layer_hidden'] = 2048
#     cloned_task_parameters['Args/debug'] = False
#     # cloned_task_parameters['Args/early_stop_epochs'] = 15 if pooling_sch in ['avg', 'max'] else 5
#     cloned_task_parameters['Args/early_stop_epochs'] = 15
#     cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
#     cloned_task_parameters['Args/pooling_mode'] = pooling_sch
#     cloned_task_parameters['Args/global_pooling'] = global_pooling
#     cloned_task_parameters['Args/save_checkpoints'] = True
#     # cloned_task_parameters['Args/predictions_dir'] = f'/home/huze/.cache/predictions/v1_{pooling_sch}/'
#     cloned_task_parameters['Args/predictions_dir'] = f'/home/huze/.cache/predictions/v1_x2_global_pool/'
#
#     # put back into the new cloned task
#     cloned_task.set_parameters(cloned_task_parameters)
#     print('Experiment set with parameters {}'.format(cloned_task_parameters))
#
#     # enqueue the task for execution
#     Task.enqueue(cloned_task.id, queue_name=queue)
#     print('Experiment id={} enqueue for execution'.format(cloned_task.id))
#
#     task_ids.append(cloned_task.id)

print(task_ids)
