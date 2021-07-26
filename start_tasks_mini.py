import itertools

from clearml import Task

PROJECT_NAME = 'Algonauts V2 flow'
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

rois = ['LOC', 'FFA', 'STS', 'EBA', 'PPA']

for roi in rois:
    queue = next(queues_buffer)

    cloned_task = Task.clone(source_task=template_task,
                             name=template_task.name + f'{roi}',
                             parent=template_task.id)

    cloned_task.add_tags([roi, 'none', 'pyramid', 'x2,x3,x4', 'x2_spp', 'x3_no', 'x4_avg'])

    cloned_task_parameters = cloned_task.get_parameters()
    # cloned_task_parameters['rois'] = [roi]
    cloned_task_parameters['Args/rois'] = roi
    cloned_task_parameters['Args/track'] = 'mini_track'
    # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
    cloned_task_parameters['Args/batch_size'] = 8
    cloned_task_parameters['Args/accumulate_grad_batches'] = 4
    cloned_task_parameters['Args/num_layers'] = 1
    cloned_task_parameters['Args/conv_size'] = 256
    cloned_task_parameters['Args/first_layer_hidden'] = 1024
    cloned_task_parameters['Args/layer_hidden'] = 2048
    cloned_task_parameters['Args/debug'] = False
    cloned_task_parameters['Args/freeze_bn'] = True
    cloned_task_parameters['Args/early_stop_epochs'] = 10
    cloned_task_parameters['Args/max_epochs'] = 100
    cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
    cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
    # cloned_task_parameters['Args/x1_pooling_mode'] = 'spp'
    cloned_task_parameters['Args/x2_pooling_mode'] = 'spp'
    cloned_task_parameters['Args/pooling_mode'] = 'max'
    cloned_task_parameters['Args/x3_pooling_mode'] = 'no'
    cloned_task_parameters['Args/x4_pooling_mode'] = 'avg'
    cloned_task_parameters['Args/backbone_type'] = 'all'
    cloned_task_parameters['Args/fc_fusion'] = 'concat'
    cloned_task_parameters['Args/pyramid_layers'] = 'x2,x3,x4'
    cloned_task_parameters['Args/additional_features'] = 'i3d_flow'
    cloned_task_parameters['Args/pathways'] = 'none'
    cloned_task_parameters['Args/aux_loss_weight'] = 0.0
    cloned_task_parameters['Args/val_check_interval'] = 0.5
    cloned_task_parameters['Args/val_ratio'] = 0.1
    cloned_task_parameters['Args/save_checkpoints'] = True
    cloned_task_parameters['Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'
    cloned_task_parameters['Args/voxel_wise'] = False

    cloned_task.set_parameters(cloned_task_parameters)
    print('Experiment set with parameters {}'.format(cloned_task_parameters))

    # enqueue the task for execution
    Task.enqueue(cloned_task.id, queue_name=queue)
    print('Experiment id={} enqueue for execution'.format(cloned_task.id))

    task_ids.append(cloned_task.id)



rois = ['V4']

for roi in rois:
    queue = next(queues_buffer)

    cloned_task = Task.clone(source_task=template_task,
                             name=template_task.name + f'{roi}',
                             parent=template_task.id)

    cloned_task.add_tags([roi, 'none', 'pyramid', 'x2,x3,x4', 'x2_spp', 'x3_no', 'x4_avg'])

    cloned_task_parameters = cloned_task.get_parameters()
    # cloned_task_parameters['rois'] = [roi]
    cloned_task_parameters['Args/rois'] = roi
    cloned_task_parameters['Args/track'] = 'mini_track'
    # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
    cloned_task_parameters['Args/batch_size'] = 24
    cloned_task_parameters['Args/accumulate_grad_batches'] = 1
    cloned_task_parameters['Args/num_layers'] = 1
    cloned_task_parameters['Args/conv_size'] = 256
    cloned_task_parameters['Args/first_layer_hidden'] = 1024
    cloned_task_parameters['Args/layer_hidden'] = 2048
    cloned_task_parameters['Args/debug'] = False
    cloned_task_parameters['Args/freeze_bn'] = False
    cloned_task_parameters['Args/early_stop_epochs'] = 10
    cloned_task_parameters['Args/max_epochs'] = 100
    cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
    cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
    # cloned_task_parameters['Args/x1_pooling_mode'] = 'spp'
    cloned_task_parameters['Args/x2_pooling_mode'] = 'spp'
    cloned_task_parameters['Args/pooling_mode'] = 'max'
    cloned_task_parameters['Args/x3_pooling_mode'] = 'no'
    cloned_task_parameters['Args/x4_pooling_mode'] = 'avg'
    cloned_task_parameters['Args/backbone_type'] = 'all'
    cloned_task_parameters['Args/fc_fusion'] = 'concat'
    cloned_task_parameters['Args/pyramid_layers'] = 'x2,x3,x4'
    cloned_task_parameters['Args/additional_features'] = 'i3d_flow'
    cloned_task_parameters['Args/pathways'] = 'none'
    cloned_task_parameters['Args/aux_loss_weight'] = 0.0
    cloned_task_parameters['Args/val_check_interval'] = 0.5
    cloned_task_parameters['Args/val_ratio'] = 0.1
    cloned_task_parameters['Args/save_checkpoints'] = True
    cloned_task_parameters['Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'
    cloned_task_parameters['Args/voxel_wise'] = False



rois = ['V1', 'V2', 'V3']

for roi in rois:
    queue = next(queues_buffer)

    cloned_task = Task.clone(source_task=template_task,
                             name=template_task.name + f'{roi}',
                             parent=template_task.id)

    cloned_task.add_tags([roi, 'none', 'pyramid', 'x3', 'x3_no'])

    cloned_task_parameters = cloned_task.get_parameters()
    # cloned_task_parameters['rois'] = [roi]
    cloned_task_parameters['Args/rois'] = roi
    cloned_task_parameters['Args/track'] = 'mini_track'
    # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
    cloned_task_parameters['Args/batch_size'] = 24
    cloned_task_parameters['Args/accumulate_grad_batches'] = 1
    cloned_task_parameters['Args/num_layers'] = 1
    cloned_task_parameters['Args/conv_size'] = 256
    cloned_task_parameters['Args/first_layer_hidden'] = 1024
    cloned_task_parameters['Args/layer_hidden'] = 2048
    cloned_task_parameters['Args/debug'] = False
    cloned_task_parameters['Args/freeze_bn'] = False
    cloned_task_parameters['Args/early_stop_epochs'] = 10
    cloned_task_parameters['Args/max_epochs'] = 100
    cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
    cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
    # cloned_task_parameters['Args/x1_pooling_mode'] = 'spp'
    cloned_task_parameters['Args/x2_pooling_mode'] = 'spp'
    cloned_task_parameters['Args/pooling_mode'] = 'max'
    cloned_task_parameters['Args/x3_pooling_mode'] = 'no'
    cloned_task_parameters['Args/x4_pooling_mode'] = 'avg'
    cloned_task_parameters['Args/backbone_type'] = 'all'
    cloned_task_parameters['Args/fc_fusion'] = 'concat'
    cloned_task_parameters['Args/pyramid_layers'] = 'x3'
    cloned_task_parameters['Args/additional_features'] = 'i3d_flow'
    cloned_task_parameters['Args/pathways'] = 'none'
    cloned_task_parameters['Args/aux_loss_weight'] = 0.0
    cloned_task_parameters['Args/val_check_interval'] = 0.5
    cloned_task_parameters['Args/val_ratio'] = 0.1
    cloned_task_parameters['Args/save_checkpoints'] = True
    cloned_task_parameters['Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'
    cloned_task_parameters['Args/voxel_wise'] = False


print(task_ids)