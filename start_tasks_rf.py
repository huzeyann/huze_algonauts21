import itertools

from clearml import Task

PROJECT_NAME = 'Algonauts V2 adaptive_pooling search RF'
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

rois = ['STS', 'FFA', 'PPA', 'LOC', 'EBA', 'V2', 'V3', 'V4', 'V1']
rfs = [2, 3, 4, 6, 9, 12]


layers = ['x3']
for roi in rois:
    for layer in layers:
        queue = next(queues_buffer)

        cloned_task = Task.clone(source_task=template_task,
                                 name=template_task.name + f'{roi}',
                                 parent=template_task.id)

        cloned_task.add_tags([roi, layer, 'no'])

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
        cloned_task_parameters['Args/early_stop_epochs'] = 6
        cloned_task_parameters['Args/max_epochs'] = 100
        cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
        cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
        # cloned_task_parameters['Args/x1_pooling_mode'] = 'spp'
        cloned_task_parameters['Args/pooling_mode'] = 'max'
        cloned_task_parameters[f'Args/{layer}_pooling_mode'] = 'no'
        # cloned_task_parameters[f'Args/pooling_size'] = rf
        cloned_task_parameters['Args/backbone_type'] = 'all'
        cloned_task_parameters['Args/fc_fusion'] = 'concat'
        cloned_task_parameters['Args/pyramid_layers'] = layer
        # cloned_task_parameters['Args/additional_features'] = 'i3d_flow'
        cloned_task_parameters['Args/pathways'] = 'none'
        cloned_task_parameters['Args/aux_loss_weight'] = 0.0
        cloned_task_parameters['Args/val_check_interval'] = 0.5
        cloned_task_parameters['Args/val_ratio'] = 0.1
        cloned_task_parameters['Args/save_checkpoints'] = False
        cloned_task_parameters['Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'
        cloned_task_parameters['Args/voxel_wise'] = False

        cloned_task.set_parameters(cloned_task_parameters)
        print('Experiment set with parameters {}'.format(cloned_task_parameters))

        # enqueue the task for execution
        Task.enqueue(cloned_task.id, queue_name=queue)
        print('Experiment id={} enqueue for execution'.format(cloned_task.id))

        task_ids.append(cloned_task.id)

layers = ['x1', 'x3']
for roi in rois:
    for layer in layers:
        for rf in rfs:
            queue = next(queues_buffer)

            cloned_task = Task.clone(source_task=template_task,
                                     name=template_task.name + f'{roi}',
                                     parent=template_task.id)

            cloned_task.add_tags([roi, layer, f'rf_{rf}', 'max'])

            cloned_task_parameters = cloned_task.get_parameters()
            # cloned_task_parameters['rois'] = [roi]
            cloned_task_parameters['Args/rois'] = roi
            cloned_task_parameters['Args/track'] = 'mini_track'
            # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
            cloned_task_parameters['Args/batch_size'] = 32
            cloned_task_parameters['Args/accumulate_grad_batches'] = 1
            cloned_task_parameters['Args/num_layers'] = 1
            cloned_task_parameters['Args/conv_size'] = 256
            cloned_task_parameters['Args/first_layer_hidden'] = 2048
            cloned_task_parameters['Args/layer_hidden'] = 2048
            cloned_task_parameters['Args/debug'] = False
            cloned_task_parameters['Args/freeze_bn'] = False
            cloned_task_parameters['Args/early_stop_epochs'] = 6
            cloned_task_parameters['Args/max_epochs'] = 100
            cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
            cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
            # cloned_task_parameters['Args/x1_pooling_mode'] = 'spp'
            cloned_task_parameters['Args/pooling_mode'] = 'max'
            cloned_task_parameters[f'Args/{layer}_pooling_mode'] = 'adaptive_max'
            cloned_task_parameters[f'Args/pooling_size'] = rf
            cloned_task_parameters['Args/backbone_type'] = 'all'
            cloned_task_parameters['Args/fc_fusion'] = 'concat'
            cloned_task_parameters['Args/pyramid_layers'] = layer
            # cloned_task_parameters['Args/additional_features'] = 'i3d_flow'
            cloned_task_parameters['Args/pathways'] = 'none'
            cloned_task_parameters['Args/aux_loss_weight'] = 0.0
            cloned_task_parameters['Args/val_check_interval'] = 0.5
            cloned_task_parameters['Args/val_ratio'] = 0.1
            cloned_task_parameters['Args/save_checkpoints'] = False
            cloned_task_parameters['Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'
            cloned_task_parameters['Args/voxel_wise'] = False

            cloned_task.set_parameters(cloned_task_parameters)
            print('Experiment set with parameters {}'.format(cloned_task_parameters))

            # enqueue the task for execution
            Task.enqueue(cloned_task.id, queue_name=queue)
            print('Experiment id={} enqueue for execution'.format(cloned_task.id))

            task_ids.append(cloned_task.id)


print(task_ids)
