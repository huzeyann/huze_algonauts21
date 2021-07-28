import itertools

from clearml import Task

PROJECT_NAME = 'Algonauts BDCN debug'
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

# rois = ['STS', 'FFA', 'PPA', 'LOC', 'EBA', 'V2', 'V3', 'V4', 'V1']

num_frames = [3, 8, 16]
video_sizes = [32, 64, 128]
bdcn_pool_sizes = [1, 2, 4]
rois = ['V1', 'EBA', 'V4', 'LOC']

num_frames = [8, ]
video_sizes = [64, 128]
bdcn_pool_sizes = [1, 2, 4, 6, 8, 16]

# rois = ['V1', 'EBA', 'STS', 'FFA', 'V2', 'V3', 'V4', 'PPA', 'LOC']
rois = ['V1']
for roi in rois:
    for num_frame in num_frames:
        for video_size in video_sizes:
            for bdcn_pool_size in bdcn_pool_sizes:
                if video_sizes == 64:
                    if bdcn_pool_size == 16:
                        continue

                queue = next(queues_buffer)

                cloned_task = Task.clone(source_task=template_task,
                                         name=template_task.name + f'{roi}',
                                         parent=template_task.id)

                cloned_task.add_tags([roi, f'{num_frame}_frames', f'{video_size}_resolution',
                                      f'{bdcn_pool_size}_pooling'])

                cloned_task_parameters = cloned_task.get_parameters()
                # cloned_task_parameters['rois'] = [roi]
                cloned_task_parameters['Args/rois'] = roi
                cloned_task_parameters['Args/batch_size'] = 32
                cloned_task_parameters['Args/video_size'] = video_size
                cloned_task_parameters['Args/video_frames'] = num_frame
                cloned_task_parameters['Args/accumulate_grad_batches'] = 1
                cloned_task_parameters['Args/layer_hidden'] = 1024
                cloned_task_parameters['Args/debug'] = False
                cloned_task_parameters['Args/freeze_bn'] = False
                cloned_task_parameters['Args/early_stop_epochs'] = 5
                cloned_task_parameters['Args/max_epochs'] = 100
                cloned_task_parameters['Args/backbone_freeze_epochs'] = 5
                cloned_task_parameters['Args/bdcn_pool_size'] = bdcn_pool_size
                cloned_task_parameters['Args/backbone_lr_ratio'] = 1 / num_frame
                cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                # cloned_task_parameters['Args/x1_pooling_mode'] = 'spp'
                cloned_task_parameters['Args/pooling_mode'] = 'max'
                cloned_task_parameters['Args/backbone_type'] = 'bdcn_edge'
                cloned_task_parameters['Args/preprocessing_type'] = 'bdcn'
                cloned_task_parameters['Args/val_check_interval'] = 0.5
                cloned_task_parameters['Args/val_ratio'] = 0.1
                cloned_task_parameters['Args/save_checkpoints'] = False
                # cloned_task_parameters['Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'
                # cloned_task_parameters['Args/voxel_wise'] = False

                cloned_task.set_parameters(cloned_task_parameters)
                print('Experiment set with parameters {}'.format(cloned_task_parameters))

                # enqueue the task for execution
                Task.enqueue(cloned_task.id, queue_name=queue)
                print('Experiment id={} enqueue for execution'.format(cloned_task.id))

                task_ids.append(cloned_task.id)

print(task_ids)
