import itertools

from clearml import Task

PROJECT_NAME = 'Algonauts audio'
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


def start_tasks_mini(rois, batch_size=64):
    for roi in rois:
        queue = next(queues_buffer)

        tags = [roi]
        cloned_task = Task.clone(source_task=template_task,
                                 name='task ' + ','.join(tags),
                                 parent=template_task.id)

        cloned_task.add_tags(tags)

        cloned_task_parameters = cloned_task.get_parameters()
        # cloned_task_parameters['rois'] = [roi]
        cloned_task_parameters['Args/rois'] = roi
        cloned_task_parameters['Args/track'] = 'mini_track'
        # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
        cloned_task_parameters['Args/learning_rate'] = 1e-4
        cloned_task_parameters['Args/batch_size'] = int(batch_size / 8)
        cloned_task_parameters['Args/accumulate_grad_batches'] = 8
        cloned_task_parameters['Args/num_layers'] = 2
        cloned_task_parameters['Args/layer_hidden'] = 2048
        cloned_task_parameters['Args/debug'] = False
        cloned_task_parameters['Args/early_stop_epochs'] = 15
        cloned_task_parameters['Args/max_epochs'] = 100
        cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
        cloned_task_parameters['Args/backbone_type'] = 'vggish'
        cloned_task_parameters['Args/preprocessing_type'] = 'vggish'
        cloned_task_parameters['Args/load_from_np'] = True
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


def start_tasks_full(rois, batch_size=64):
    for roi in rois:
        queue = next(queues_buffer)

        tags = [roi]
        cloned_task = Task.clone(source_task=template_task,
                                 name=','.join(tags),
                                 parent=template_task.id)

        cloned_task.add_tags(tags)

        cloned_task_parameters = cloned_task.get_parameters()
        # cloned_task_parameters['rois'] = [roi]
        cloned_task_parameters['Args/rois'] = roi
        cloned_task_parameters['Args/track'] = 'full_track'
        # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
        cloned_task_parameters['Args/learning_rate'] = 1e-4
        cloned_task_parameters['Args/batch_size'] = int(batch_size / 8)
        cloned_task_parameters['Args/accumulate_grad_batches'] = 8
        cloned_task_parameters['Args/num_layers'] = 2
        cloned_task_parameters['Args/layer_hidden'] = 2048
        cloned_task_parameters['Args/debug'] = False
        cloned_task_parameters['Args/early_stop_epochs'] = 15
        cloned_task_parameters['Args/max_epochs'] = 100
        cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
        cloned_task_parameters['Args/backbone_type'] = 'vggish'
        cloned_task_parameters['Args/preprocessing_type'] = 'vggish'
        cloned_task_parameters['Args/load_from_np'] = True
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


start_tasks_mini(
    rois=['V1', 'V2', 'V3', 'V4', 'EBA', 'LOC', 'PPA', 'FFA', 'STS'],
    batch_size=64
)

print(task_ids)
