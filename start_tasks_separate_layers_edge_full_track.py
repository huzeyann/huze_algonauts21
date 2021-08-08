import itertools

from clearml import Task

PROJECT_NAME = 'Algonauts full_track model zoo'
BASE_TASK = 'task template'

task = Task.init(project_name=PROJECT_NAME,
                 task_name='Task Manager',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

template_task = Task.get_task(project_name=PROJECT_NAME,
                              task_name=BASE_TASK)

available_devices = {
    '57': [2, 3],
    '58': [2, 3],
    '59': [2, 3],
}

queue_names = []
for k, vs in available_devices.items():
    for v in vs:
        queue_names.append(f'{k}-{v}')
queues_buffer = itertools.cycle(queue_names)

task_ids = []


def start_tasks_spp(rois, video_sizes, num_frames, ps, freeze_bns, pooling_modes, num_lstm_layers, layer_hiddens,
                    batch_size=32):
    for video_size in video_sizes:
        for num_frame in num_frames:
            for p in ps:
                for freeze_bn in freeze_bns:
                    for pooling_mode in pooling_modes:
                        for num_lstm_layer in num_lstm_layers:
                            for layer_hidden in layer_hiddens:
                                for roi in rois:
                                    assert pooling_mode in ['max', 'avg']
                                    queue = next(queues_buffer)

                                    p_text = '-'.join([str(i) for i in p])
                                    pooling_text = f'spp_{p_text}'
                                    lstm_text = f'lstm_{num_lstm_layer}'
                                    video_text = f'{video_size}_{num_frame}'

                                    tags = [roi, video_text, pooling_text, pooling_mode, lstm_text,
                                            str(layer_hidden)]
                                    cloned_task = Task.clone(source_task=template_task,
                                                             name=','.join(tags),
                                                             parent=template_task.id)

                                    cloned_task.add_tags(tags)

                                    cloned_task_parameters = cloned_task.get_parameters()
                                    # cloned_task_parameters['rois'] = [roi]
                                    cloned_task_parameters['Args/rois'] = roi
                                    cloned_task_parameters['Args/track'] = 'full_track'
                                    cloned_task_parameters['Args/video_size'] = video_size
                                    cloned_task_parameters['Args/crop_size'] = 0
                                    cloned_task_parameters['Args/video_frames'] = num_frame
                                    cloned_task_parameters['Args/backbone_type'] = 'bdcn_edge'
                                    cloned_task_parameters['Args/preprocessing_type'] = 'bdcn'
                                    cloned_task_parameters['Args/load_from_np'] = False
                                    # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
                                    cloned_task_parameters['Args/learning_rate'] = 3e-4
                                    cloned_task_parameters['Args/batch_size'] = batch_size if not freeze_bn else 4
                                    cloned_task_parameters[
                                        'Args/accumulate_grad_batches'] = 1 if not freeze_bn else int(
                                        batch_size / 4)
                                    cloned_task_parameters['Args/num_layers'] = 2
                                    cloned_task_parameters['Args/lstm_layers'] = num_lstm_layer
                                    cloned_task_parameters['Args/layer_hidden'] = layer_hidden
                                    cloned_task_parameters['Args/debug'] = False
                                    cloned_task_parameters['Args/fp16'] = False
                                    cloned_task_parameters['Args/old_mix'] = True
                                    cloned_task_parameters['Args/no_convtrans'] = False
                                    cloned_task_parameters['Args/freeze_bn'] = freeze_bn
                                    cloned_task_parameters['Args/early_stop_epochs'] = 10
                                    cloned_task_parameters['Args/backbone_lr_ratio'] = 0.25
                                    cloned_task_parameters['Args/backbone_freeze_epochs'] = 12
                                    cloned_task_parameters['Args/max_epochs'] = 100
                                    cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                                    cloned_task_parameters['Args/pooling_mode'] = 'max'
                                    cloned_task_parameters['Args/spp'] = True
                                    cloned_task_parameters['Args/spp_size'] = p
                                    cloned_task_parameters['Args/val_check_interval'] = 1.0
                                    cloned_task_parameters['Args/val_ratio'] = 0.1
                                    cloned_task_parameters['Args/save_checkpoints'] = True
                                    cloned_task_parameters['Args/checkpoints_dir'] = '/data/huze/checkpoints/'
                                    cloned_task_parameters[
                                        'Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'

                                    cloned_task.set_parameters(cloned_task_parameters)
                                    print('Experiment set with parameters {}'.format(cloned_task_parameters))

                                    # enqueue the task for execution
                                    Task.enqueue(cloned_task.id, queue_name=queue)
                                    print('Experiment id={} enqueue for execution'.format(cloned_task.id))

                                    task_ids.append(cloned_task.id)
#
# # 675
start_tasks_spp(
    rois=['WB'],
    video_sizes=[32, 48, 64, 96, 128],
    num_frames=[4, 10],
    ps=[
        [3, 6, 9],
        [6, 12, 18],
        [8, 16, 24],
        [12, 20, 32],
    ],
    freeze_bns=[False],
    pooling_modes=['max'],
    num_lstm_layers=[1],
    layer_hiddens=[2048],
    batch_size=32,
)

print(task_ids)
