import itertools

from clearml import Task

PROJECT_NAME = 'Algonauts exensemble full track'
BASE_TASK = 'task template'

task = Task.init(project_name=PROJECT_NAME,
                 task_name='Task Manager',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

template_task = Task.get_task(project_name=PROJECT_NAME,
                              task_name=BASE_TASK)

available_devices = {
    '16': [0],
}

queue_names = []
for k, vs in available_devices.items():
    for v in vs:
        queue_names.append(f'{k}-{v}')
queues_buffer = itertools.cycle(queue_names)

task_ids = []


def start_tasks_spp_bit(rois, video_sizes, num_frames, ps, freeze_bns, pooling_modes, num_lstm_layers,
                    layer_hiddens, layers,
                    batch_size=32):
    for video_size in video_sizes:
        for num_frame in num_frames:
            for p in ps:
                for freeze_bn in freeze_bns:
                    for pooling_mode in pooling_modes:
                        for num_lstm_layer in num_lstm_layers:
                            for layer_hidden in layer_hiddens:
                                for layer in layers:
                                    for roi in rois:
                                        assert pooling_mode in ['max', 'avg']
                                        queue = next(queues_buffer)

                                        p_text = '-'.join([str(i) for i in p])
                                        freeze_text = 'f_bn' if freeze_bn else 'nof_bn'
                                        lstm_text = f'lstm_{num_lstm_layer}'
                                        video_text = f'{video_size}_{num_frame}'

                                        tags = [roi, video_text, p_text, pooling_mode, lstm_text, freeze_text,
                                                str(layer_hidden), layer]
                                        cloned_task = Task.clone(source_task=template_task,
                                                                 name=','.join(tags),
                                                                 parent=template_task.id)

                                        cloned_task.add_tags(tags)

                                        cloned_task_parameters = cloned_task.get_parameters()
                                        # cloned_task_parameters['rois'] = [roi]
                                        cloned_task_parameters['Args/rois'] = roi
                                        cloned_task_parameters['Args/track'] = 'full_track'
                                        cloned_task_parameters['Args/video_size'] = video_size
                                        cloned_task_parameters['Args/video_frames'] = num_frame
                                        # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
                                        cloned_task_parameters['Args/learning_rate'] = 1e-4
                                        cloned_task_parameters['Args/batch_size'] = batch_size if not freeze_bn else 8
                                        cloned_task_parameters[
                                            'Args/accumulate_grad_batches'] = 1 if not freeze_bn else int(
                                            batch_size / 8)
                                        cloned_task_parameters['Args/num_layers'] = 2
                                        cloned_task_parameters['Args/lstm_layers'] = num_lstm_layer
                                        cloned_task_parameters['Args/layer_hidden'] = layer_hidden
                                        cloned_task_parameters['Args/conv_size'] = 256
                                        cloned_task_parameters['Args/debug'] = False
                                        cloned_task_parameters['Args/fp16'] = True
                                        cloned_task_parameters['Args/freeze_bn'] = freeze_bn
                                        cloned_task_parameters['Args/early_stop_epochs'] = 10
                                        cloned_task_parameters['Args/backbone_lr_ratio'] = 0.1
                                        cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
                                        cloned_task_parameters['Args/max_epochs'] = 100
                                        cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                                        cloned_task_parameters['Args/spp'] = True
                                        cloned_task_parameters['Args/pooling_mode'] = pooling_mode
                                        cloned_task_parameters['Args/spp_size'] = p
                                        cloned_task_parameters['Args/pyramid_layers'] = layer
                                        cloned_task_parameters['Args/pathways'] = 'none'
                                        cloned_task_parameters['Args/old_mix'] = True
                                        cloned_task_parameters['Args/no_convtrans'] = True
                                        cloned_task_parameters['Args/backbone_type'] = 'bit'
                                        cloned_task_parameters['Args/preprocessing_type'] = 'bit'
                                        cloned_task_parameters['Args/load_from_np'] = False
                                        cloned_task_parameters['Args/val_check_interval'] = 1.0
                                        cloned_task_parameters['Args/val_ratio'] = 0.1
                                        cloned_task_parameters['Args/save_checkpoints'] = True
                                        cloned_task_parameters['Args/rm_checkpoints'] = False
                                        cloned_task_parameters['Args/checkpoints_dir'] = '/mnt/huze/exensemble_ckpts/'
                                        cloned_task_parameters[
                                            'Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'

                                        cloned_task.set_parameters(cloned_task_parameters)
                                        print('Experiment set with parameters {}'.format(cloned_task_parameters))

                                        # enqueue the task for execution
                                        Task.enqueue(cloned_task.id, queue_name=queue)
                                        print('Experiment id={} enqueue for execution'.format(cloned_task.id))

                                        task_ids.append(cloned_task.id)



def start_tasks_spp_bdcn(rois, video_sizes, num_frames, ps, freeze_bns, pooling_modes, num_lstm_layers,
                    layer_hiddens, layers,
                    batch_size=32):
    for video_size in video_sizes:
        for num_frame in num_frames:
            for p in ps:
                for freeze_bn in freeze_bns:
                    for pooling_mode in pooling_modes:
                        for num_lstm_layer in num_lstm_layers:
                            for layer_hidden in layer_hiddens:
                                for layer in layers:
                                    for roi in rois:
                                        assert pooling_mode in ['max', 'avg']
                                        queue = next(queues_buffer)

                                        p_text = '-'.join([str(i) for i in p])
                                        freeze_text = 'f_bn' if freeze_bn else 'nof_bn'
                                        lstm_text = f'lstm_{num_lstm_layer}'
                                        video_text = f'{video_size}_{num_frame}'

                                        tags = [roi, video_text, p_text, pooling_mode, lstm_text, freeze_text,
                                                str(layer_hidden), layer]
                                        cloned_task = Task.clone(source_task=template_task,
                                                                 name=','.join(tags),
                                                                 parent=template_task.id)

                                        cloned_task.add_tags(tags)

                                        cloned_task_parameters = cloned_task.get_parameters()
                                        # cloned_task_parameters['rois'] = [roi]
                                        cloned_task_parameters['Args/rois'] = roi
                                        cloned_task_parameters['Args/track'] = 'full_track'
                                        cloned_task_parameters['Args/video_size'] = video_size
                                        cloned_task_parameters['Args/video_frames'] = num_frame
                                        # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
                                        cloned_task_parameters['Args/learning_rate'] = 1e-4
                                        cloned_task_parameters['Args/batch_size'] = batch_size if not freeze_bn else 8
                                        cloned_task_parameters[
                                            'Args/accumulate_grad_batches'] = 1 if not freeze_bn else int(
                                            batch_size / 8)
                                        cloned_task_parameters['Args/num_layers'] = 2
                                        cloned_task_parameters['Args/lstm_layers'] = num_lstm_layer
                                        cloned_task_parameters['Args/layer_hidden'] = layer_hidden
                                        cloned_task_parameters['Args/conv_size'] = 256
                                        cloned_task_parameters['Args/debug'] = False
                                        cloned_task_parameters['Args/fp16'] = True
                                        cloned_task_parameters['Args/freeze_bn'] = freeze_bn
                                        cloned_task_parameters['Args/early_stop_epochs'] = 10
                                        cloned_task_parameters['Args/backbone_lr_ratio'] = 0.1
                                        cloned_task_parameters['Args/backbone_freeze_epochs'] = 4
                                        cloned_task_parameters['Args/max_epochs'] = 100
                                        cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                                        cloned_task_parameters['Args/spp'] = True
                                        cloned_task_parameters['Args/pooling_mode'] = pooling_mode
                                        cloned_task_parameters['Args/spp_size'] = p
                                        cloned_task_parameters['Args/pyramid_layers'] = layer
                                        cloned_task_parameters['Args/pathways'] = 'none'
                                        cloned_task_parameters['Args/old_mix'] = True
                                        cloned_task_parameters['Args/no_convtrans'] = True
                                        cloned_task_parameters['Args/backbone_type'] = 'bdcn_edge'
                                        cloned_task_parameters['Args/preprocessing_type'] = 'bdcn'
                                        cloned_task_parameters['Args/load_from_np'] = False
                                        cloned_task_parameters['Args/val_check_interval'] = 1.0
                                        cloned_task_parameters['Args/val_ratio'] = 0.1
                                        cloned_task_parameters['Args/save_checkpoints'] = True
                                        cloned_task_parameters['Args/rm_checkpoints'] = False
                                        cloned_task_parameters['Args/checkpoints_dir'] = '/mnt/huze/exensemble_ckpts/'
                                        cloned_task_parameters[
                                            'Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'

                                        cloned_task.set_parameters(cloned_task_parameters)
                                        print('Experiment set with parameters {}'.format(cloned_task_parameters))

                                        # enqueue the task for execution
                                        Task.enqueue(cloned_task.id, queue_name=queue)
                                        print('Experiment id={} enqueue for execution'.format(cloned_task.id))

                                        task_ids.append(cloned_task.id)

start_tasks_spp_bit(
    rois=['WB'],
    video_sizes=[224],
    num_frames=[4],
    ps=[
        [1],
        [3],
        [5],
        [7],
        [11],
        [1, 3, 5],
        [4, 6, 9],
    ],
    freeze_bns=[True],
    pooling_modes=['avg'],
    num_lstm_layers=[1],
    layer_hiddens=[2048],
    layers=['x1', 'x2', 'x3', 'x4'],
    batch_size=32,
)

start_tasks_spp_bdcn(
    rois=['WB'],
    video_sizes=[64, 128, 224],
    num_frames=[4],
    ps=[
        [3, 6, 9],
        [6, 12, 18],
        [8, 16, 24],
        [12, 20, 32],
        [16, 24, 32],
    ],
    freeze_bns=[True],
    pooling_modes=['avg'],
    num_lstm_layers=[1],
    layer_hiddens=[2048],
    layers=['iya'],
    batch_size=32,
)

print(task_ids)
