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


def start_tasks_spp(rois, layers, ps, pts, freeze_bns, pooling_modes, pathways, batch_size=32, conv_size=256):
    for roi in rois:
        for layer in layers:
            for p in ps:
                for pt in pts:
                    for freeze_bn in freeze_bns:
                        for pooling_mode in pooling_modes:
                            for pathway in pathways:
                                assert pooling_mode in ['max', 'avg']
                                queue = next(queues_buffer)

                                p_text = '-'.join([str(i) for i in p]) + '_' + '-'.join([str(i) for i in pt])
                                freeze_text = 'f_bn' if freeze_bn else 'nof_bn'
                                pooling_text = f'spp_{p_text}_{pooling_mode}'

                                tags = [roi, layer, pooling_text, freeze_text, pathway]
                                cloned_task = Task.clone(source_task=template_task,
                                                         name=','.join(tags),
                                                         parent=template_task.id)

                                cloned_task.add_tags(tags)

                                cloned_task_parameters = cloned_task.get_parameters()
                                # cloned_task_parameters['rois'] = [roi]
                                cloned_task_parameters['Args/rois'] = roi
                                cloned_task_parameters['Args/track'] = 'full_track'
                                cloned_task_parameters['Args/video_size'] = 256
                                cloned_task_parameters['Args/crop_size'] = 224
                                cloned_task_parameters['Args/video_frames'] = 16
                                cloned_task_parameters['Args/backbone_type'] = 'i3d_flow'
                                cloned_task_parameters['Args/preprocessing_type'] = 'i3d_flow'
                                cloned_task_parameters['Args/load_from_np'] = True
                                # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
                                cloned_task_parameters['Args/learning_rate'] = 1e-4
                                cloned_task_parameters['Args/step_lr_epochs'] = [10]
                                cloned_task_parameters['Args/step_lr_ratio'] = 1.0
                                cloned_task_parameters['Args/batch_size'] = batch_size if not freeze_bn else 4
                                cloned_task_parameters['Args/accumulate_grad_batches'] = 1 if not freeze_bn else int(
                                    batch_size / 4)
                                cloned_task_parameters['Args/num_layers'] = 2
                                cloned_task_parameters['Args/conv_size'] = conv_size
                                cloned_task_parameters['Args/first_layer_hidden'] = 2048
                                cloned_task_parameters['Args/layer_hidden'] = 2048
                                cloned_task_parameters['Args/debug'] = False
                                cloned_task_parameters['Args/fp16'] = False
                                cloned_task_parameters['Args/old_mix'] = True
                                cloned_task_parameters['Args/no_convtrans'] = False
                                cloned_task_parameters['Args/freeze_bn'] = freeze_bn
                                cloned_task_parameters['Args/old_mix'] = True
                                cloned_task_parameters['Args/early_stop_epochs'] = 10
                                cloned_task_parameters['Args/backbone_lr_ratio'] = 1.0
                                cloned_task_parameters['Args/backbone_freeze_epochs'] = 0
                                cloned_task_parameters['Args/max_epochs'] = 100
                                cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                                cloned_task_parameters['Args/pooling_mode'] = pooling_mode
                                for l in layer.split(','):
                                    cloned_task_parameters[f'Args/{l}_pooling_mode'] = 'spp'
                                    cloned_task_parameters[f'Args/spp_size_{l}'] = p
                                    cloned_task_parameters[f'Args/spp_size_t_{l}'] = pt
                                cloned_task_parameters['Args/final_fusion'] = 'concat'
                                cloned_task_parameters['Args/pyramid_layers'] = layer
                                cloned_task_parameters['Args/pathways'] = pathway
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


def start_tasks_x5(rois, layers, freeze_bns, batch_size=32, conv_size=1024):
    for roi in rois:
        for layer in layers:
            for freeze_bn in freeze_bns:
                assert layer == 'x5'

                queue = next(queues_buffer)

                freeze_text = 'f_bn' if freeze_bn else 'nof_bn'

                tags = [roi, layer, freeze_text]
                cloned_task = Task.clone(source_task=template_task,
                                         name=','.join(tags),
                                         parent=template_task.id)

                cloned_task.add_tags(tags)

                cloned_task_parameters = cloned_task.get_parameters()
                # cloned_task_parameters['rois'] = [roi]
                cloned_task_parameters['Args/rois'] = roi
                cloned_task_parameters['Args/track'] = 'full_track'
                cloned_task_parameters['Args/video_size'] = 256
                cloned_task_parameters['Args/crop_size'] = 224
                cloned_task_parameters['Args/video_frames'] = 16
                cloned_task_parameters['Args/backbone_type'] = 'i3d_flow'
                cloned_task_parameters['Args/preprocessing_type'] = 'i3d_flow'
                cloned_task_parameters['Args/load_from_np'] = True
                # cloned_task_parameters['Args/batch_size'] = 32 if pooling_sch in ['avg', 'max'] else 24
                cloned_task_parameters['Args/learning_rate'] = 1e-4
                cloned_task_parameters['Args/step_lr_epochs'] = [10]
                cloned_task_parameters['Args/step_lr_ratio'] = 1.0
                cloned_task_parameters['Args/batch_size'] = batch_size if not freeze_bn else 4
                cloned_task_parameters['Args/accumulate_grad_batches'] = 1 if not freeze_bn else int(
                    batch_size / 4)
                cloned_task_parameters['Args/num_layers'] = 2
                cloned_task_parameters['Args/conv_size'] = conv_size
                cloned_task_parameters['Args/first_layer_hidden'] = 2048
                cloned_task_parameters['Args/layer_hidden'] = 2048
                cloned_task_parameters['Args/debug'] = False
                cloned_task_parameters['Args/fp16'] = False
                cloned_task_parameters['Args/old_mix'] = True
                cloned_task_parameters['Args/no_convtrans'] = False
                cloned_task_parameters['Args/freeze_bn'] = freeze_bn
                cloned_task_parameters['Args/old_mix'] = True
                cloned_task_parameters['Args/early_stop_epochs'] = 10
                cloned_task_parameters['Args/backbone_lr_ratio'] = 1.0
                cloned_task_parameters['Args/backbone_freeze_epochs'] = 0
                cloned_task_parameters['Args/max_epochs'] = 100
                cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                cloned_task_parameters['Args/final_fusion'] = 'concat'
                cloned_task_parameters['Args/pyramid_layers'] = layer
                cloned_task_parameters['Args/pathways'] = 'none'
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


# # 9
start_tasks_x5(
    rois=['WB'],
    layers=['x5'],
    freeze_bns=[True],
    batch_size=32,
    conv_size=1024,
)
# # 45
start_tasks_spp(
    rois=['WB'],
    layers=['x1,x2,x3,x4', 'x2,x3,x4'],
    ps=[
        [1, 2, 3],
        [1, 3, 5],
        [1, 4, 7],
        [2, 4, 8],
        [3, 5, 7],
    ],
    pts=[
        [1, 1, 1],
    ],
    freeze_bns=[True],
    pooling_modes=['avg'],
    pathways=['none', 'topdown'],
    batch_size=24,
    conv_size=256,
)

print(task_ids)
