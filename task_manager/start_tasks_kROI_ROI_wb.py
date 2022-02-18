import itertools
import os.path
import torch

from clearml import Task

PROJECT_NAME = 'kROI explore'
BASE_TASK = 'task template'

task = Task.init(project_name=PROJECT_NAME,
                 task_name='Task Manager',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

template_task = Task.get_task(project_name=PROJECT_NAME,
                              task_name=BASE_TASK)

score_dict = {'MC2': 0.12,
 'SMC2': 0.006,
 'EBA': 0.18,
 'STS': 0.105,
 'V3': 0.15,
 'REST': 0.06,
 'V4': 0.12,
 'V2': 0.15,
 'V1': 0.15,
 'PPA': 0.105,
 'SC4': 0.105,
 'LC3': 0.105,
 'LC5': 0.18,
 'SMC1': 0.036,
 'LC4': 0.21,
 'LOC': 0.15,
 'FFA': 0.135,
 'LC1': 0.027,
 'SC3': 0.135,
 'WB': 0.06,
 'MC1': 0.006,
 'LC2': 0.15}

available_devices = {
    '16': [0, 1]
}

queue_names = []
for k, vs in available_devices.items():
    for v in vs:
        queue_names.append(f'{k}-{v}')
queues_buffer = itertools.cycle(queue_names)

task_ids = []


def start_tasks_spp(krois, layers, ps, freeze_bns, pooling_modes,
                    pathways, pretrainings,
                    batch_size=32):
    for p in ps:
        for layer in layers:
            for freeze_bn in freeze_bns:
                for pooling_mode in pooling_modes:
                    for pathway in pathways:
                        for pretraining in pretrainings:
                            for kroi in krois:
                                assert pooling_mode in ['max', 'avg']
                                queue = next(queues_buffer)



                                p_text = '-'.join([str(i) for i in p])
                                pooling_text = f'{p_text}'

                                tags = [kroi, layer, pooling_text]
                                cloned_task = Task.clone(source_task=template_task,
                                                         name=','.join(tags),
                                                         parent=template_task.id)

                                cloned_task.add_tags(tags)

                                cloned_task_parameters = cloned_task.get_parameters()
                                # cloned_task_parameters['rois'] = [roi]
                                cloned_task_parameters['Args/rois'] = 'WB'
                                cloned_task_parameters['Args/kroi'] = kroi
                                cloned_task_parameters['Args/track'] = 'full_track'
                                cloned_task_parameters['Args/video_size'] = 288
                                cloned_task_parameters['Args/crop_size'] = 0
                                cloned_task_parameters['Args/video_frames'] = 16
                                cloned_task_parameters['Args/backbone_type'] = 'i3d_rgb'
                                cloned_task_parameters['Args/preprocessing_type'] = 'mmit'
                                cloned_task_parameters['Args/load_from_np'] = False
                                cloned_task_parameters['Args/learning_rate'] = 3e-4  # 1e-4
                                cloned_task_parameters['Args/step_lr_epochs'] = [10]
                                cloned_task_parameters['Args/step_lr_ratio'] = 1.0
                                cloned_task_parameters['Args/batch_size'] = batch_size if not freeze_bn else 8
                                cloned_task_parameters[
                                    'Args/accumulate_grad_batches'] = 1 if not freeze_bn else int(
                                    batch_size / 8)
                                cloned_task_parameters['Args/num_layers'] = 2
                                cloned_task_parameters['Args/conv_size'] = 256
                                cloned_task_parameters['Args/first_layer_hidden'] = 2048
                                cloned_task_parameters['Args/layer_hidden'] = 2048
                                cloned_task_parameters['Args/debug'] = False
                                cloned_task_parameters['Args/fp16'] = True
                                cloned_task_parameters['Args/pretrained'] = pretraining
                                cloned_task_parameters['Args/freeze_bn'] = freeze_bn
                                cloned_task_parameters['Args/old_mix'] = True
                                cloned_task_parameters['Args/no_convtrans'] = True
                                cloned_task_parameters['Args/early_stop_epochs'] = 6
                                cloned_task_parameters['Args/backbone_lr_ratio'] = 0.1
                                cloned_task_parameters['Args/backbone_freeze_score'] = score_dict[kroi]
                                cloned_task_parameters['Args/max_epochs'] = 100
                                cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                                cloned_task_parameters['Args/pooling_mode'] = pooling_mode
                                for l in layer.split(','):
                                    cloned_task_parameters[f'Args/{l}_pooling_mode'] = 'spp'
                                    cloned_task_parameters[f'Args/spp_size_{l}'] = p
                                    cloned_task_parameters[f'Args/spp_size_t_{l}'] = [1 for _ in p]
                                cloned_task_parameters[f'Args/pooling_size'] = p # quick save for 1 layer model
                                cloned_task_parameters['Args/final_fusion'] = 'concat'
                                cloned_task_parameters['Args/pyramid_layers'] = layer
                                cloned_task_parameters['Args/pathways'] = pathway
                                cloned_task_parameters['Args/val_check_interval'] = 1.0
                                cloned_task_parameters['Args/val_ratio'] = 0.1
                                cloned_task_parameters['Args/save_checkpoints'] = True
                                cloned_task_parameters['Args/rm_checkpoints'] = False
                                cloned_task_parameters['Args/checkpoints_dir'] = '/mnt/huze/ckpts_mkii/'
                                cloned_task_parameters[
                                    'Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'

                                cloned_task.set_parameters(cloned_task_parameters)
                                print('Experiment set with parameters {}'.format(cloned_task_parameters))

                                # enqueue the task for execution
                                Task.enqueue(cloned_task.id, queue_name=queue)
                                print('Experiment id={} enqueue for execution'.format(cloned_task.id))

                                task_ids.append(cloned_task.id)


def start_tasks_spp_flow(krois, layers, ps, freeze_bns, pooling_modes,
                    pathways, pretrainings,
                    batch_size=32):
    for p in ps:
        for layer in layers:
            for freeze_bn in freeze_bns:
                for pooling_mode in pooling_modes:
                    for pathway in pathways:
                        for pretraining in pretrainings:
                            for kroi in krois:
                                assert pooling_mode in ['max', 'avg']
                                queue = next(queues_buffer)



                                p_text = '-'.join([str(i) for i in p])
                                pooling_text = f'{p_text}'

                                tags = [kroi, layer, pooling_text]
                                cloned_task = Task.clone(source_task=template_task,
                                                         name=','.join(tags),
                                                         parent=template_task.id)

                                cloned_task.add_tags(tags)

                                cloned_task_parameters = cloned_task.get_parameters()
                                # cloned_task_parameters['rois'] = [roi]
                                cloned_task_parameters['Args/rois'] = 'WB'
                                cloned_task_parameters['Args/kroi'] = kroi
                                cloned_task_parameters['Args/track'] = 'full_track'
                                cloned_task_parameters['Args/video_size'] = 256
                                cloned_task_parameters['Args/crop_size'] = 224
                                cloned_task_parameters['Args/video_frames'] = 64
                                cloned_task_parameters['Args/backbone_type'] = 'i3d_flow'
                                cloned_task_parameters['Args/preprocessing_type'] = 'i3d_flow'
                                cloned_task_parameters['Args/load_from_np'] = False
                                cloned_task_parameters['Args/learning_rate'] = 3e-4  # 1e-4
                                cloned_task_parameters['Args/step_lr_epochs'] = [10]
                                cloned_task_parameters['Args/step_lr_ratio'] = 1.0
                                cloned_task_parameters['Args/batch_size'] = batch_size if not freeze_bn else 4
                                cloned_task_parameters[
                                    'Args/accumulate_grad_batches'] = 1 if not freeze_bn else int(
                                    batch_size / 4)
                                cloned_task_parameters['Args/num_layers'] = 2
                                cloned_task_parameters['Args/conv_size'] = 256
                                cloned_task_parameters['Args/first_layer_hidden'] = 2048
                                cloned_task_parameters['Args/layer_hidden'] = 2048
                                cloned_task_parameters['Args/debug'] = False
                                cloned_task_parameters['Args/fp16'] = True
                                cloned_task_parameters['Args/pretrained'] = pretraining
                                cloned_task_parameters['Args/freeze_bn'] = freeze_bn
                                cloned_task_parameters['Args/old_mix'] = True
                                cloned_task_parameters['Args/no_convtrans'] = True
                                cloned_task_parameters['Args/early_stop_epochs'] = 6
                                cloned_task_parameters['Args/backbone_lr_ratio'] = 0.1
                                cloned_task_parameters['Args/backbone_freeze_score'] = score_dict[kroi] * 0.81
                                cloned_task_parameters['Args/max_epochs'] = 100
                                cloned_task_parameters['Args/gpus'] = queue.split('-')[1]
                                cloned_task_parameters['Args/pooling_mode'] = pooling_mode
                                for l in layer.split(','):
                                    cloned_task_parameters[f'Args/{l}_pooling_mode'] = 'spp'
                                    cloned_task_parameters[f'Args/spp_size_{l}'] = p
                                    cloned_task_parameters[f'Args/spp_size_t_{l}'] = [1 for _ in p]
                                cloned_task_parameters[f'Args/pooling_size'] = p # quick save for 1 layer model
                                cloned_task_parameters['Args/final_fusion'] = 'concat'
                                cloned_task_parameters['Args/pyramid_layers'] = layer
                                cloned_task_parameters['Args/pathways'] = pathway
                                cloned_task_parameters['Args/val_check_interval'] = 1.0
                                cloned_task_parameters['Args/val_ratio'] = 0.1
                                cloned_task_parameters['Args/save_checkpoints'] = True
                                cloned_task_parameters['Args/rm_checkpoints'] = False
                                cloned_task_parameters['Args/checkpoints_dir'] = '/mnt/huze/ckpts_mkii/'
                                cloned_task_parameters[
                                    'Args/predictions_dir'] = f'/data_smr/huze/projects/my_algonauts/predictions/'

                                cloned_task.set_parameters(cloned_task_parameters)
                                print('Experiment set with parameters {}'.format(cloned_task_parameters))

                                # enqueue the task for execution
                                Task.enqueue(cloned_task.id, queue_name=queue)
                                print('Experiment id={} enqueue for execution'.format(cloned_task.id))

                                task_ids.append(cloned_task.id)



# start_tasks_spp(
#     krois=score_dict.keys(),
#     layers=['x1', 'x2', 'x3', 'x4'],
#     ps=[
#         [1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6],
#         [7],
#         [8],
#         [9],
#     ],
#     freeze_bns=[True],
#     pooling_modes=['avg'],
#     pathways=['none'],
#     pretrainings=[True],
#     batch_size=32,
# )
#
# start_tasks_spp(
#     krois=score_dict.keys(),
#     layers=['x1', 'x2', 'x3', 'x4'],
#     ps=[
#         [1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6],
#         [7],
#     ],
#     freeze_bns=[True],
#     pooling_modes=['avg'],
#     pathways=['none'],
#     pretrainings=[True],
#     batch_size=32,
# )

start_tasks_spp(
    krois=['SM2'],
    layers=['x4'],
    ps=[
        [1],
    ],
    freeze_bns=[True],
    pooling_modes=['avg'],
    pathways=['none'],
    pretrainings=[True],
    batch_size=32,
)

start_tasks_spp(
    krois=['SM2'],
    layers=['x4'],
    ps=[
        [1],
    ],
    freeze_bns=[True],
    pooling_modes=['avg'],
    pathways=['none'],
    pretrainings=[True],
    batch_size=32,
)


print(task_ids)
