import argparse
import itertools
import logging
import os

import pandas as pd
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch,
    UniformIntegerParameterRange)

from clearml.automation.optuna import OptimizerOptuna

aSearchStrategy = OptimizerOptuna

PROJECT_NAME = 'ROI LAYER search RF'
BASE_TASK = 'task template'
TASK_NAME = 'HPO'

parser = argparse.ArgumentParser(description="ROI")
parser.add_argument('--roi', type=str, default='LOC')
parser.add_argument('--layer', type=str, default='x1')
parser.add_argument('--local_port', type=int, default=50091)
args = parser.parse_args()

ROIS = ['V1', 'V2', 'V3', 'V4', 'EBA', 'LOC', 'PPA', 'FFA', 'STS']
LAYERS = ['x1', 'x2', 'x3', 'x4']
COMB = list(itertools.product(ROIS, LAYERS))

available_devices = {
    '16': [0, 1],
}

queue_names = []
for k, vs in available_devices.items():
    for v in vs:
        queue_names.append(f'{k}-{v}')
queues_buffer = itertools.cycle(queue_names)
queue_length = len(queue_names)
queue_status = {q: 'empty' for q in queue_names}

queue_dict = {}
for i, c in enumerate(COMB):
    queue_dict[c] = next(queues_buffer)

execution_queue = queue_dict[(args.roi, args.layer)]
DEVICE = execution_queue.split('-')[-1]

task = Task.init(project_name=PROJECT_NAME,
                 task_name=f"{TASK_NAME},{args.roi},{args.layer}",
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)
task.connect(args)
template_taskid = Task.get_task(project_name=PROJECT_NAME, task_name=BASE_TASK).id

RES_DF = pd.DataFrame()
dir_path = 'hpo_rf'
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
CSV_PATH = f'{dir_path}/{args.roi}_{args.layer}.csv'


def job_complete_callback(
        job_id,  # type: str
        objective_value,  # type: float
        objective_iteration,  # type: int
        job_parameters,  # type: dict
        top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    res = {
        'roi': args.roi,
        'layer': args.layer,
        'pool_rf': job_parameters[f'Args/pooling_size_{args.layer}'],
        'objective_value': objective_value,
    }
    global RES_DF
    RES_DF = RES_DF.append(res, ignore_index=True)
    RES_DF.to_csv(CSV_PATH)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


pool_size_max = 14 if args.layer != 'x4' else 9
total_max_jobs = 12 if args.layer != 'x4' else 8
pool_mode = 'adaptive_max' if args.layer != 'x4' else 'adaptive_avg'

a_optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id=template_taskid,
    # here we define the hyper-parameters to optimize
    # Notice: The parameter name should exactly match what you see in the UI: <section_name>/<parameter>
    # For Example, here we see in the base experiment a section Named: "General"
    # under it a parameter named "batch_size", this becomes "General/batch_size"
    # If you have `argparse` for example, then arguments will appear under the "Args" section,
    # and you should instead pass "Args/batch_size"
    hyper_parameters=[
        UniformIntegerParameterRange(f'Args/pooling_size_{args.layer}', min_value=1, max_value=pool_size_max,
                                     step_size=1),
        DiscreteParameterRange(f'Args/{args.layer}_pooling_mode', values=[pool_mode]),
        DiscreteParameterRange('Args/gpus', values=[DEVICE]),
        DiscreteParameterRange('Args/rois', values=[args.roi]),
        DiscreteParameterRange('Args/pyramid_layers', values=[args.layer]),
    ],
    # this is the objective metric we want to maximize/minimize
    objective_metric_title='validation',
    objective_metric_series='correlation',
    # now we decide if we want to maximize it or minimize it (accuracy we maximize)
    objective_metric_sign='max_global',
    # let us limit the number of concurrent experiments,
    # this in turn will make sure we do dont bombard the scheduler with experiments.
    # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
    max_number_of_concurrent_tasks=1,
    # this is the optimizer class (actually doing the optimization)
    # Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
    # more are coming soon...
    optimizer_class=aSearchStrategy,
    # Select an execution queue to schedule the experiments for execution
    execution_queue=execution_queue,
    # If specified all Tasks created by the HPO process will be created under the `spawned_project` project
    spawn_project=None,  # 'HPO spawn project',
    # If specified only the top K performing Tasks will be kept, the others will be automatically archived
    # save_top_k_tasks_only=5,  # 5,
    # Optional: Limit the execution time of a single experiment, in minutes.
    # (this is optional, and if using  OptimizerBOHB, it is ignored)
    # time_limit_per_job=10.,
    # Check the experiments every 12 seconds is way too often, we should probably set it to 5 min,
    # assuming a single experiment is usually hours...
    pool_period_min=1,
    # set the maximum number of jobs to launch for the optimization, default (None) unlimited
    # If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
    # basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
    total_max_jobs=total_max_jobs,
    # set the minimum number of iterations for an experiment, before early stopping.
    # Does not apply for simple strategies such as RandomSearch or GridSearch
    # min_iteration_per_job=10,
    # Set the maximum number of iterations for an experiment to execute
    # (This is optional, unless using OptimizerBOHB where this is a must)
    # max_iteration_per_job=100,
    # optimizer_kwargs={'local_port': LOCAL_PORT},
    local_port=args.local_port,
)

a_optimizer.set_report_period(1)
a_optimizer.start(job_complete_callback=job_complete_callback)
a_optimizer.set_time_limit(in_minutes=48 * 60)
a_optimizer.wait()
a_optimizer.stop()
