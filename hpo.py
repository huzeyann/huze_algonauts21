import itertools
import logging

from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch,
    UniformIntegerParameterRange)

from clearml.automation.optuna import OptimizerOptuna

aSearchStrategy = OptimizerOptuna

PROJECT_NAME = 'debug'
BASE_TASK = 'task template'
TASK_NAME = 'HPO'

ROIS = ['V1', 'V2', 'V3', 'V4', 'EBA', 'LOC', 'PPA', 'FFA', 'STS']
LAYERS = ['x1', 'x2', 'x3', 'x4']
COMB = list(itertools.product(ROIS, LAYERS))

available_devices = {
    '16': [0, 1],
}


def job_complete_callback(
        job_id,  # type: str
        objective_value,  # type: float
        objective_iteration,  # type: int
        job_parameters,  # type: dict
        top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name=PROJECT_NAME,
                 task_name=f"{TASK_NAME}",
                 # task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

template_taskid = Task.get_task(project_name=PROJECT_NAME, task_name=BASE_TASK).id

queue_names = []
for k, vs in available_devices.items():
    for v in vs:
        queue_names.append(f'{k}-{v}')
queues_buffer = itertools.cycle(queue_names)
queue_length = len(queue_names)
queue_status = {q: 'empty' for q in queue_names}

# for roi, layer in COMB[:2]:
#     optimizers = []



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
        UniformIntegerParameterRange('Args/pooling_size_x3', min_value=2, max_value=14, step_size=1),
        DiscreteParameterRange('General/gpus', values=['0']),
        DiscreteParameterRange('General/rois', values=['V1']),
        DiscreteParameterRange('General/pyramid_layers', values=['x3']),
        # DiscreteParameterRange('General/pooling_size_x3', values=[4]),
    ],
    # this is the objective metric we want to maximize/minimize
    objective_metric_title='final',
    objective_metric_series='val_corr',
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
    execution_queue='16-0',
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
    total_max_jobs=8,
    # set the minimum number of iterations for an experiment, before early stopping.
    # Does not apply for simple strategies such as RandomSearch or GridSearch
    min_iteration_per_job=10,
    # Set the maximum number of iterations for an experiment to execute
    # (This is optional, unless using OptimizerBOHB where this is a must)
    max_iteration_per_job=100,
    # optimizer_kwargs={'local_port': LOCAL_PORT},
    local_port=50901,
)

b_optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id=template_taskid,
    # here we define the hyper-parameters to optimize
    # Notice: The parameter name should exactly match what you see in the UI: <section_name>/<parameter>
    # For Example, here we see in the base experiment a section Named: "General"
    # under it a parameter named "batch_size", this becomes "General/batch_size"
    # If you have `argparse` for example, then arguments will appear under the "Args" section,
    # and you should instead pass "Args/batch_size"
    hyper_parameters=[
        UniformIntegerParameterRange('Args/pooling_size_x3', min_value=2, max_value=14, step_size=1),
        DiscreteParameterRange('General/gpus', values=['1']),
        DiscreteParameterRange('General/rois', values=['V2']),
        DiscreteParameterRange('General/pyramid_layers', values=['x3']),
        # DiscreteParameterRange('General/pooling_size_x3', values=[4]),
    ],
    # this is the objective metric we want to maximize/minimize
    objective_metric_title='final',
    objective_metric_series='val_corr',
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
    execution_queue='16-1',
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
    total_max_jobs=8,
    # set the minimum number of iterations for an experiment, before early stopping.
    # Does not apply for simple strategies such as RandomSearch or GridSearch
    min_iteration_per_job=10,
    # Set the maximum number of iterations for an experiment to execute
    # (This is optional, unless using OptimizerBOHB where this is a must)
    max_iteration_per_job=100,
    # optimizer_kwargs={'local_port': LOCAL_PORT},
    local_port=50902,
)

a_task = Task.create(project_name=PROJECT_NAME,
                 task_name=f"optimizer A",
                 task_type=Task.TaskTypes.optimizer,)
a_optimizer.set_optimizer_task(a_task)

b_task = Task.create(project_name=PROJECT_NAME,
                 task_name=f"optimizer B",
                 task_type=Task.TaskTypes.optimizer,)
b_optimizer.set_optimizer_task(b_task)

a_optimizer.set_report_period(0.2)
b_optimizer.set_report_period(0.2)

a_optimizer.start(job_complete_callback=job_complete_callback)
b_optimizer.start(job_complete_callback=job_complete_callback)

a_optimizer.set_time_limit(in_minutes=120.0)
b_optimizer.set_time_limit(in_minutes=120.0)

a_optimizer.wait()
a_optimizer.stop()

b_optimizer.wait()
b_optimizer.stop()


