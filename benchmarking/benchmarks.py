from dataclasses import dataclass
from typing import Optional, List, Dict

#from syne_tune.blackbox_repository.conversion_scripts.scripts.tabrepo_import import (
#    TABREPO_DATASETS,
#)


@dataclass
class BenchmarkDefinition:
    max_wallclock_time: float
    n_workers: int
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name: str
    max_num_evaluations: Optional[int] = None
    surrogate: Optional[str] = None
    use_surrogate: Optional[bool] = True,
    surrogate_kwargs: Optional[Dict] = None
    datasets: Optional[List[str]] = None


n_full_evals = 200


def fcnet_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=7200,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_loss",
        mode="min",
        blackbox_name="fcnet",
        dataset_name=dataset_name,
        use_surrogate=True,
        # allow to stop after having seen the equivalent of `n_full_evals` evaluations
        max_num_evaluations=n_full_evals,
    )


def nas201_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=72000 if dataset_name == "ImageNet16-120" else 36000,
        max_num_evaluations=n_full_evals,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        use_surrogate=True,
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
    )


def lcbench_benchmark(dataset_name, datasets):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=n_full_evals,
        n_workers=4,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        use_surrogate=True,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
        datasets=datasets,
    )



benchmark_definitions = {
    "fcnet-protein": fcnet_benchmark("protein_structure"),
    "fcnet-naval": fcnet_benchmark("naval_propulsion"),
    "fcnet-parkinsons": fcnet_benchmark("parkinsons_telemonitoring"),
    "fcnet-slice": fcnet_benchmark("slice_localization"),
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
}


# 5 most expensive lcbench datasets
lc_bench_datasets = [
    "Fashion-MNIST",
    "airlines",
    "albert",
    "covertype",
    "christine",
]
for task in lc_bench_datasets:
    benchmark_definitions[
        "lcbench-" + task.replace("_", "-").replace(".", "")
    ] = lcbench_benchmark(task, datasets=lc_bench_datasets)

n_full_evals = 100


def pd1_benchmark(dataset_name: str):
    return BenchmarkDefinition(
        max_wallclock_time=3600000000,
        max_num_evaluations=n_full_evals,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error_rate",
        mode="min",
        blackbox_name='pd1',
        dataset_name=dataset_name,
        use_surrogate=True,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )


tasks = ['imagenet_resnet_batch_size_512',
         'uniref50_transformer_batch_size_128',
         'translate_wmt_xformer_translate_batch_size_64',
         'lm1b_transformer_batch_size_2048',
         'imagenet_resnet_batch_size_256',
         'mnist_max_pooling_cnn_tanh_batch_size_2048',
         'mnist_max_pooling_cnn_tanh_batch_size_256',
         'mnist_max_pooling_cnn_relu_batch_size_2048',
         'mnist_max_pooling_cnn_relu_batch_size_256',
         'mnist_simple_cnn_batch_size_2048',
         'mnist_simple_cnn_batch_size_256',
         'fashion_mnist_max_pooling_cnn_tanh_batch_size_2048',
         'fashion_mnist_max_pooling_cnn_tanh_batch_size_256',
         'fashion_mnist_max_pooling_cnn_relu_batch_size_2048',
         'fashion_mnist_max_pooling_cnn_relu_batch_size_256',
         'fashion_mnist_simple_cnn_batch_size_2048',
         'fashion_mnist_simple_cnn_batch_size_256',
         'svhn_no_extra_wide_resnet_batch_size_1024',
         'svhn_no_extra_wide_resnet_batch_size_256',
         'cifar100_wide_resnet_batch_size_2048',
         'cifar100_wide_resnet_batch_size_256',
         'cifar10_wide_resnet_batch_size_2048',
         'cifar10_wide_resnet_batch_size_256']


for ds in tasks:
    benchmark_definitions["pd1_" + ds] = pd1_benchmark(ds)


if __name__ == "__main__":
    from syne_tune.blackbox_repository import load_blackbox

    for benchmark, benchmark_case in benchmark_definitions.items():
        print(benchmark)
        bb = load_blackbox(benchmark_case.blackbox_name)
