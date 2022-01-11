# LightGBM Inferencing Benchmark

!!! note
    The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).

!!! warning
    This is work in progress, to check out current work items check the [project open inferencing issues](https://github.com/microsoft/lightgbm-benchmark/issues?q=is%3Aissue+is%3Aopen+label%3Ainferencing-benchmark). In particular, **do not trust these numbers yet** until we've removed this header!

## Variants

{{variants_table}}

## Metric time_inferencing per prediction (usecs)

{{metrics_table}}

## Percentile metrics for each variant

Some variants above report percentile metrics. Those are reported by computing inferencing latency per request batch (currently, batch size = 1, and number of threads = 1). Not all variants provide those (work in progress).

{% for entry in percentile_metrics_reports %}
### {{entry.variant_id}}

{{entry.report}}
{% endfor %}