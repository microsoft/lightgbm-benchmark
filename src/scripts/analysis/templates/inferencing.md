# LightGBM Inferencing Benchmark

> The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).

IMPORTANT: This is work in progress, to check out current work items check the [project open issues](https://github.com/microsoft/lightgbm-benchmark/issues). In particular, **do not trust these numbers yet** until we've removed this header!

## Variants

| Index | Framework | Version | Build |
| :-- | :-- | :-- | :-- |
{% for variant in variants -%}
| {{variant[0]}}_{{loop.index}} | {{variant[0]}} | {{variant[1]}} | {{variant[2]}} |
{% endfor %}

## Metric time_inferencing

| &nbsp; |{% for variant in variants %} {{variant[0]}}_{{loop.index}} |{% endfor %}
| :-- |{% for variant in variants %} :-- |{% endfor %}
{% for task in tasks -%}
    {%- set outer_loop_index = loop.index0 -%}
    | trees={{task[0]}}<br/>leaves={{task[1]}}<br/>cols={{task[2]}}<br/> |
        {%- for variant in variants %} {{ "{:,.4f}".format(metrics[outer_loop_index][loop.index0]) }} |{% endfor %}
{% endfor %}
