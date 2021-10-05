# LightGBM Inferencing Benchmark

!!! note
    The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).

!!! warning
    This is work in progress, to check out current work items check the [project open issues](https://github.com/microsoft/lightgbm-benchmark/issues). In particular, **do not trust these numbers yet** until we've removed this header!

## Variants

| Index | Framework | Version | Build |
| :-- | :-- | :-- | :-- |
{% for variant in variants -%}
| {{variant[0]}}_{{loop.index-}}
| {{variant[0]-}}
| {{variant[1]-}}
| {% if variant[2].startswith("dockers/") -%}
    [{{variant[2]}}](https://github.com/microsoft/lightgbm-benchmark/tree/main/src/scripts/lightgbm_python/{{variant[2]}})
  {%- else -%}
    {{variant[2]}}
  {%- endif -%}
|
{% endfor %}

## Metric time_inferencing per prediction (usecs)

| &nbsp; |{% for variant in variants %} {{variant[0]}}_{{loop.index}} |{% endfor %}
| :-- |{% for variant in variants %} :-- |{% endfor %}
{% for task in tasks -%}
    {%- set outer_loop_index = loop.index0 -%}
    | trees={{task[0]}}<br/>leaves={{task[1]}}<br/>cols={{task[2]}}<br/> |
        {%- for variant in variants-%}
        {{- " {:,.2f} ".format(metrics[outer_loop_index][loop.index0]) -}} |
        {%- endfor %}
{% endfor %}
