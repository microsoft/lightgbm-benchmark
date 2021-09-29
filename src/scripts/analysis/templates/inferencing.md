# LightGBM Inferencing Benchmark

> The report below has been automatically generated with results from the [lightgbm-benchmark repo](https://github.com/microsoft/lightgbm-benchmark).

IMPORTANT: this is work in progress, to check out current work items check the [project open issues](https://github.com/microsoft/lightgbm-benchmark/issues)

## Variants

| Index | Framework | Version | Build |
| :-- | :-- | :-- | :-- |
{% for variant in variants -%}
| {{variant[0]}}_{{loop.index}} | {{variant[0]}} | {{variant[1]}} | {{variant[2]}} |
{% endfor %}

## Metric time_inferencing

| &nbsp; |{% for variant in variants %} {{variant[0]}}_{{loop.index}} |{% endfor %}
| :-- |{% for variant in variants %} :-- |{% endfor %}
{% for config in configs -%}
    {%- set outer_loop_index = loop.index -%}
    | trees={{config[0]}}<br/>leaves={{config[1]}}<br/>cols={{config[2]}}<br/> |
        {%- for variant in variants %} {{ "{:,.4f}".format(metrics[outer_loop_index-1][loop.index-1]) }} |{% endfor %}
{% endfor %}
