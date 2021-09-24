## Usage

``` text hl_lines="1"
> python -m src/scripts/partition_data/partition.py -h

usage:
Partitions input data (text/lines) into chunks for parallel processing.

NOTE: current script assumes all records are independent.

 [-h] --input INPUT --output OUTPUT --mode {chunk,roundrobin}
 --number NUMBER [--verbose VERBOSE] [--custom_properties CUSTOM_PROPERTIES]

optional arguments:
  -h, --help            show this help message and exit

Partitioning arguments:
  --input INPUT         file/directory to split
  --output OUTPUT       location to store partitioned files
  --mode {chunk,roundrobin}
                        Partitioning mode
  --number NUMBER       If roundrobin number of partition, if chunk number of records per partition

General parameters:
  --verbose VERBOSE     set True to show DEBUG logs
  --custom_properties CUSTOM_PROPERTIES
                        provide custom properties as json dict
```

::: src.scripts.partition_data.partition
