# Profiler

A flexible and easy-to-use timing utility for tracking execution times across your application. The profiler supports nested timing operations, multiple executions tracking, and provides both simple and detailed metrics output.

## Features

- Easy start/stop timing operations
- Context manager for timing code blocks
- Function decorator for timing function executions
- Nested timing support (e.g., "retrieval.chunking", "retrieval.embedding")
- Multiple execution tracking with count and average duration
- Hierarchical timing structure
- Built-in logging integration

## Usage

### Basic Usage

```python
from core.utils.profiler import Profiler

profiler = Profiler()

# Using context manager
with profiler.track("retrieval.chunking"):
    # Your code here
    time.sleep(0.1)

# Get timing metrics
metrics = profiler.get_metrics()
```

### Nested Operations

```python
with profiler.track("retrieval.chunking"):
    time.sleep(0.1)
    with profiler.track("retrieval.embedding"):
        time.sleep(0.2)
```

### Function Decorator

```python
@profiler.track_func("decorated.function")
def example_function():
    time.sleep(0.1)

example_function()
```

### Multiple Executions

```python
for _ in range(3):
    with profiler.track("repeated.operation"):
        time.sleep(0.1)
```

## Output Examples

### Simple Metrics (without counts)

```python
profiler.get_metrics()
```

Output:
```python
{
    'retrieval.chunking': 0.3104,
    'retrieval.embedding': 0.2053,
    'generation': 0.3052,
    'repeated.operation': 0.1013,
    'decorated.function': 0.1023
}
```

### Detailed Metrics (with counts)

```python
profiler.get_metrics(include_counts=True)
```

Output:
```python
{
    'retrieval.chunking': {
        'duration': 0.3104,
        'count': 1,
        'avg_duration': 0.3104
    },
    'retrieval.embedding': {
        'duration': 0.2053,
        'count': 1,
        'avg_duration': 0.2053
    },
    'generation': {
        'duration': 0.3052,
        'count': 1,
        'avg_duration': 0.3052
    },
    'repeated.operation': {
        'duration': 0.1013,
        'count': 3,
        'avg_duration': 0.0337
    },
    'decorated.function': {
        'duration': 0.1023,
        'count': 1,
        'avg_duration': 0.1023
    }
}
```

## Logging

The profiler integrates with the application's logging system and provides detailed timing information at various log levels:
- INFO: Timer start/stop events with elapsed times
- DEBUG: Detailed timing information and context manager operations
- ERROR: Timer operation errors (e.g., stopping non-started timer)
- WARNING: Timer hierarchy issues

## Example Use Cases

1. **API Response Times**
   ```python
   with profiler.track("api.response"):
       response = api.get_data()
   ```

2. **Database Operations**
   ```python
   with profiler.track("db.query"):
       results = db.execute_query()
   ```

3. **Complex Processing Pipeline**
   ```python
   with profiler.track("pipeline"):
       with profiler.track("pipeline.preprocess"):
           data = preprocess()
       with profiler.track("pipeline.process"):
           results = process(data)
       with profiler.track("pipeline.postprocess"):
           final = postprocess(results)
   ```
