## This file defines functions to generate performance traces,
## using the chrome tracing format.

# Just add `@trace` decorator to any function to trace it.
# Or just add `with tracer("my_section"):` to trace a block.
# At the end of your program call `save_trace()` to save the trace to a file.
# Take this file to the chrome://tracing page to view the trace.
# Easy, no drama, you control everything!

# Example usage:
#
# @trace
# def my_function(a, b):
#     return a + b
#
# class MyClass:
#     @trace
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#
#     @trace
#     def my_method(self, a, b):
#         return a + b
#
#    my_class = MyClass(1, 2)
#    my_class.my_method(3, 4)
#
# with tracer("my_section"):
#     my_function(1, 2)
#
# save_trace("trace.json")

import json
import os
import threading
import time
import traceback

start_time = time.time()
trace_events = []


def trace(fn):
    """Adds tracing to a function.
    Usage:
    @trace
    def my_function(a, b):
        return a + b
    my_function(1, 2)
    """

    def trace_wrapper(*args, **kwargs):
        name = fn.__name__
        # Is the function bound to a class?
        if args and hasattr(args[0], "__class__"):
            cls = args[0].__class__
            name = f"{cls.__name__}.{name}"
        pid = os.getpid()
        tid = threading.get_ident()
        # make stack_trace in format of function>function>function
        stack = traceback.extract_stack()
        stack_trace = ">".join([frame.name for frame in stack if frame.name != "trace_wrapper"])
        trace_events.append(
            {
                "name": name,
                "category": "function",
                "ph": "B",
                "pid": pid,
                "tid": tid,
                "ts": int((time.time() - start_time) * 1000000),
                "args": {
                    "filename": fn.__code__.co_filename,
                    "lineno": fn.__code__.co_firstlineno,
                    "stack_trace": stack_trace,
                },
            }
        )
        result = fn(*args, **kwargs)
        trace_events.append(
            {
                "name": name,
                "category": "function",
                "ph": "E",
                "pid": pid,
                "tid": tid,
                "ts": int((time.time() - start_time) * 1000000),
            }
        )
        return result

    return trace_wrapper


class Tracer:
    """Helper class for with tracer("my_section"):"""

    def __init__(self, name: str):
        self.name = name
        self.start = 0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.duration = self.end - self.start
        trace_events.append(
            {
                "name": self.name,
                "category": "function",
                "ph": "X",
                "pid": os.getpid(),
                "tid": threading.get_ident(),
                "ts": int((self.start - start_time) * 1000000),
                "dur": int(self.duration * 1000000),
            }
        )


def tracer(name: str):
    """Tracing a block with statement
    Usage:
    with tracer("my_section"):
        # Your code here
    """
    return Tracer(name)


def save_trace(filename):
    ## Saves the trace to a file.
    with open(filename, "w") as f:
        data = {
            "traceEvents": trace_events,
            "displayTimeUnit": "ms",
        }
        json.dump(data, f)
