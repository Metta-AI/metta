## This file defines functions to generate traces, in the chrome tracing format.

# Just add @trace decorator to any function to trace it.
# At the end of your program call `save_trace()` to save the trace to a file.

import json
import os
import threading
import time
import traceback

start_time = time.time()
trace_events = []


def trace(fn):
    ## Adds tracing to a function.
    def trace_wrapper(*args, **kwargs):
        name = fn.__name__
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


def save_trace(filename):
    ## Saves the trace to a file.
    with open(filename, "w") as f:
        data = {
            "traceEvents": trace_events,
            "displayTimeUnit": "ms",
        }
        json.dump(data, f)
