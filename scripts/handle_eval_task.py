from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.eval_task_worker import SimTaskExecutor
from metta.tools.utils.auto_config import auto_stats_server_uri


async def complete_remote_eval(eval_task_id: str):
    backend_url = auto_stats_server_uri()
    if not backend_url:
        raise ValueError("No backend URL found")
    client = EvalTaskClient(backend_url)
    task_executor = SimTaskExecutor(backend_url)
    task = client.get_task_by_id(eval_task_id)
    task_result = await task_executor.execute_task(task)
    return task_result
