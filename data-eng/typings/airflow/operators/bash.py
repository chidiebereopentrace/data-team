from airflow.models.baseoperator import BaseOperator


class BashOperator(BaseOperator):
    def __init__(self, task_id: str, bash_command: str, **kwargs: object) -> None:
        super().__init__(task_id, **kwargs)
        self.bash_command = bash_command
