from airflow.models.baseoperator import BaseOperator


class EmptyOperator(BaseOperator):
    def __init__(self, task_id: str, **kwargs: object) -> None:
        super().__init__(task_id, **kwargs)
