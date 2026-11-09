class BaseOperator:
    def __init__(self, task_id: str, **kwargs: object) -> None:
        self.task_id = task_id

    def __rshift__(self, other: "BaseOperator") -> "BaseOperator":
        return other
