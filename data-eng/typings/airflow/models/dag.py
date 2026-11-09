from datetime import datetime


class DAG:
    def __init__(
        self,
        dag_id: str,
        start_date: datetime | None = None,
        schedule: str | None = None,
        catchup: bool = False,
        tags: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        self.dag_id = dag_id
        self.start_date = start_date
        self.schedule = schedule
        self.catchup = catchup
        self.tags = tags
        self.kwargs = kwargs

    def __enter__(self) -> "DAG":
        return self

    def __exit__(self, *args: object) -> None:
        return None
