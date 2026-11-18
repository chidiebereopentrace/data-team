# Image for running populate_local_db (live discovery + local snapshot refresh) inside Docker.
# Used by the "setup" service in docker-compose; repo is mounted at runtime.
FROM python:3.11-slim

WORKDIR /app

# The population task does not need the full ML/GIS development environment.
COPY data/local/requirements-populate.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Default: run the full populate script (overridden in compose)
CMD ["bash", "data/local/scripts/populate_local_db.sh"]
