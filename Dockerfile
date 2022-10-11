FROM python:3.10

LABEL org.opencontainers.image.source="https://github.com/UST-QuAntiL/qhana-plugin-runner"

# install git and remove caches again in same layer
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd gunicorn


ENV FLASK_APP=qhana_plugin_runner
ENV PLUGIN_FOLDERS=/app/plugins:/app/extra-plugins:/app/git-plugins
ENV TEMPLATE_FOLDERS=/app/templates


# can be server or worker
ENV CONTAINER_MODE=server
ENV DEFAULT_LOG_LEVEL=INFO
ENV CONCURRENCY=2
ENV CELERY_WORKER_POOL=threads

# make directories and set user rights
RUN mkdir /venv && mkdir --parents /app/instance \
    &&  mkdir --parents /app/extra-plugins && \
    mkdir --parents /app/git-plugins \
    && chown --recursive gunicorn /app \
    && chmod --recursive u+rw /app \
    && mkdir /home/gunicorn && chown --recursive gunicorn /home/gunicorn

# Wait for database
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.9.0/wait /wait
RUN chmod +x /wait

RUN python -m pip install poetry gunicorn

COPY --chown=gunicorn . /app

RUN python -m poetry export --without-hashes --extras=psycopg2 --extras=PyMySQL --format=requirements.txt -o requirements.txt && python -m pip install -r requirements.txt

VOLUME ["/app/instance"]
ENV QHANA_PLUGIN_RUNNER_INSTANCE_FOLDER="/app/instance"

EXPOSE 8080

ENTRYPOINT ["python","-m", "invoke", "start-docker"]
