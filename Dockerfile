FROM python:3.9

LABEL org.opencontainers.image.source="https://github.com/UST-QuAntiL/qhana-plugin-runner"

# install git and remove caches again in same layer
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd gunicorn


ENV FLASK_APP=qhana_plugin_runner
ENV FLASK_ENV=production
ENV PLUGIN_FOLDERS=/app/plugins:/app/extra-plugins:/app/git-plugins
ENV TEMPLATE_FOLDERS=/app/templates


# can be server or worker
ENV CONTAINER_MODE=server
ENV DEFAULT_LOG_LEVEL=INFO
ENV CONCURRENCY=2
ENV CELERY_WORKER_POOL=threads

# make directories and set user rights
RUN mkdir /venv && mkdir --parents /app/instance &&  mkdir --parents /app/extra-plugins && mkdir --parents /app/git-plugins \
    && chown --recursive gunicorn /app && chmod --recursive u+rw /app && chown --recursive gunicorn /venv

# Wait for database
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.7.3/wait /wait
RUN chmod +x /wait

USER gunicorn

# create and activate virtualenv as gunicorn user to allow installing additional dependencies later
ARG VIRTUAL_ENV=/venv
RUN python -m venv ${VIRTUAL_ENV}

# change path to include virtualenv first (affects all following commands)
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

RUN python -m pip install --upgrade pip && python -m pip install gunicorn poetry invoke

COPY --chown=gunicorn . /app

RUN python -m pip install .

VOLUME ["/app/instance"]

EXPOSE 8080

ENTRYPOINT ["python", "-m", "invoke", "start-docker"]
