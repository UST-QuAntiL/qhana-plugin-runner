FROM ubuntu:focal
WORKDIR /app

RUN apt-get -y update && apt-get install -y python3 python3-distutils python3-apt wget
RUN wget https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py
RUN python3 get-poetry.py
RUN rm get-poetry.py
ENV PATH=${PATH}:/root/.poetry/bin

COPY . /app
RUN poetry install
RUN poetry run flask create-db
RUN poetry run flask install

ENTRYPOINT ["poetry", "run", "flask", "run", "--host", "0.0.0.0"]
