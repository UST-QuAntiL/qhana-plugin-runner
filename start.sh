#!/bin/sh
poetry run flask create-db
poetry run flask run --host 0.0.0.0
