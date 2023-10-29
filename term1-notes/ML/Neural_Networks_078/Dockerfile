FROM python:3.10-slim AS base
RUN apt-get -y update && apt-get -y install \
    git

FROM base AS dependencies
COPY ./requirements.txt .
RUN pip install --user -r requirements.txt

FROM dependencies AS part1
COPY iris.dat .
COPY part1_nn_lib.py .
CMD python part1_nn_lib.py

FROM dependencies AS part2
COPY housing.csv .
COPY part2_house_value_regression.py .
CMD part2_house_value_regression.py
