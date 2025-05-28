FROM tensorflow/tensorflow:latest-gpu

RUN apt update && apt install -y python3-pip git curl

RUN pip install --upgrade pip && pip install pipenv

WORKDIR /workspace

COPY Pipfile Pipfile.lock ./

RUN pipenv install --deploy --system

EXPOSE 8888
