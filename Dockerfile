FROM ubuntu:jammy AS build
WORKDIR /app

RUN apt update && apt install -y python3 python3-pip

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED 1

COPY entrypoint.sh /entrypoint.sh
RUN chmod a+x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

COPY . /app

CMD python3 main.py
