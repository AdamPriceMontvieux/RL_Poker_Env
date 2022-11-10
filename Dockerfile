FROM stepankuzmin/pytorch-notebook
WORKDIR /app

#ADD ./Requirements.txt /Requirements.txt

COPY . /app
RUN pip install "ray[rllib]" 
RUN pip install -e /app


ENV PYTHONPATH="/app"



# To build the docker image:
#   docker build -t poker:1.0 .
#
# run with docker-compose 
