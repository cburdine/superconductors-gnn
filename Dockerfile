FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime 

# create project directory:
RUN mkdir /sc-gnn/
WORKDIR /sc-gnn/

# update pip & install packages:
COPY ./requirements.txt /requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt

# Set visible devices (optional):
#ENV CUDA_VISIBLE_DEVICES=0
