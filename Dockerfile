FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
MAINTAINER Kiran Shila <me@kiranshila.com>
ADD . /src
WORKDIR /src
run apt-get update
RUN apt-get install -y autoconf libtool python-is-python3
RUN ./bootstrap
RUN ./configure
RUN make
