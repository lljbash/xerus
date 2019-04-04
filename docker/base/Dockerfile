FROM phusion/baseimage:0.9.22
USER root
WORKDIR "/tmp/"

# Get Ubuntu updates
RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade

# Set locale environment
RUN apt-get -y install locales sudo && \
    echo "C.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8

# Turn off forwarding of syslog messages to stdout
# https://github.com/phusion/baseimage-docker/issues/186
RUN touch /etc/service/syslog-forwarder/down

# Non-Python utilities and libraries
RUN apt-get -y install \
    cmake \
    doxygen \
    g++ \
    gfortran \
    git \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-python-dev \
    liblapack* \
    libopenblas-dev \
    libiberty-dev \
    binutils-dev \
    zlib1g-dev

# Install Python2 based environment
RUN apt-get -y install \
    python-dev \
    python-numpy

# Install Python3 based environment
RUN apt-get -y install \
    python3-dev \
    python3-numpy

# Clean apt
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Get number of physical cores to use
RUN lscpu | grep '^CPU(s)' | awk '{print $2}' > NCORES
RUN echo $(($(cat NCORES)-1)) > NCORES_1
