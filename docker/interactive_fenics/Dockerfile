FROM firemarmot/xerus AS xerus

FROM quay.io/fenicsproject/stable:2017.1.0
USER root
WORKDIR "/tmp/"

# Non-Python utilities and libraries
RUN apt-get -qq update && \
    apt-get -y install \
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

# Check Python versions
RUN python -c'import sys; raise SystemExit("{0}.{1}".format(*sys.version_info) != "2.7")'
RUN python3 -c'import sys; raise SystemExit("{0}.{1}".format(*sys.version_info) != "3.5")'

# Clean apt
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create directories
RUN mkdir -p /usr/local/lib && \
	mkdir -p /usr/local/include && \
	mkdir -p /usr/local/lib/python2.7/site-packages && \
	mkdir -p /usr/local/lib/python3.5/site-packages

COPY --from=xerus /usr/local/include/suitesparse /usr/local/include/suitesparse
COPY --from=xerus /usr/local/lib/suitesparse /usr/local/lib/suitesparse
COPY --from=xerus /usr/local/include/xerus.h /usr/local/include
COPY --from=xerus /usr/local/include/xerus /usr/local/include
COPY --from=xerus /usr/local/lib/libxerus_misc.so /usr/local/lib
COPY --from=xerus /usr/local/lib/libxerus.so /usr/local/lib
COPY --from=xerus /usr/local/lib/python2.7/site-packages/xerus.so /usr/local/lib/python2.7/site-packages
COPY --from=xerus /usr/local/lib/python3.5/site-packages/xerus.so /usr/local/lib/python3.5/site-packages

# Install custom Python2 packages
RUN pip install cython && pip install ttpy matplotlib2tikz joblib

# Set environment variables
USER fenics
RUN mkdir -p $(python -m site --user-site) && \
	echo /usr/local/lib/python2.7/site-packages/ >$(python -m site --user-site)/xerus.pth && \
	mkdir -p $(python3 -m site --user-site) && \
	echo /usr/local/lib/python3.5/site-packages/ >$(python3 -m site --user-site)/xerus.pth
ENV LD_LIBRARY_PATH "/usr/local/lib/:/usr/local/lib/suitesparse:$LD_LIBRARY_PATH"

WORKDIR "/home/fenics/"
USER root
