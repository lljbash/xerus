FROM firemarmot/suitesparse
USER root
WORKDIR "/tmp/"

# Install Xerus
RUN git clone https://git.hemio.de/xerus/xerus.git
WORKDIR "/tmp/xerus/"
RUN git checkout development && \
	cp ./docker/xerus/config.mk . && \
	make install -j $(cat /tmp/NCORES_1)

ENV LD_LIBRARY_PATH "/usr/local/lib/:$LD_LIBRARY_PATH"
ENV PYTHONPATH "/usr/local/lib/python2.7/site-packages/:$PYTHONPATH"
