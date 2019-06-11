FROM firemarmot/suitesparse
USER root
WORKDIR "/tmp/"

# Check Python versions
RUN python -c'import sys; raise SystemExit("{0}.{1}".format(*sys.version_info) != "2.7")'
RUN python3 -c'import sys; raise SystemExit("{0}.{1}".format(*sys.version_info) != "3.5")'

# Install Xerus
RUN git clone https://git.hemio.de/xerus/xerus.git
WORKDIR "/tmp/xerus/"
RUN git checkout vmc && \
	git pull && \
	cp ./docker/xerus/config.mk . && \
	mkdir -p /usr/local/lib && \
	mkdir -p /usr/local/include && \
	mkdir -p /usr/local/lib/python2.7/site-packages && \
	mkdir -p /usr/local/lib/python3.5/site-packages && \
	make install -j $(cat /tmp/NCORES_1)

RUN mkdir -p $(python -m site --user-site) && \
	echo /usr/local/lib/python2.7/site-packages/ >$(python -m site --user-site)/xerus.pth && \
	mkdir -p $(python3 -m site --user-site) && \
	echo /usr/local/lib/python3.5/site-packages/ >$(python3 -m site --user-site)/xerus.pth
ENV LD_LIBRARY_PATH "/usr/local/lib/:$LD_LIBRARY_PATH"
