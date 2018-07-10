FROM firemarmot/suitesparse
USER root
WORKDIR "/tmp/"

# Install Xerus
RUN git clone https://git.hemio.de/xerus/xerus.git
WORKDIR "/tmp/xerus/"
RUN git checkout development && \
	cp config.mk.default config.mk && \
	sed -i "s|# BUILD_PYTHON_BINDINGS = TRUE|BUILD_PYTHON_BINDINGS = TRUE|g" config.mk && \
	sed -i "s|HIGH_OPTIMIZATION = TRUE|# HIGH_OPTIMIZATION = TRUE|g" config.mk && \
	sed -i "s|#INSTALL_LIB_PATH = /usr/local/lib64/|INSTALL_LIB_PATH = /usr/local/lib/|g" config.mk && \
	sed -i "s|#INSTALL_HEADER_PATH = /usr/local/include/|INSTALL_HEADER_PATH = /usr/local/include/|g" config.mk && \
	sed -i "s|#INSTALL_PYTHON_PATH = /usr/local/lib64/python2.7/site-packages/|INSTALL_PYTHON_PATH = /usr/local/lib/python2.7/site-packages|g" config.mk && \
	sed -i "s|SUITESPARSE = -lcholmod -lspqr|SUITESPARSE = -L/usr/local/lib/suitesparse -lcholmod -lspqr|g" config.mk && \
	make install -j $(cat /tmp/NCORES_1)

ENV LD_LIBRARY_PATH "/usr/local/lib/:$LD_LIBRARY_PATH"
ENV PYTHONPATH "/usr/local/lib/python2.7/site-packages/:$PYTHONPATH"
