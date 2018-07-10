#!/bin/bash
DIRNAME=$(dirname $0)
docker build -t firemarmot/base $DIRNAME/base
docker build -t firemarmot/suitesparse $DIRNAME/suitesparse
docker build -t firemarmot/xerus $DIRNAME/xerus
docker build -t firemarmot $DIRNAME
