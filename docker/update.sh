#!/bin/bash
DIRNAME=$(dirname $0)
docker image rm -f firemarmot/xerus
docker image rm -f firemarmot
docker build -t firemarmot/xerus $DIRNAME/xerus
docker build -t firemarmot $DIRNAME
