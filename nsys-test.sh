#!/bin/bash
nsys profile \
  --force-overwrite=true \
  -t cuda,nvtx \
  -o nsys-log.%p \
  "$@"