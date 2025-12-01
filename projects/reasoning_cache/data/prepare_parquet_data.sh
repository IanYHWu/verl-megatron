#!/usr/bin/env bash

export VERL_HOME=${VERL_HOME:-"${HOME}/verl"}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/acemath_rl_30b_a3b_inst_hard_train.parquet"}
export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/data/acemath_rl_30b_a3b_inst_hard_test.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${VERL_HOME}/data"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/HerrHruby/acemath_rl_30b_a3b_inst_hard/resolve/main/train.parquet"
fi

if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/HerrHruby/acemath_rl_30b_a3b_inst_hard/resolve/main/test.parquet"
fi
