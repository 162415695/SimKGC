#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_mamba_370m_new"
python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model "/mnt/data/yhy/model/mamba-370m-hf" \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--batch-size 8 \
--print-freq 200 \
--additive-margin 0.02 \
--use-self-negative \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 4 \
--max-to-keep 50 \
--add-extra-batch \
--extra-batch-limit -1 "$@"
