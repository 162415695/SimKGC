#!/usr/bin/env bash

set -x
set -e

TASK=WN18RR

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_dino"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model /mnt/data/yhy/model/bert-base-uncased \
--pooling mean \
--lr 5e-4 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--neighbor-weight 0 \
--rerank-n-hop 0 \
--batch-size 2048 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--pre-batch 0 \
--finetune-t \
--epochs 500 \
--use-self-negative \
--workers 4 \
--max-to-keep 5 \
--use-dino \
--dino-loss \
--ema-decay 0.996 "$@"