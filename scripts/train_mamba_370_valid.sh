#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_mamba_390_whole_valid"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

neighbor_weight=0.05
rerank_n_hop=2
if [ "${task}" = "WN18RR" ]; then
# WordNet is a sparse graph, use more neighbors for re-rank
  rerank_n_hop=5
fi
if [ "${task}" = "wiki5m_ind" ]; then
# for inductive setting of wiki5m, test nodes never appear in the training set
  neighbor_weight=0.0
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
--batch-size 100 \
--print-freq 100 \
--additive-margin 0.02 \
--use-self-negative \
--use-amp \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 4 \
--max-to-keep 50 \
--add-extra-batch \
--extra-batch-limit -1 "$@"
