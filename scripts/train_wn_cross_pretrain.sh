#!/usr/bin/env bash

set -x
set -e

TASK=WN18RR

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_trans_pretrain"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi


neighbor_weight=0.05
rerank_n_hop=2
if [ "${TASK}" = "WN18RR" ]; then
# WordNet is a sparse graph, use more neighbors for re-rank
  rerank_n_hop=5
fi
if [ "${TASK}" = "wiki5m_ind" ]; then
# for inductive setting of wiki5m, test nodes never appear in the training set
  neighbor_weight=0.0
fi
python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model /mnt/data/yhy/model/bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--batch-size 1024 \
--print-freq 20 \
--use-amp \
--additive-margin 0.02 \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--use-self-negative \
--workers 4 \
--max-to-keep 5 \
--pretrained-ckpt /mnt/data/yhy/projects/SimKGC_git/checkpoint/WN18RR_bert_base_best/model_last.mdl \
--use-cross-attention  "$@"
