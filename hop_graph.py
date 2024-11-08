from igraph import Graph
from config import args
from logger_config import logger
import json
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
import pickle
import os
entity_dict = get_entity_dict()
hop_graph = None
name_to_index=None
index_to_name=None

# 使用 Python 的 pickle 模块
def graph_build():
    file_name = '{}/igraph.pkl'.format(os.path.dirname(args.train_path))
    global hop_graph
    global name_to_index
    global index_to_name
    if os.path.exists(file_name):
        logger.info('Loading graph from {}'.format(file_name))
        with open(file_name, "rb") as f:
            hop_graph = pickle.load(f)
        logger.info('Loaded graph from {}'.format(file_name))
        name_to_index = {name: idx for idx, name in enumerate(hop_graph.vs["name"])}
        index_to_name = {idx:name for idx, name in enumerate(hop_graph.vs["name"])}
        return
    else:
        logger.info("构建多跳子图")
        hop_graph = Graph(directed=False)
        nodes = set()
        # 定义三元组列表 (source, target, weight)，节点为字符串
        examples = json.load(open(args.train_path, 'r', encoding='utf-8'))
        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            nodes.update([head_id, tail_id])
        hop_graph.add_vertices(list(nodes))
        logger.info("节点添加完成")
        # 添加边和权重
        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            hop_graph.add_edge(head_id, tail_id)
        logger.info("边添加完成")
        with open(file_name, "wb") as f:
            pickle.dump(hop_graph, f)
        name_to_index = {name: idx for idx, name in enumerate(hop_graph.vs["name"])}
        index_to_name = {idx: name for idx, name in enumerate(hop_graph.vs["name"])}


def get_n_hop_node(node_id, n_hop=0):
    global hop_graph
    if n_hop == 0:
        return []
    node_index = name_to_index.get(node_id)
    hops = n_hop  # 跳数
    try:
        neighborhood = hop_graph.neighborhood(vertices=node_index, order=hops)
        neighborhood_names = [index_to_name[index] for index in neighborhood]
    except:
        neighborhood_names = []
    return neighborhood_names