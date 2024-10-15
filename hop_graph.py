from igraph import Graph
from config import args
from logger_config import logger
import json
import pickle
import os
from py2neo import Node,Relationship,Graph,Path,Subgraph


class Neo4jExample:

    def __init__(self, uri, user, password):
        self.graph = Graph(uri, auth=(user, password))

    def add_node(self, node_name):
        # 检查节点是否已存在
        if not self.graph.nodes.match("Node", name=node_name).first():
            node = Node("Node", name=node_name)
            self.graph.create(node)

    def add_edge(self, node1_name, node2_name):
        node1 = self.graph.nodes.match("Node", name=node1_name).first()
        node2 = self.graph.nodes.match("Node", name=node2_name).first()
        if node1 and node2:
            # 检查关系是否已存在（双向检查）
            if not (self.graph.relationships.match((node1, node2), "RELATED").first() or
                    self.graph.relationships.match((node2, node1), "RELATED").first()):
                relationship1 = Relationship(node1, "RELATED", node2)
                relationship2 = Relationship(node2, "RELATED", node1)
                self.graph.create(relationship1)
                self.graph.create(relationship2)

    def find_n_hop_neighbors(self, node_name, n):
        query = (
            f"MATCH (start:Node {{name: $node_name}})-[:RELATED*1..{n}]->(neighbor) "
            "RETURN DISTINCT neighbor.name AS name"
        )
        result = self.graph.run(query, node_name=node_name)
        return [record["name"] for record in result]


uri = 'bolt://10.10.2.106:7687'
user = "neo4j"
password = "04686763537"
neo4j_example = Neo4jExample(uri, user, password)

def graph_build():
    global neo4j_example
    # 定义三元组列表 (source, target, weight)，节点为字符串
    examples = json.load(open(args.train_path, 'r', encoding='utf-8'))
    for ex in examples:
        head_id, tail_id = ex['head_id'], ex['tail_id']
        neo4j_example.add_node(head_id)
        neo4j_example.add_node(tail_id)
        neo4j_example.add_edge(head_id, tail_id)


def get_n_hop_node(node_id, n_hop=0):
    n_hop_neighbors = neo4j_example.find_n_hop_neighbors(node_id, n_hop)
    return n_hop_neighbors