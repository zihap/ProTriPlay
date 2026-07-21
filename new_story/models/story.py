from typing import List, Dict, Any


class StoryNode:
    def __init__(self, node_id: str, description: str, branches: List[Dict] = None):
        self.node_id = node_id
        self.description = description
        self.branches = branches if branches else []

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "description": self.description,
            "branches": self.branches
        }


class StoryOutline:
    def __init__(self, title: str, background: str, main_nodes: List[StoryNode] = None):
        self.title = title
        self.background = background
        self.main_nodes = main_nodes if main_nodes else []
        self.current_node_index = 0

    def add_node(self, node: StoryNode):
        self.main_nodes.append(node)

    def get_current_node(self):
        if self.current_node_index < len(self.main_nodes):
            return self.main_nodes[self.current_node_index]
        return None

    def advance_node(self):
        if self.current_node_index < len(self.main_nodes) - 1:
            self.current_node_index += 1
            return True
        return False

    def get_node_by_id(self, node_id: str):
        for node in self.main_nodes:
            if node.node_id == node_id:
                return node
        return None

    def to_dict(self):
        return {
            "title": self.title,
            "background": self.background,
            "main_nodes": [node.to_dict() for node in self.main_nodes],
            "current_node_index": self.current_node_index
        }
