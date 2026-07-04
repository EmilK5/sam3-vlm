"""
graph.py

A tracking database to store object's coords,
scores and verification classification across
multiple passes
"""

import uuid 
import numpy as np

class OrchardNode:
    """
    Represents a single detected candidate
    entry
    """
    def __init__(self, box: list, score: float, found_in_pass: int):
        # Unique ID for the node
        self.id = f"node_{uuid.uuid4().hex[:8]}"

        # Box coords
        self.box = [float(coord) for coord in box]

        # Pass when it was found
        self.found_in_pass = int(found_in_pass)

        # Scores
        self.scores = {
            "detection_confidence": float(score),   # Initial SAM3 proposal conf
            "fruit_verification": 0.0,              # Self-audit score
            "leaf_verification": 0.0                # Self-audti score
        }

        # Classification tag
        self.classification = "unresolved"          # 'unresolved' | 'fruit' | 'leaf'
        self.tree_roi = None
        self.cached_leaf_boxes = None

    def to_dict(self) -> dict:
        """
        Helper to print out node details
        """
        return {
            "id": self.id,
            "box": self.box,
            "found_in_pass": self.found_in_pass,
            "scores": self.scores,
            "classification": self.classification
        }


class OrchardGraph:
    """
    Tracks all nodes discovered
    across all passes
    """
    def __init__(self):
        self.nodes = {}
    
    def clear(self):
        """
        Deletes all the nodes
        """
        self.nodes.clear()
        self.tree_roi = None
        self.cached_leaf_boxes = None
    
    def add_candidate(self, box: list, score: float, found_in_pass: int) -> str:
        """
        Creates ans stores
        a new node
        """
        node = OrchardNode(box, score, found_in_pass)
        self.nodes[node.id] = node
        return node.id
    
    def update_verdict(self, node_id: str, fruit_score: float, leaf_score: float):
        """
        Saves the score of the verification
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.scores["fruit_verification"] = float(fruit_score)
            node.scores["leaf_verification"] = float(leaf_score)

            # Decide whether it should be a fruit or leaf
            if fruit_score > 0.2 and leaf_score < 0.75: 
                node.classification = "fruit"
            else:
                node.classification = "leaf"

    def get_exemplars(self) -> tuple:
        """
        Groups verified boxes to serve as prompt
        for the next passes
        """
        pos_boxes = []
        neg_boxes = []

        for node in self.nodes.values():
            if node.classification == "fruit":
                pos_boxes.append(node.box)
            elif node.classification == "leaf":
                neg_boxes.append(node.box)
        
        return np.array(pos_boxes), np.array(neg_boxes)

    def to_dict(self) -> dict:
        """
        Serializes all nodes for JSON export
        """
        return {"nodes": [node.to_dict() for node in self.nodes.values()]}
