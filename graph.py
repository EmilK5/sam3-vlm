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

        # FM+V-IP verifier trace (populated only when verifier="vip"); left None
        # under the default IoC verifier so to_dict output is unchanged.
        self.vip_chain = None                        # [{"q": text, "a": "yes/no/unsure"}, ...]
        self.vip_posterior = None                    # posterior over classes, as a list

        # Belief-state support statistics (proposal §"Candidate Graph").
        self.support = 1                             # k_i: number of detections backing this track
        self.signatures = set()                      # Q_i: distinct query signatures that hit it
        self.jitter = 0.0                            # Delta_i: running-mean center displacement (px)
        self.area = self._box_area()                 # A_i: representative box area (px^2)

        # Optional instance mask (overlap_mode="mask"): a boolean numpy array
        # cropped to this node's box (row 0 / col 0 = box's y1 / x1). Not
        # serialized in to_dict. None under the default box-overlap mode.
        self.mask = None

    def _box_area(self) -> float:
        return float((self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))

    def _center(self, box) -> tuple:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def reinforce(self, box: list, signature: str):
        """Record a cross-pass re-detection of this track.

        Increments support k, adds the query signature, and folds the re-detection's
        center displacement (from this node's representative box) into jitter as a
        running mean over the re-detections.
        """
        cx, cy = self._center(box)
        rx, ry = self._center(self.box)
        displacement = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5

        self.support += 1
        if signature is not None:
            self.signatures.add(signature)
        n_redetections = self.support - 1            # this is the k-1'th re-detection
        self.jitter += (displacement - self.jitter) / n_redetections

    def to_dict(self) -> dict:
        """
        Helper to print out node details
        """
        d = {
            "id": self.id,
            "box": self.box,
            "found_in_pass": self.found_in_pass,
            "scores": self.scores,
            "classification": self.classification,
            "support": self.support,
            "jitter": self.jitter,
            "area": self.area,
            "signatures": sorted(self.signatures),
        }
        if self.vip_chain is not None:
            d["vip_chain"] = self.vip_chain
        if self.vip_posterior is not None:
            d["vip_posterior"] = self.vip_posterior
        return d


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
        self.cached_leaf_roi = None
    
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
