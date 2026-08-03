import json

from graph import GRAPH_SCHEMA_VERSION, OrchardGraph


def test_to_dict_empty_graph():
    graph = OrchardGraph()

    data = graph.to_dict()

    assert data["graph_schema_version"] == GRAPH_SCHEMA_VERSION
    assert data["nodes"] == []
    assert data["metadata"] == {
        "tree_roi": None,
        "cached_leaf_roi": None,
    }


def test_to_dict_contains_node_fields():
    graph = OrchardGraph()
    node_id = graph.add_candidate(
        box=[1.0, 2.0, 3.0, 4.0],
        score=0.9,
        found_in_pass=1,
    )
    graph.update_verdict(
        node_id,
        fruit_score=0.8,
        leaf_score=0.1,
    )

    data = graph.to_dict()

    assert {
        "graph_schema_version",
        "nodes",
        "metadata",
    } <= data.keys()

    assert data["graph_schema_version"] == GRAPH_SCHEMA_VERSION
    assert data["metadata"] == {
        "tree_roi": None,
        "cached_leaf_roi": None,
    }

    assert len(data["nodes"]) == 1
    node_dict = data["nodes"][0]

    assert node_dict["id"] == node_id
    assert node_dict["box"] == [1.0, 2.0, 3.0, 4.0]
    assert node_dict["found_in_pass"] == 1
    assert node_dict["classification"] == "fruit"
    assert node_dict["scores"]["fruit_verification"] == 0.8

def test_to_dict_is_json_serializable_with_multiple_nodes():
    graph = OrchardGraph()
    ids = [
        graph.add_candidate(box=[0.0, 0.0, 10.0, 10.0], score=0.9, found_in_pass=1),
        graph.add_candidate(box=[20.0, 20.0, 30.0, 30.0], score=0.7, found_in_pass=2),
    ]
    data = graph.to_dict()

    # round-trips through JSON without error and preserves node count/ids
    reloaded = json.loads(json.dumps(data))
    assert len(reloaded["nodes"]) == 2
    assert {n["id"] for n in reloaded["nodes"]} == set(ids)


def test_to_dict_reflects_verdict_leaf():
    graph = OrchardGraph()
    node_id = graph.add_candidate(box=[1.0, 1.0, 2.0, 2.0], score=0.5, found_in_pass=1)
    # low fruit score -> classified as leaf by update_verdict's gate
    graph.update_verdict(node_id, fruit_score=0.1, leaf_score=0.9)
    node_dict = graph.to_dict()["nodes"][0]
    assert node_dict["classification"] == "leaf"


def test_from_dict_accepts_legacy_nodes_only_payload():
    payload = {
        "nodes": [
            {
                "id": "node_legacy",
                "box": [1.0, 2.0, 3.0, 4.0],
                "found_in_pass": 1,
                "scores": {
                    "detection_confidence": 0.9,
                    "fruit_verification": 0.8,
                    "leaf_verification": 0.1,
                },
                "classification": "fruit",
            }
        ]
    }

    graph = OrchardGraph.from_dict(payload)

    assert list(graph.nodes) == ["node_legacy"]
    assert graph.nodes["node_legacy"].classification == "fruit"
    assert graph.tree_roi is None
    assert graph.cached_leaf_roi is None