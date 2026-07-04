import json

from graph import OrchardGraph


def test_to_dict_empty_graph():
    graph = OrchardGraph()
    assert graph.to_dict() == {"nodes": []}


def test_to_dict_contains_node_fields():
    graph = OrchardGraph()
    node_id = graph.add_candidate(box=[1.0, 2.0, 3.0, 4.0], score=0.9, found_in_pass=1)
    graph.update_verdict(node_id, fruit_score=0.8, leaf_score=0.1)

    data = graph.to_dict()

    assert list(data.keys()) == ["nodes"]
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
