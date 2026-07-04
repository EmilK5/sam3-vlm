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
