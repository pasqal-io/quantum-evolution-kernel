from qek.data.datatools import add_graph_coord

RADIUS = 5.001
EPS = 0.01


def test_add_graph_coord(original_ptcfm_data, ptcfm_maps) -> None:
    counter = 0
    for graph in original_ptcfm_data:
        print(graph)
        augmented_graph = add_graph_coord(data=graph, blockade_radius=RADIUS, node_mapping=ptcfm_maps[0], edge_mapping=ptcfm_maps[1])
        print(augmented_graph)
        breakpoint()
        if counter == 2:
            break
        counter += 1
    
