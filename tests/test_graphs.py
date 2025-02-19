import torch_geometric.datasets as pyg_dataset


def test_graph_init() -> None:
    # Load dataset
    original_ptcfm_data = [d for d in pyg_dataset.TUDataset(root="dataset", name="PTC_FM")]
    print(len(original_ptcfm_data))
