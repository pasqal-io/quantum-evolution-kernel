from time import sleep
import torch_geometric.datasets as pyg_dataset


if __name__ == "__main__":
    # Attempt to force download of the PTC_FM dataset, which has difficulties on some
    # platforms. We suspect that it's a race condition somewhere in pytorch geometric.
    exn = None
    for i in range(0, 10):
        sleep(i * i)
        try:
            print(f"Attempt {i+1} to download dataset")
            pyg_dataset.TUDataset(root="dataset", name="PTC_FM")
            print(f"Attempt {i+1} to download dataset succeeded")
            exn = None
            break
        except FileNotFoundError as e:
            print(f"Attempt {i+1} to download failed: {e}")
            exn = e
    if exn is not None:
        raise exn
