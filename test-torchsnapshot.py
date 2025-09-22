import torch
import time
import os
import sys
from torchsnapshot import StateDict, Snapshot
import pickle
SIZE = 1<<15

def foo():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    x = torch.randn(SIZE, device=device)
    y = torch.randn(SIZE, device=device) # torch.nn.Linear(1000, 1000)(x).to(device)
    z = torch.randn(SIZE, device=device) # torch.nn.Linear(1000, 1000)(y).to(device)
    a = "some string"*100

    obj = {
        "x": x,
        "y": y,
        "z": z,
        "a": a,
    }
    ckpt_dict = {"app_state": StateDict(ckpt=obj)}

    path = "/dev/shm/torchsnapshot_test"
    t = time.time()
    if os.path.exists(path):
        os.system(f"rm -rf {path}")
    snapshot = Snapshot.take(path=path, app_state=ckpt_dict, replicated=[])
    print(f"Snapshot saved at {path} in {time.time() - t:.2f} seconds of size {len(pickle.dumps(obj))} bytes")
    print(f"Snapshot obj: ", obj)
    time.sleep(5)

    snapshot = Snapshot(path=path)
    restore_dict = {"app_state": StateDict(ckpt={})}
    t = time.time()
    snapshot.restore(app_state=restore_dict)
    res_obj = restore_dict["app_state"].data['ckpt']
    print(f"Restored obj: ", res_obj)
    print(f"Snapshot restored in {time.time() - t:.2f} seconds")
    assert torch.allclose(res_obj["x"], x)
    assert torch.allclose(res_obj["y"], y)
    assert torch.allclose(res_obj["z"], z)
    assert res_obj["a"] == a

if __name__ == "__main__":
    foo()
    sys.exit(0)