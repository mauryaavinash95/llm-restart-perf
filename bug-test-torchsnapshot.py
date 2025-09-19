import torch
from torchsnapshot import StateDict, Snapshot

snapshot = Snapshot("/grand/VeloC/am6429/public/mp_rank_00_model_states.pt") 
try:
    print("Methods available for snapshot: ", snapshot.__dir__())
    manifest = snapshot.get_manifest()
    print("Manifest:")
    print(manifest)
    a = {"app_state": StateDict(ckpt={})}
    snapshot.restore(app_state=a)
    print("Restored state dict: ", a)
except Exception as e:
    print("Error occurred while restoring state dict:", e)
    print("Let's debug now....")
    import pdb; pdb.set_trace()
