import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import pickle
import numpy as np
import torch
import smplx
import trimesh

for name, value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if name not in np.__dict__:
        setattr(np, name, value)

print("RUNNING NEW SCRIPT")
# Load data
with open("data/raw/BMCLab.pkl", "rb") as f:
    data = pickle.load(f)

subject = list(data.keys())[0]
recording = list(data[subject].keys())[0]
sample = data[subject][recording]

pose = torch.tensor(sample["pose"], dtype=torch.float32)
trans = torch.tensor(sample["trans"], dtype=torch.float32)
betas = torch.zeros((1, 10))  # dataset uses neutral shape

# Load SMPL model
model = smplx.create(
    "models",
    model_type="smpl",
    gender="neutral"
)

# Take one frame
frame = 0

output = model(
    global_orient=pose[frame, :3].unsqueeze(0),
    body_pose=pose[frame, 3:].unsqueeze(0),
    transl=trans[frame].unsqueeze(0),
    betas=betas
)

vertices = output.vertices.detach().cpu().numpy()

# Save mesh as OBJ instead of using mesh.show()
mesh = trimesh.Trimesh(vertices=vertices[0], faces=model.faces)

output_path = "data/processed/test_smpl_frame.obj"
mesh.export(output_path)

print("Saved mesh to:", output_path)
print("Open it in Blender or Windows 3D Viewer.")
