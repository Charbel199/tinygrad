from __future__ import annotations
import zipfile, math, random, tempfile, urllib.request, shutil
from pathlib import Path

import numpy as np
from tinygrad import Tensor
from tinygrad.nn import Linear, BatchNorm2d
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import Timing
from tinygrad import TinyJit, dtypes
from tqdm import tqdm, trange


######## Paths ########
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELNET_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"

######## Visualization Helper ########
def show_pointcloud(pc,
                    s: int = 3,            # marker size
                    elev: int = 20, azim: int = 45,   # view angle
                    figsize=(5, 5)):
    """
    Quick 3-D scatter plot for a (N,3) numpy array or tinygrad Tensor.

    Usage
    -----
    pc, _ = dset.sample_from_class("airplane")   # or dset[123]
    show_pointcloud(pc)
    """
    import numpy as np
    import matplotlib.pyplot as plt              # Matplotlib is in the std Colab/venv stacks
    from mpl_toolkits.mplot3d import Axes3D      # noqa: F401 (side-effect import)

    # tinygrad Tensor ➜ numpy
    if not isinstance(pc, np.ndarray):
        pc = pc.numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                    c=pc[:, 2],                 # color by Z for a bit of depth
                    cmap='viridis',
                    s=s, depthshade=False)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

######## Download Dataset ########
def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            print("Downloading", url)
            urllib.request.urlretrieve(url, tmp.name)
            shutil.move(tmp.name, dest)
    return dest

def download_modelnet40() -> Path:
    zfile = _download(MODELNET_URL, DATA_DIR / "modelnet40_normal_resampled.zip")
    out   = DATA_DIR / "ModelNet40"
    if not out.exists():
        print("Extracting ModelNet40 ...")
        with zipfile.ZipFile(zfile) as zf:
            zf.extractall(DATA_DIR)
    return out

######## PointNet Model ########
class BatchNorm1D:
    def __init__(self, c: int): self.bn = BatchNorm2d(c)
    def __call__(self, x: Tensor) -> Tensor:
        # x is originally (B, N, C)
        if x.ndim == 2:
            return self.bn(x[:, :, None, None]).reshape(x.shape)
        if x.ndim == 3:
            return self.bn(x.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1) # (B, N, C) -> (B, C, N, 1) -> (B, N, C)
        raise ValueError("expect 2-D or 3-D")

class TNet:
    def __init__(self, k: int):
        self.k  = k
        self.id = Tensor.eye(k).reshape(1, k*k)
        self.fc1, self.bn1 = Linear(k, 64),   BatchNorm1D(64)
        self.fc2, self.bn2 = Linear(64, 64),   BatchNorm1D(64)
        self.fc3, self.bn3 = Linear(64,128),  BatchNorm1D(128)
        self.fc4, self.bn4 = Linear(128,1024),BatchNorm1D(1024)
        self.fc5, self.bn5 = Linear(1024,512),BatchNorm1D(512)
        self.fc6, self.bn6 = Linear(512,256), BatchNorm1D(256)
        self.fc7 = Linear(256,k*k)
    def __call__(self, x: Tensor) -> Tensor:
        x = self.bn1(self.fc1(x)).relu()
        x = self.bn2(self.fc2(x)).relu()
        x = self.bn3(self.fc3(x)).relu()
        x = self.bn4(self.fc4(x)).relu()
        x = x.max(1)
        x = self.bn5(self.fc5(x)).relu()
        x = self.bn6(self.fc6(x)).relu()
        return self.fc7(x) + self.id

class PointNet:
    def __init__(self, n_cls: int = 40):
        self.input_tnet = TNet(3)
        self.fc1, self.bn1 = Linear(3,64),   BatchNorm1D(64)
        self.feature_tnet  = TNet(64)
        self.fc2, self.bn2 = Linear(64,128), BatchNorm1D(128)
        self.fc3, self.bn3 = Linear(128,1024),BatchNorm1D(1024)
        self.h1,  self.hbn1 = Linear(1024,512),BatchNorm1D(512)
        self.h2,  self.hbn2 = Linear(512,256), BatchNorm1D(256)
        self.h3 = Linear(256,n_cls)
    def __call__(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        x = x @ self.input_tnet(x).reshape(B,3,3)
        x = self.bn1(self.fc1(x)).relu()
        x = x @ self.feature_tnet(x).reshape(B,64,64)
        x = self.bn2(self.fc2(x)).relu()
        x = self.bn3(self.fc3(x)).relu()
        x = x.max(1)
        x = self.hbn1(self.h1(x)).relu()
        x = self.hbn2(self.h2(x)).relu()
        return self.h3(x)


######## Dataset ########

def read_off(file):
    first = file.readline().lstrip("\ufeff").strip()
    if not first.upper().startswith("OFF"):
        raise ValueError("bad OFF header")
    header_rest = first[3:].strip()
    if header_rest:
        counts = header_rest
    else:
        counts = file.readline().strip()
    n_verts, n_faces, _ = map(int, counts.split())
    verts = [list(map(float, file.readline().split()))
             for _ in range(n_verts)]
    faces = [list(map(int,   file.readline().split()[1:4]))
             for _ in range(n_faces)]
    return verts, faces


class PointSampler:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, mesh):
        verts, faces = mesh
        verts  = np.asarray(verts, dtype=np.float32)
        faces  = np.asarray(faces, dtype=np.int64)

        # Vectorised cross-product area (faster and avoids NaN)
        tri   = verts[faces]       # (F,3,3)
        cross = np.cross(tri[:,1] - tri[:,0], tri[:,2] - tri[:,0])
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        good  = np.isfinite(areas) & (areas > 1e-9)
        if not good.any():
            probs = np.full(len(areas), 1/len(areas), dtype=np.float32)
        else:
            probs = np.where(good, areas, 0)
            probs = probs / probs.sum()

        face_idx = np.random.choice(len(faces), size=self.k, p=probs)
        tri_sel  = tri[face_idx]

        # Vectorised barycentric
        u, v = np.random.rand(self.k, 1), np.random.rand(self.k, 1)
        swap = u + v > 1
        u[swap], v[swap] = 1 - u[swap], 1 - v[swap]

        samples = (
            tri_sel[:, 0]
            + u * (tri_sel[:, 1] - tri_sel[:, 0])
            + v * (tri_sel[:, 2] - tri_sel[:, 0])
        ).astype(np.float32)

  
        return Tensor(samples)     # ready for tinygrad


def normalize(pc: Tensor) -> Tensor:
    pc = pc - pc.mean(axis=0, keepdim=True)
    scale = (pc*pc).sum(axis=1).sqrt().max()
    return pc / scale


class ModelNet40:
    def __init__(self, split="train", num_points=1024, root=None):
        root = root or download_modelnet40()
        self.sampler = PointSampler(num_points)
        classes = sorted(d.name for d in root.iterdir() if d.is_dir())
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        self.files   = [p for c in classes for p in (root/c/split).glob("*.off")]
        self.split = split
    def __len__(self): return len(self.files)
    def __getitem__(self, i: int):
        # Get output index
        path = self.files[i]
        y = self.cls2idx[path.parents[1].name]

        # print(f"Reading: {path}")

        # Read pointcloud
        with open(path, "r") as f:
            v,f_ = read_off(f)
        
        try:
            # Sample and noramlize
            pc = normalize(self.sampler((v,f_)))
            #show_pointcloud(pc)
        except Exception as e:
            print(f"Exception: {str(e)} on pointcloud {path}")

        # Apply random rotation
        if self.split == "train":
            theta = random.random()*2*math.pi
            R = Tensor([[math.cos(theta),-math.sin(theta),0],
                          [math.sin(theta), math.cos(theta),0],
                          [0,0,1]], dtype=dtypes.float32)
            
            # Add random noise
            pc = pc.matmul(R.T) + Tensor(np.random.normal(0,0.02,pc.shape).astype(np.float32))
        #show_pointcloud(pc)
        return pc, y

######## Train/Test ########
@Tensor.train()
@TinyJit
def _train_step(model, opt, xb, yb):
    opt.zero_grad()
    loss = model(xb).sparse_categorical_crossentropy(yb).mean().backward()
    return loss.realize(*opt.schedule_step())

@TinyJit
def _test_batch(model, xb):
    return model(xb).argmax(axis=1)



def train(epochs: int = 20, batch: int = 4, lr: float = 1e-3, seed: int = 0):
    random.seed(seed); np.random.seed(seed)

    train_ds, test_ds = ModelNet40("train"), ModelNet40("test")
    model = PointNet()
    opt   = Adam(get_parameters(model), lr=lr)

    history = {"loss": [], "acc": []}

    for ep in trange(1, epochs + 1, desc="epochs"):
        order = np.random.permutation(len(train_ds))

        # Train
        with Timing(f"epoch {ep} train"):
            pbar_train = tqdm(range(0, len(order), batch),
                              desc=f"train {ep:02}",
                              unit="batch", leave=False)

            for i in pbar_train:
                xb, yb = zip(*(train_ds[j] for j in order[i:i + batch]))
                loss = _train_step(
                    model, opt,
                    Tensor.stack(xb),
                    Tensor(np.array(yb, np.int32))
                )
                pbar_train.set_postfix(loss=float(loss.item()))

        history["loss"].append(float(loss.item()))

        # Eval
        correct = 0
        pbar_eval = tqdm(range(0, len(test_ds), batch),
                         desc=f"eval  {ep:02}",
                         unit="batch", leave=False)

        for i in pbar_eval:
            xb, yb = zip(*(test_ds[j] for j in range(i, min(i + batch, len(test_ds)))))
            preds = _test_batch(model, Tensor.stack(xb)).numpy()
            correct += int((preds == np.array(yb)).sum())
            pbar_eval.set_postfix(progress=f"{i + batch}/{len(test_ds)}")

        acc = correct / len(test_ds)
        history["acc"].append(acc)
        print(f"epoch {ep:02} | loss {history['loss'][-1]:.4f} | acc {acc:.3%}")

    return model, history

if __name__ == "__main__":
    train()
