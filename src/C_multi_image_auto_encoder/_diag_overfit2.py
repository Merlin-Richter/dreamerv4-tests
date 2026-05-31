"""Test what makes the encoder actually encode input: temporal context (L) and MAE masking."""
import sys
from pathlib import Path
import numpy as np, torch, cv2
_SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(_SRC))
from video_auto_encoder import AutoEncoder, AutoEncoderConfig
device="cuda"
raw=np.load(_SRC.parent.parent/"occluded.npy",mmap_mode="r"); act=np.load(_SRC.parent.parent/"occluded_actions.npy",mmap_mode="r")
rng=np.random.default_rng(2)
def grab(n):
    fr=[]
    while len(fr)<n:
        ep=rng.integers(raw.shape[0]); s=rng.integers(0,raw.shape[1]-16)
        a=np.asarray(act[ep,s:s+16]); idx=np.where(a==0)[0]
        if len(idx): fr.append(np.asarray(raw[ep,s+idx[0]]))
    return torch.from_numpy(np.stack(fr).astype(np.float32)/255.).to(device)
def bgr(img):
    u8=(img.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8); return cv2.cvtColor(u8,cv2.COLOR_RGB2BGR)
def run(name,L,mask):
    torch.manual_seed(0)
    cfg=AutoEncoderConfig(dtype=torch.float32,img_input_H=64,img_input_W=64,max_temporal_length=L,
                          mae_min_mask=0.0,mae_max_mask=mask,drop_rate=0.0,att_drop_rate=0.0)
    m=AutoEncoder(cfg).to(device)
    x=grab(L).unsqueeze(0)  # (1,L,64,64,3): one clip of L distinct frames
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    m.train()
    for _ in range(1000):
        loss=((m(x)-x)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): pred=m(x).float()
    emse=((pred-x)**2).mean().item()
    z=m.encoder(x).float().reshape(L,-1); zc=z/(z.norm(dim=1,keepdim=True)+1e-6)
    cos=(zc@zc.T)[~torch.eye(L,dtype=bool,device=device)].mean().item() if L>1 else 1.0
    print(f"{name:20s} | eval MSE {emse:.5f} | pred std/img {pred.reshape(L,-1).std(1).mean():.4f} | latent cos {cos:.3f}")
    k=min(L,6); mont=np.hstack([np.vstack([bgr(x[0,b]),bgr(pred[0,b])]) for b in range(k)])
    cv2.imwrite(str(_SRC/f"_ovf2_{name}.png"),cv2.resize(mont,(k*64*3,2*64*3),interpolation=cv2.INTER_NEAREST))
print("Overfit one clip, 1000 steps, lr 1e-3\n")
run("L1_nomask",1,0.0)
run("L8_nomask",8,0.0)
run("L1_mask0.75",1,0.75)
run("L8_mask0.75",8,0.75)
