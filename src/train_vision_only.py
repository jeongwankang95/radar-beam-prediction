#!/usr/bin/env python3
"""Train the vision-only student on the pass-level split, with the protocol of the radar
students (focal loss, Adam + cosine warm restarts, validation-ADBA model selection, up to 70
epochs, patience 20), then evaluate it on the test set. Self-contained (does not import
train_cmkd). Usage: python src/train_vision_only.py SEED [DATA_ROOT TAG TRAIN_CSV WORKERS [gap]];
writes cmkd_results/<TAG>_s<SEED>/; 'gap' selects the pooled encoder of matched size."""
import torch, numpy as np, os, sys, glob
import torchvision.transforms as transf
from torch.utils.data import DataLoader
from MyFunc import set_seed, FocalLoss, calculate_topk_accuracy, calculate_dba_score
from DataFeed import DataFeed
from model_vision_only import VisionOnlyStudent

SEED=int(sys.argv[1]) if len(sys.argv)>1 else 42
ROOT=sys.argv[2] if len(sys.argv)>2 else 'dataset/scenario9'      # e.g. dataset/scenario32
TAG=sys.argv[3] if len(sys.argv)>3 else '_v3_visiononly'
TRAIN_CSV=sys.argv[4] if len(sys.argv)>4 else 'train_seqs_RA_tr.csv'
NW=int(sys.argv[5]) if len(sys.argv)>5 else 0
GAP=(len(sys.argv)>6 and sys.argv[6]=='gap')                          # parameter-matched encoder
set_seed(SEED); EP=int(os.environ.get('VIS_EPOCHS','70')); PAT=20; NP=7
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE=f'cmkd_results/{TAG}_s{SEED}'; os.makedirs(SAVE,exist_ok=True)
proc=transf.Compose([transf.ToPILImage(),transf.Resize((224,224))])
def dl(csv): return DataLoader(DataFeed(ROOT,f'{ROOT}/{csv}',seq_len=8,transform=proc,clipped_range=128),batch_size=32,shuffle=('tr' in csv),num_workers=NW,persistent_workers=NW>0)
tr,va,te=dl(TRAIN_CSV),dl('val_seqs_RA.csv'),dl('test_seqs_RA.csv')
print(f"vision-only seed={SEED}  Train {len(tr.dataset)} Val {len(va.dataset)} Test {len(te.dataset)}")

def img_batch(img):
    B=img.shape[0]; im=img.unsqueeze(2)                              # (B,7,1,224,224)
    pad=torch.zeros(B,NP,1,224,224)
    return torch.cat([im,pad],1)                                    # (B,14,1,224,224)
def targets_of(beam,label): return torch.cat([beam[:,-1:],label[:,:6]],-1)

m=VisionOnlyStudent(64,64,64,gru_layers=1,num_heads_mha=8,gap=GAP).to(DEV)
print(f"vision-only params: {sum(p.numel() for p in m.parameters()):,}  gap={GAP}")
crit=FocalLoss(alpha=1,gamma=2)
opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=2,eta_min=1e-6)

@torch.no_grad()
def validate(loader):
    m.eval(); outs,labs=[],[]
    for img,ra,da,beam,label in loader:
        lg=m(img_batch(img).to(DEV),NP)[0][:,-NP:,:]; outs.append(lg.cpu()); labs.append(targets_of(beam,label))
    O=torch.cat(outs,0); L=torch.cat(labs,0)
    return float(np.mean(calculate_dba_score(O,L,delta=5))), O, L

best=-1.0; noimp=0
for ep in range(EP):
    m.train(); rl=0.
    for i,(img,ra,da,beam,label) in enumerate(tr):
        ib=img_batch(img).to(DEV); tg=targets_of(beam,label).to(DEV)
        opt.zero_grad(); lg=m(ib,NP)[0][:,-NP:,:]
        loss=crit(lg.reshape(-1,64),tg.flatten()); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),10); opt.step(); rl=(loss.item()+i*rl)/(i+1)
    sched.step()
    adba,_,_=validate(va)
    print(f"  ep{ep:02d} loss {rl:.4f}  val ADBA {100*adba:.2f}% (best {100*best:.2f}%)",flush=True)
    if adba>best+1e-5:
        best=adba; torch.save(m.state_dict(),f'{SAVE}/vision_best.pth'); noimp=0
    else:
        noimp+=1
        if noimp>=PAT: print(f"  early stop ep{ep}"); break

m.load_state_dict(torch.load(f'{SAVE}/vision_best.pth',map_location=DEV))
_,O,L=validate(te)
topk,_=calculate_topk_accuracy(O,L); dba=calculate_dba_score(O,L,delta=5)
res={f'top{k}':np.array(topk[k]) for k in topk}; res['dba']=dba
np.savez(f'{SAVE}/eval_results.npz',**res)
print("\n=== Vision-only TEST ===")
for k in [1,3,5]: print(f"  ATop-{k}: {100*np.mean(topk[k]):.2f}%")
print(f"  ADBA:    {100*np.mean(dba):.2f}%")
