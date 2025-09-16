# train.py
import os, math, time, glob, argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision import datasets

from ddpm import (
    DiffusionConfig, make_beta_schedule, precompute_alphas,
    sample_timesteps, diffusion_loss, UNetTiny,
    sample_loop, denorm_to_01
)

# ---------------------------
# EMA (Exponential Moving Avg)
# ---------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


def make_loader(batch_size=64, img_size=64, num_workers=2, pin_memory=True):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = datasets.ImageFolder(root="./data/celeba", transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory)


def save_samples(model, cfg, alphas, out_path, n=16, size=32, channels=3):
    model.eval()
    with torch.no_grad():
        x = sample_loop(model, cfg, alphas, batch_size=n, img_size=size, channels=channels)  # [-1,1]
        x01 = denorm_to_01(x)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_image(x01, out_path, nrow=int(math.sqrt(n)))


def latest_ckpt(path_pattern: str):
    files = sorted(glob.glob(path_pattern))
    return files[-1] if files else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--save_every", type=int, default=1000, help="save checkpoint every N steps")
    p.add_argument("--sample_every", type=int, default=2000, help="save sample grid every N steps")
    p.add_argument("--resume", type=str, default="", help="path to checkpoint .pt to resume from")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    
    

    
    args = p.parse_args()

    cfg = DiffusionConfig(T=args.timesteps)
    device = cfg.device
    print("Device:", device)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # speedup for fixed-size inputs

    # diffusion constants
    betas = make_beta_schedule(cfg)
    alphas = precompute_alphas(betas)

    # model / opt / scaler / ema
    model = UNetTiny(in_ch=3, base=32, time_dim=128).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available() and not args.no_amp))
    ema = EMA(model, decay=args.ema_decay)

    # data
    pin_mem = torch.cuda.is_available()
    loader = make_loader(batch_size=args.batch_size, img_size=args.img_size,
                         num_workers=args.workers, pin_memory=pin_mem)

    # state
    start_epoch = 0
    global_step = 0

    # resume logic
    if args.resume:
        ckpt_path = args.resume
    else:
        ckpt_path = latest_ckpt("checkpoints/step_*.pt")

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if "scaler" in state and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])
        if "ema" in state and state["ema"] is not None:
            ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
        start_epoch = state.get("epoch", 0)
        global_step = state.get("global_step", 0)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # training
    model.train()
    for epoch in range(start_epoch, args.epochs):
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            t = sample_timesteps(x.size(0), cfg.T, device)

            with torch.cuda.amp.autocast(enabled=(torch.cuda.is_available() and not args.no_amp)):
                loss = diffusion_loss(model, x, t, alphas)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # EMA after optimizer step
            ema.update(model)

            global_step += 1
            if global_step % 100 == 0:
                print(f"epoch {epoch:03d} step {global_step:07d} | loss {loss.item():.4f}")

            if global_step % args.save_every == 0:
                # save full training state
                state = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict() if torch.cuda.is_available() and not args.no_amp else None,
                    "ema": ema.shadow,
                    "epoch": epoch,
                    "global_step": global_step,
                    "cfg_T": cfg.T,
                }
                path = f"checkpoints/step_{global_step:07d}.pt"
                torch.save(state, path)
                print(f"[ckpt] saved {path}")

            if global_step % args.sample_every == 0:
                # sample with EMA weights for nicer images
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
                out_path = f"samples/step_{global_step:07d}.png"
                save_samples(model, cfg, alphas, out_path, n=16, size=args.img_size, channels=3)
                print(f"[sample] wrote {out_path}")
                # restore training weights
                model.load_state_dict(backup)

        # end epoch checkpoint (optional)
        if (global_step % args.save_every) != 0:
            state = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if torch.cuda.is_available() and not args.no_amp else None,
                "ema": ema.shadow,
                "epoch": epoch + 1,
                "global_step": global_step,
                "cfg_T": cfg.T,
            }
            path = f"checkpoints/epoch_{epoch+1:03d}_step_{global_step:07d}.pt"
            torch.save(state, path)
            print(f"[ckpt] saved {path}")

    print("Training done.")


if __name__ == "__main__":
    main()
