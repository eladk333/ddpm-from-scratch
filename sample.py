# sample.py
import os, glob, math, argparse, torch
from torchvision.utils import save_image
from ddpm import (
    DiffusionConfig, make_beta_schedule, precompute_alphas,
    UNetTiny, sample_loop, denorm_to_01
)

def latest_ckpt():
    files = sorted(glob.glob("checkpoints/*.pt"))
    return files[-1] if files else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="", help="path to .pt (full-state or raw weights)")
    p.add_argument("--n", type=int, default=16, help="number of images")
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--use_ema", action="store_true", help="use EMA weights if checkpoint has them")
    args = p.parse_args()

    ckpt_path = args.ckpt or latest_ckpt()
    assert ckpt_path and os.path.exists(ckpt_path), "No checkpoint found. Train first."

    # device + (maybe) T from ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt_path, map_location=device)
    T = state.get("cfg_T", 1000) if isinstance(state, dict) else 1000
    cfg = DiffusionConfig(T=T, device=device)

    # diffusion constants
    betas = make_beta_schedule(cfg)
    alphas = precompute_alphas(betas)

    # build model (must match training architecture)
    model = UNetTiny(in_ch=3, base=64, time_dim=128).to(device)

    # load weights (handles both full training state or raw state_dict)
    if isinstance(state, dict) and "model" in state:
        if args.use_ema and ("ema" in state) and state["ema"]:
            print(f"Loading EMA weights from {ckpt_path}")
            model.load_state_dict(state["ema"], strict=True)
        else:
            print(f"Loading model weights from {ckpt_path}")
            model.load_state_dict(state["model"], strict=True)
    else:
        print(f"Loading raw state_dict from {ckpt_path}")
        model.load_state_dict(state, strict=True)

    # sample a batch
    x = sample_loop(model, cfg, alphas, batch_size=args.n, img_size=args.img_size, channels=3)  # [-1,1]
    x01 = denorm_to_01(x)  # [0,1]

    os.makedirs("samples", exist_ok=True)
    out_path = f"samples/grid.png"
    save_image(x01, out_path, nrow=int(math.sqrt(args.n)))
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
