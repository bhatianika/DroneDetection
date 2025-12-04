#!/usr/bin/env python3
# gradcam_from_crops.py
# Processes all crop_*.jpg in a folder and produces Grad-CAM overlays.
import os, argparse, time
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
from ultralytics.yolo.data.transforms import letterbox

# --- Hook and helpers (same pattern as batch script) ---
class Hook:
    def __init__(self, module):
        self.module = module
        self.activ = None
        self.grad = None
        self.fwd = module.register_forward_hook(self._f)
        self.bwd = module.register_full_backward_hook(self._b)
    def _f(self, module, inp, out): self.activ = out
    def _b(self, module, gi, go): self.grad = go[0]
    def close(self):
        try: self.fwd.remove()
        except: pass
        try: self.bwd.remove()
        except: pass

def find_last_conv(mod):
    last=None
    for m in mod.modules():
        if isinstance(m, torch.nn.Conv2d):
            last=m
    return last

def compute_cam(acts, grads):
    if acts.dim()==4: acts = acts[0]
    if grads.dim()==4: grads = grads[0]
    weights = grads.mean(dim=(1,2))
    cam = (weights[:,None,None] * acts).sum(dim=0)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    if cam.max()>0: cam = cam / cam.max()
    return cam.cpu().numpy()

# --- Main ---
def run(folder, weights, outdir, imgsz=320, device='cpu'):
    folder = Path(folder)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    crops = sorted(folder.glob('crop_*.jpg'))
    if not crops:
        print("No crop_*.jpg found in", folder); return

    dev = torch.device(device)
    model = YOLO(weights); model.to(dev); model.model.eval()
    last_conv = find_last_conv(model.model)
    if last_conv is None:
        raise SystemExit("No Conv2d found in model.")
    hook = Hook(last_conv)

    for i, crop_path in enumerate(crops, 1):
        try:
            img_bgr = cv2.imread(str(crop_path))
            if img_bgr is None:
                print("Could not read", crop_path); continue
            h,w = img_bgr.shape[:2]
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized, _, _ = letterbox(rgb, (imgsz,imgsz), auto=False, scaleup=False, stride=32)
            inp = torch.from_numpy(resized).permute(2,0,1).unsqueeze(0).to(dev).float()/255.0

            with torch.enable_grad():
                _ = model.model(inp)   # hook captures activations
                # get scalar score by running detect on resized crop
                preds = model.predict(source=resized, imgsz=imgsz, device=dev, conf=0.001, save=False)
                scalar = 0.0
                if len(preds) and len(preds[0].boxes):
                    scalar = float(preds[0].boxes.conf.cpu().numpy().max())
                # fallback to 0.01 if no detection
                scalar = max(scalar, 0.01)
                # backward via pseudo loss
                if hook.activ is None:
                    print("No activations for", crop_path); continue
                s = torch.tensor(scalar, device=dev, requires_grad=True)
                pseudo = (hook.activ.sum() * s)
                model.model.zero_grad(); pseudo.backward()

                if hook.grad is None:
                    print("No gradients for", crop_path); continue

                cam = compute_cam(hook.activ, hook.grad)
                cam_up = cv2.resize((cam*255).astype('uint8'), (w,h), interpolation=cv2.INTER_LINEAR)
                heat = cv2.applyColorMap(cam_up, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_bgr, 0.6, heat, 0.4, 0)
                out_path = outdir / f'gradcam_{crop_path.name}'
                cv2.imwrite(str(out_path), overlay)
                print(f"[{i}/{len(crops)}] saved {out_path} (score {scalar:.3f})")
        except Exception as e:
            print("Error processing", crop_path, e)
            continue
    hook.close()
    print("Done. Results in", outdir)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='./detection_logs', help='folder with crop_*.jpg')
    ap.add_argument('--weights', required=True, help='path to .pt weights')
    ap.add_argument('--out', default='./detection_logs/gradcam_out', help='where to save overlays')
    ap.add_argument('--imgsz', type=int, default=320, help='input size for model (smaller=faster)')
    ap.add_argument('--device', default='cpu', help='cpu or cuda')
    args = ap.parse_args()
    run(args.logs, args.weights, args.out, imgsz=args.imgsz, device=args.device)
