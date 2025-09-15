from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


model = YOLO('training/runs/detect/train/weights/best.pt').to(device)  # load a pretrained model (recommended for training)

results = model.predict('08fd33_4.mp4',save = True, device=device)

print(results[0])

print("------------------------------------")

for box in results[0].boxes:
    print(box)
