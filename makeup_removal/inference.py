import os
import torch
from torchvision import transforms
from PIL import Image
from models.generator import UNetGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = UNetGenerator().to(device)
checkpoint_path = "checkpoints/generator_epoch100.pth" 
generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
generator.eval()

# --- Preprocess Function ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Postprocess Function ---
def denormalize(tensor):
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    return tensor

# --- Inference Function ---
def remove_makeup(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    output_image = denormalize(output_tensor.squeeze()).cpu()
    output_image_pil = transforms.ToPILImage()(output_image)
    output_image_pil.save(output_path)
    print(f"Saved makeup-free image to: {output_path}")

# --- Run ---
if __name__ == "__main__":
    input_img = "sample/0_0_makeup_applied_15_.png"       # <-- Your input image
    output_img = "sample/no_makeup.jpg"   # <-- Output path
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    remove_makeup(input_img, output_img)
