import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.utils as vutils
try:
    from torchvision.transforms.functional import InterpolationMode
    DEFAULT_INTERPOLATION = InterpolationMode.BICUBIC
except ImportError:
    try:
        DEFAULT_INTERPOLATION = Image.BICUBIC
    except AttributeError:
        DEFAULT_INTERPOLATION = 3

IMG_MEAN = [0.5, 0.5, 0.5]
IMG_STD = [0.5, 0.5, 0.5]

def get_transform(image_size, load_size=None, method=DEFAULT_INTERPOLATION, convert=True, flip=True):
    """Returns image preprocessing transform."""
    transform_list = []
    if load_size is None: load_size = image_size

    current_interpolation = method
    if isinstance(method, int) and not hasattr(Image, 'Resampling'):
         pass 
    elif hasattr(Image, 'Resampling'):
         try:
              current_interpolation = InterpolationMode.BICUBIC 
         except NameError:
              current_interpolation = 3 
    else:
         current_interpolation = Image.BICUBIC


    transform_list.append(transforms.Resize(load_size, interpolation=current_interpolation))
    transform_list.append(transforms.CenterCrop(image_size) if not flip else transforms.RandomCrop(image_size))
    if flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if convert:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)]
    return transforms.Compose(transform_list)

def deprocess_image_tensor(tensor):
    """Converts a batch or single tensor from [-1, 1] to [0, 1]."""
    if tensor is None: return None
    if tensor.dim() == 4 and tensor.shape[0] == 1: # If batch dim exists and size is 1, remove it
        tensor = tensor.squeeze(0)
    elif tensor.dim() != 3:
         print(f"Warning: Unexpected tensor dimension {tensor.dim()} in deprocess_image_tensor. Expected 3 or 4 (with batch 1).")
         # Attempt to handle common cases or return None
         if tensor.dim() == 4 and tensor.shape[0] > 1:
             print(" Processing only the first image in the batch.")
             tensor = tensor[0]
         # Cannot handle other dims safely here
         # return None

    # Clamp just in case model output slightly exceeds range
    tensor = tensor.clamp_(-1.0, 1.0)
    # Map [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    return tensor

def save_image_grid(tensor, filename, nrow=8):
    """Saves a batch of image tensors (range [0, 1]) as a grid."""
    if tensor is None: return
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Expects tensor in range [0, 1]
        vutils.save_image(tensor, filename, nrow=nrow, padding=2, normalize=False)
    except Exception as e:
        print(f"Error saving image grid to {filename}: {e}")

# --- Function to save a single PIL Image ---
def save_pil_image(pil_image, save_path):
    """Saves a single PIL image object to the specified path."""
    if pil_image is None: return
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the image
        pil_image.save(save_path)
        # print(f"Saved single image: {save_path}") # Optional log
    except Exception as e:
        print(f"Error saving PIL image to {save_path}: {e}")