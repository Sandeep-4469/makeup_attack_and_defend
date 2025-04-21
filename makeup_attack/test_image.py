import torch
import os
from PIL import Image
import sys
import time
from models.cyclegan_beautygan_model import GeneratorResNet, get_norm_layer
from utils.image_processing import get_transform, deprocess_image_tensor
from utils.image_processing import save_pil_image
from torchvision.transforms import ToPILImage

class TestConfig:
    input_image_path = './data/non_makeup/0_0.jpg' 
    output_dir = './output_single_test_no_parser'
    checkpoint_path = './output_training/beautygan_makeup_scratch_run/checkpoints/15_net_G_A2B.pth'

    input_nc = 3
    output_nc = 3
    ngf = 64
    n_blocks = 6 
    norm = 'instance'
    no_dropout = False 

    image_size = 256
    load_size = 258

    # --- Run parameters ---
    gpu_id = '0' # '-1' for CPU

# ==================================================
#                  MAIN FUNCTION
# ==================================================
def main():
    args = TestConfig() # Use the config class

    # --- Validate Paths ---
    if not os.path.isfile(args.input_image_path): print(f"Error: Input image not found at '{args.input_image_path}'"); sys.exit(1)
    if not os.path.isfile(args.checkpoint_path): print(f"Error: Checkpoint file not found at '{args.checkpoint_path}'"); sys.exit(1)

    # --- Device Setup ---
    if args.gpu_id == '-1': device = torch.device('cpu'); print("Using CPU.")
    else:
        if torch.cuda.is_available():
            try:
                gpu_id = int(args.gpu_id.split(',')[0])
                device = torch.device(f'cuda:{gpu_id}')
                torch.cuda.set_device(device)
                print(f"Using GPU ID {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            except Exception as e:
                print(f"Warning: Error setting GPU '{args.gpu_id}' ({e}). Falling back."); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            print("Warning: CUDA not available, using CPU."); device = torch.device('cpu')

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # --- Load Model ---
    print(f"Loading Generator weights from: {args.checkpoint_path}")
    norm_layer = get_norm_layer(args.norm)
    # Make sure the parameters passed here match the TestConfig AND the actual trained model
    netG_A2B = GeneratorResNet(args.input_nc, args.output_nc, args.ngf, norm_layer=norm_layer, use_dropout=not args.no_dropout, n_blocks=args.n_blocks)

    try:
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'): # Handle DataParallel prefix
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
             state_dict = state_dict["state_dict"] # Handle weights saved within a dict

        netG_A2B.load_state_dict(state_dict, strict=True)
        netG_A2B.to(device)
        netG_A2B.eval()
        print("Generator model loaded successfully.")
    except Exception as e:
        print(f"\nERROR: Failed to load model weights: {e}")
        print(f"       Checkpoint path: {args.checkpoint_path}")
        print(f"       Model config used: ngf={args.ngf}, n_blocks={args.n_blocks}, norm='{args.norm}', no_dropout={args.no_dropout}")
        print("       Ensure the architecture parameters in TestConfig EXACTLY match the loaded checkpoint.")
        sys.exit(1)

    # --- Prepare Image Transform ---
    # For inference transform, set flip=False
    transform = get_transform(args.image_size, load_size=args.load_size, flip=False)

    # --- Load and Process Single Image ---
    print(f"Processing image: {args.input_image_path}")
    try:
        img_pil = Image.open(args.input_image_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading or transforming image: {e}"); sys.exit(1)

    # --- Perform Inference ---
    with torch.no_grad():
        start_time = time.time()
        try:
            fake_B_tensor = netG_A2B(img_tensor)
        except Exception as e:
             print(f"\nERROR during model forward pass: {e}")
             print(" Check tensor shapes and model compatibility.")
             sys.exit(1)
        end_time = time.time()
        print(f"Inference took {end_time - start_time:.4f} seconds")

    # --- Deprocess and Save Output ---
    try:
        output_tensor_deprocessed = deprocess_image_tensor(fake_B_tensor) # deprocess handles squeeze now
        output_pil = ToPILImage()(output_tensor_deprocessed.cpu())

        base_name = os.path.splitext(os.path.basename(args.input_image_path))[0]
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0].replace('latest_','').replace('net_G_A2B','')
        output_filename = f"{base_name}_makeup_applied_{ckpt_name}.png"
        output_path = os.path.join(args.output_dir, output_filename)

        # --- !! CORRECTED SAVE CALL !! ---
        save_pil_image(output_pil, output_path) # Use the correctly imported function
        print(f"Saved result to: {output_path}")

    except Exception as e:
        print(f"Error deprocessing or saving image: {e}")

if __name__ == "__main__":
    main()