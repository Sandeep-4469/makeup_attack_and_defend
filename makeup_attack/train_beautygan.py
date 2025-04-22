# train_beautygan.py (No ArgParse Version - Corrected Import)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import itertools
from tqdm import tqdm
import time
import datetime
import sys

try:
    from models.cyclegan_beautygan_model import GeneratorResNet, NLayerDiscriminator, GANLoss, init_weights, get_norm_layer
except ImportError as e:
    print(f"ERROR importing from models: {e}")
    print("Make sure 'models/cyclegan_beautygan_model.py' exists and defines GeneratorResNet, NLayerDiscriminator, GANLoss, init_weights, and get_norm_layer.")
    sys.exit(1)
try:
    from utils.datasets import UnpairedImageDataset, collate_fn_skip_none
    from utils.image_processing import get_transform, save_image_grid
    from utils.helpers import ImagePool, get_scheduler, set_requires_grad
except ImportError as e:
    print(f"ERROR importing from utils: {e}")
    print("Make sure files in the 'utils' directory exist.")
    sys.exit(1)
class TrainConfig:
    dataroot_A = './data/non_makeup'  
    dataroot_B = './data/makeup'    
    name = 'beautygan_makeup_scratch_run'

    checkpoints_dir = './output_training' 
    samples_dir = './output_training'    

    input_nc = 3       # Input image channels
    output_nc = 3      # Output image channels
    ngf = 64           # Gen filters
    ndf = 64           # Disc filters
    n_blocks = 6      # Gen ResNet blocks (6 or 9)
    norm = 'instance'  # Normalization: 'instance', 'batch', 'none'
    init_type = 'normal' # Weight init
    init_gain = 0.02   # Weight init gain
    no_dropout = False # Use dropout in Gen ResBlocks?

    # --- Dataset parameters ---
    max_dataset_size = float("inf") # Max images per domain (inf for all)
    image_size = 256   # Final image size
    load_size = 286    # Load size before cropping
    batch_size = 1     # Batch size (CycleGAN often uses 1)
    num_threads = 2    # Dataloader workers (adjust based on CPU cores)

    # --- Training parameters ---
    gpu_ids = '0'      # GPU IDs ('0', '0,1', '-1' for CPU)
    n_epochs = 12   # Epochs with initial LR
    n_epochs_decay = 3 # Epochs to linearly decay LR
    epoch_count = 1    # Starting epoch number (usually 1, change if resuming)
    beta1 = 0.5        # Adam beta1
    lr = 0.0002        # Initial learning rate
    lr_policy = 'linear' # LR policy: 'linear', 'step', 'cosine'
    lr_decay_iters = 50 # Iters for 'step' policy
    pool_size = 50     # Image pool size

    # --- Loss weights ---
    lambda_A = 10.0    # Cycle loss A->B->A weight
    lambda_B = 10.0    # Cycle loss B->A->B weight
    lambda_identity = 0.5 # Identity loss weight (0 to disable)

    # --- Display and Save Frequency ---
    print_freq = 100       # Print loss every N iterations
    save_latest_freq = 5000 # Save latest ckpt every N iterations
    save_epoch_freq = 4    # Save ckpt every N epochs
    display_freq = 200     # Save sample images every N iterations
    continue_train = False # Resume from latest checkpoint?

# ==================================================
#                  MAIN FUNCTION
# ==================================================
def main():
    args = TrainConfig() # Use the config class

    # --- Validate Paths ---
    if not os.path.isdir(args.dataroot_A): print(f"Error: dataroot_A not found at '{args.dataroot_A}'"); sys.exit(1)
    if not os.path.isdir(args.dataroot_B): print(f"Error: dataroot_B not found at '{args.dataroot_B}'"); sys.exit(1)

    # --- Device Setup ---
    if args.gpu_ids == '-1': device = torch.device('cpu'); print("Using CPU.")
    else:
        if torch.cuda.is_available():
            try:
                gpu_id_list = [int(g) for g in args.gpu_ids.split(',')]
                gpu_id = gpu_id_list[0] # Use the first GPU specified
                device = torch.device(f'cuda:{gpu_id}')
                torch.cuda.set_device(device)
                print(f"Using GPU ID {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            except (ValueError, IndexError, RuntimeError) as e:
                print(f"Warning: Error setting GPU '{args.gpu_ids}' ({e}). Falling back.")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            print("Warning: CUDA not available, using CPU.")
            device = torch.device('cpu')

    # --- Directories ---
    expr_dir = os.path.join(args.checkpoints_dir, args.name)
    samples_dir = os.path.join(args.samples_dir, args.name, 'samples')
    checkpoints_save_dir = os.path.join(expr_dir, 'checkpoints')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_save_dir, exist_ok=True)
    print(f"Experiment Output Dir: {expr_dir}")

    # --- Dataset and DataLoader ---
    transform = get_transform(args.image_size, args.load_size, flip=True)
    try:
        dataset = UnpairedImageDataset(args.dataroot_A, args.dataroot_B, transform=transform, max_dataset_size=args.max_dataset_size)
    except Exception as e:
        print(f"Error creating dataset: {e}"); sys.exit(1)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=int(args.num_threads), pin_memory=True, collate_fn=collate_fn_skip_none)
    dataset_size = len(dataset)
    if dataset_size == 0: print("Error: Dataset is empty after creation."); sys.exit(1)
    print(f'Training images = {dataset_size}')

    # --- Models ---
    # !! This line uses the imported get_norm_layer !!
    norm_layer = get_norm_layer(args.norm)
    netG_A2B = GeneratorResNet(args.input_nc, args.output_nc, args.ngf, norm_layer=norm_layer, use_dropout=not args.no_dropout, n_blocks=args.n_blocks)
    netG_B2A = GeneratorResNet(args.output_nc, args.input_nc, args.ngf, norm_layer=norm_layer, use_dropout=not args.no_dropout, n_blocks=args.n_blocks)
    netD_A = NLayerDiscriminator(args.input_nc, args.ndf, n_layers=3, norm_layer=norm_layer)
    netD_B = NLayerDiscriminator(args.output_nc, args.ndf, n_layers=3, norm_layer=norm_layer)

    # --- Initialization or Loading ---
    start_epoch = args.epoch_count
    if args.continue_train:
        print("Attempting to resume training from latest checkpoint...")
        ckpt_path_G_A2B = os.path.join(checkpoints_save_dir, 'latest_net_G_A2B.pth')
        ckpt_path_G_B2A = os.path.join(checkpoints_save_dir, 'latest_net_G_B2A.pth')
        ckpt_path_D_A = os.path.join(checkpoints_save_dir, 'latest_net_D_A.pth')
        ckpt_path_D_B = os.path.join(checkpoints_save_dir, 'latest_net_D_B.pth')
        if all(os.path.isfile(p) for p in [ckpt_path_G_A2B, ckpt_path_G_B2A, ckpt_path_D_A, ckpt_path_D_B]):
            try:
                netG_A2B.load_state_dict(torch.load(ckpt_path_G_A2B, map_location=device))
                netG_B2A.load_state_dict(torch.load(ckpt_path_G_B2A, map_location=device))
                netD_A.load_state_dict(torch.load(ckpt_path_D_A, map_location=device))
                netD_B.load_state_dict(torch.load(ckpt_path_D_B, map_location=device))
                print("Successfully loaded latest model checkpoints.")
                # Consider loading optimizers and epoch count here for full resume
            except Exception as e:
                print(f"Warning: Error loading checkpoints ({e}). Re-initializing.")
                init_weights(netG_A2B, args.init_type, args.init_gain)
                init_weights(netG_B2A, args.init_type, args.init_gain)
                init_weights(netD_A, args.init_type, args.init_gain)
                init_weights(netD_B, args.init_type, args.init_gain)
        else:
            print("Latest checkpoint files not found. Initializing from scratch.")
            init_weights(netG_A2B, args.init_type, args.init_gain)
            init_weights(netG_B2A, args.init_type, args.init_gain)
            init_weights(netD_A, args.init_type, args.init_gain)
            init_weights(netD_B, args.init_type, args.init_gain)
    else:
        print("Initializing models from scratch...")
        init_weights(netG_A2B, args.init_type, args.init_gain)
        init_weights(netG_B2A, args.init_type, args.init_gain)
        init_weights(netD_A, args.init_type, args.init_gain)
        init_weights(netD_B, args.init_type, args.init_gain)

    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)

    # --- Image Pools ---
    fake_A_pool = ImagePool(args.pool_size)
    fake_B_pool = ImagePool(args.pool_size)

    # --- Loss Functions ---
    criterionGAN = GANLoss().to(device)
    criterionCycle = nn.L1Loss()
    criterionIdt = nn.L1Loss()

    # --- Optimizers ---
    optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
    optimizers = [optimizer_G, optimizer_D]

    # --- Schedulers ---
    schedulers = [get_scheduler(optimizer, args) for optimizer in optimizers]

    # --- Training Loop ---
    total_iters = 0
    print(f"Starting Training Loop from epoch {start_epoch}...")
    overall_start_time = time.time()

    for epoch in range(start_epoch, args.n_epochs + args.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{args.n_epochs + args.n_epochs_decay}")

        netG_A2B.train(); netG_B2A.train(); netD_A.train(); netD_B.train() # Set train mode

        for i, data in progress_bar:
            if data is None: continue # Skip if batch failed collation

            iter_start_time = time.time()
            total_iters += args.batch_size
            epoch_iter += args.batch_size

            real_A = data['A'].to(device, non_blocking=True)
            real_B = data['B'].to(device, non_blocking=True)

            # --- Train Generators ---
            set_requires_grad([netD_A, netD_B], False)
            optimizer_G.zero_grad()

            fake_B = netG_A2B(real_A); rec_A = netG_B2A(fake_B) # A -> B -> A
            fake_A = netG_B2A(real_B); rec_B = netG_A2B(fake_A) # B -> A -> B

            loss_G_A2B = criterionGAN(netD_B(fake_B), True)
            loss_G_B2A = criterionGAN(netD_A(fake_A), True)
            loss_cycle_A = criterionCycle(rec_A, real_A) * args.lambda_A
            loss_cycle_B = criterionCycle(rec_B, real_B) * args.lambda_B

            loss_idt_A = 0; loss_idt_B = 0
            if args.lambda_identity > 0:
                idt_A = netG_B2A(real_A); loss_idt_A = criterionIdt(idt_A, real_A) * args.lambda_A * args.lambda_identity
                idt_B = netG_A2B(real_B); loss_idt_B = criterionIdt(idt_B, real_B) * args.lambda_B * args.lambda_identity

            loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            loss_G.backward()
            optimizer_G.step()

            # --- Train Discriminators ---
            set_requires_grad([netD_A, netD_B], True)
            optimizer_D.zero_grad()

            loss_D_A_real = criterionGAN(netD_A(real_A), True)
            fake_A_pooled = fake_A_pool.query(fake_A)
            loss_D_A_fake = criterionGAN(netD_A(fake_A_pooled.detach()), False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            loss_D_A.backward()

            loss_D_B_real = criterionGAN(netD_B(real_B), True)
            fake_B_pooled = fake_B_pool.query(fake_B)
            loss_D_B_fake = criterionGAN(netD_B(fake_B_pooled.detach()), False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            loss_D_B.backward()

            optimizer_D.step()

            # --- Logging ---
            if total_iters % args.print_freq == 0:
                losses = {
                    'G_GAN': (loss_G_A2B.item() + loss_G_B2A.item()),
                    'G_Cycle': (loss_cycle_A.item() + loss_cycle_B.item()),
                    'G_Idt': (loss_idt_A.item() + loss_idt_B.item()) if args.lambda_identity > 0 else 0,
                    'D_A': loss_D_A.item(), 'D_B': loss_D_B.item()
                }
                iter_time = time.time() - iter_start_time
                avg_iter_time = (time.time() - epoch_start_time) / (i + 1) if i > 0 else iter_time
                eta_seconds = avg_iter_time * (len(dataloader) - i - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                loss_log = " | ".join([f"{k}: {v:.3f}" for k, v in losses.items()])
                progress_bar.set_postfix_str(f"{loss_log} | IterTime: {iter_time:.3f}s | ETA: {eta_str}")

            # --- Save Sample Images ---
            if total_iters % args.display_freq == 0:
                 with torch.no_grad():
                    # Use current batch images for visualization
                    vis_real_A = real_A
                    vis_real_B = real_B
                    vis_fake_B = netG_A2B(vis_real_A)
                    vis_rec_A = netG_B2A(vis_fake_B)
                    vis_fake_A = netG_B2A(vis_real_B)
                    vis_rec_B = netG_A2B(vis_fake_A)
                    img_sample = torch.cat((vis_real_A.data, vis_fake_B.data, vis_rec_A.data,
                                            vis_real_B.data, vis_fake_A.data, vis_rec_B.data), 0)
                    save_path = os.path.join(samples_dir, f'epoch_{epoch:03d}_iter_{total_iters:07d}.png')
                    # Adjust nrow if batch size is > 1
                    save_image_grid(img_sample, save_path, nrow=args.batch_size * 2 if args.batch_size > 0 else 2)


            if total_iters % args.save_latest_freq == 0:
                print(f'\nSaving latest model (epoch {epoch}, total_iters {total_iters})')
                torch.save(netG_A2B.state_dict(), os.path.join(checkpoints_save_dir, 'latest_net_G_A2B.pth'))
                torch.save(netG_B2A.state_dict(), os.path.join(checkpoints_save_dir, 'latest_net_G_B2A.pth'))
                torch.save(netD_A.state_dict(), os.path.join(checkpoints_save_dir, 'latest_net_D_A.pth'))
                torch.save(netD_B.state_dict(), os.path.join(checkpoints_save_dir, 'latest_net_D_B.pth'))

        epoch_duration = time.time() - epoch_start_time
        print(f'\nEpoch {epoch}/{args.n_epochs + args.n_epochs_decay} Finished \t Time Taken: {epoch_duration:.2f} sec')

        print('Updating learning rates...')
        current_lr_before = optimizer_G.param_groups[0]['lr']
        for scheduler in schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): pass
            else: scheduler.step()
        current_lr_after = optimizer_G.param_groups[0]['lr']
        print(f'Learning rate updated from {current_lr_before:.7f} to {current_lr_after:.7f}')

        if epoch % args.save_epoch_freq == 0 or epoch == args.n_epochs + args.n_epochs_decay:
            if epoch > 0 :
                print(f'Saving epoch checkpoint: {epoch}')
                torch.save(netG_A2B.state_dict(), os.path.join(checkpoints_save_dir, f'{epoch}_net_G_A2B.pth'))
                torch.save(netG_B2A.state_dict(), os.path.join(checkpoints_save_dir, f'{epoch}_net_G_B2A.pth'))
                torch.save(netD_A.state_dict(), os.path.join(checkpoints_save_dir, f'{epoch}_net_D_A.pth'))
                torch.save(netD_B.state_dict(), os.path.join(checkpoints_save_dir, f'{epoch}_net_D_B.pth'))

    total_training_time = time.time() - overall_start_time
    print(f"Training finished. Total time: {str(datetime.timedelta(seconds=int(total_training_time)))}")

if __name__ == "__main__":
    main()