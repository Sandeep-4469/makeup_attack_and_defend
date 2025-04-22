import torch
import random
from torch.optim import lr_scheduler

class ImagePool():
    """Image buffer storing previous generated images to update discriminators."""
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0: return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

def get_scheduler(optimizer, args):
    """Return learning rate scheduler based on config class `args`."""
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            decay_start_epoch = args.n_epochs # Start decay after initial epochs
            lr_l = 1.0 - max(0, epoch + args.epoch_count - decay_start_epoch) / float(args.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    # Add 'plateau' and 'cosine' if needed, checking how args are structured
    elif args.lr_policy == 'plateau':
         print("Warning: Plateau scheduler requires validation metric for step(). Not implemented here.")
         # Placeholder - steps without metric
         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'cosine':
         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs + args.n_epochs_decay, eta_min=0)
    else:
        raise NotImplementedError(f'learning rate policy [{args.lr_policy}] is not implemented')
    return scheduler

def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad=False for networks to avoid unnecessary computations."""
    if not isinstance(nets, list): nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad