import torch
import os
from model import tensorial
import json
import numpy as np
from imageio.v2 import imread, imwrite
import argparse

class Trainer():
    def __init__(self, config):
        self.config = config
        self.experiment_config = config['Experiment']
        self.model_config = config['TensorBase']
        self.training_config = config['Training']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_dataset()
        
        self.model_config['aabb'] = torch.tensor(self.train_data['aabb'])
        self.model_config['near_far'] = [self.train_data['near'], self.train_data['far']]
        
        self.model_config['gridSize'] = self.calc_gridsize(
            self.training_config['N_voxel_init'],
            self.model_config['aabb']
        )
        
        self.model_config['alphaMask'] = None
        self.tensorbase = tensorial.TensorBase(self.model_config)

        self.optim = torch.optim.Adam(
            self.tensorbase.get_optparam_groups(
                self.training_config['lr_init'], self.training_config['lr_basis']
            ), 
            betas = (0.9, 0.99)
        )

        if self.training_config['lr_decay_iters'] > 0:
            self.lr_factor = self.training_config['lr_decay_target_ratio'] ** (1 / self.training_config['lr_decay_iters'])
        else:
            self.training_config['lr_decay_iters'] = self.training_config['n_iters']
            self.lr_factor = self.training_config['lr_decay_target_ratio'] ** (1 / self.training_config['n_iters'])

        self.N_voxel_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(self.training_config['N_voxel_init']), 
                        np.log(self.training_config['N_voxel_final']), 
                        len(self.training_config['upsamp_list']) + 1
                    )
                )
            ).long()
        ).tolist()[1:]

        self.load_training_rays()
        self.load_eval_rays()
        
        return
    
    def load_training_rays(self):

        self.all_rays = torch.zeros(0, 6)
        self.all_rgb = torch.zeros(0, 3)
        self.frame_ray_correspondence = torch.zeros(0, 1)

        for idx in range(len(self.train_data['frames'])):

            fx, fy = float(self.train_data['K'][0][0]), float(self.train_data['K'][1][1])
            
            image = torch.from_numpy(np.float32(imread(os.path.join(self.experiment_config['dataset_path'], self.train_data['frames'][idx]['image'])))) / 255.
            height, width = int(image.shape[0]), int(image.shape[1])
            c2w = torch.from_numpy(np.float32(self.train_data['frames'][idx]['transform_matrix']))

            i, j = torch.meshgrid(
                    torch.arange(width, dtype=torch.float32),
                    torch.arange(height, dtype=torch.float32),
                    indexing = 'ij'
                )
        
            i, j = i.transpose(-1, -2), j.transpose(-1, -2)

            directions = torch.stack(
                [
                    (i - width * 0.5) / fx, -(j - height * 0.5) / fy, -torch.ones_like(i)
                ], 
                dim = -1
            )
            
            rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim = -1)
            rays_o = c2w[:3, -1].expand(rays_d.shape)

            rays_o, rays_d = rays_o.flatten(0, 1), rays_d.flatten(0, 1)
            rgb = image.flatten(0, 1)

            vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
            rate_a = (self.model_config['aabb'][1] - rays_o) / vec
            rate_b = (self.model_config['aabb'][0] - rays_o) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1)
            t_max = torch.maximum(rate_a, rate_b).amin(-1)
            mask_inbbox = t_max > t_min

            rays, rgb = torch.cat([rays_o, rays_d], dim = -1)[mask_inbbox], rgb[mask_inbbox]

            self.all_rays = torch.cat([self.all_rays, rays], dim = 0)
            self.all_rgb = torch.cat([self.all_rgb, rgb], dim = 0)

            self.frame_ray_correspondence = torch.cat([self.frame_ray_correspondence, idx * torch.ones(rays.shape[0], 1)], dim = 0)
        
        self.current_frame = 0

        return
    
    def load_eval_rays(self):
        self.eval_rays = torch.zeros(0, 6)
        self.eval_rgb = torch.zeros(0, 3)

        fx, fy = float(self.val_data['K'][0][0]), float(self.val_data['K'][1][1])
        
        image = torch.from_numpy(np.float32(imread(os.path.join(self.experiment_config['dataset_path'], self.val_data['frames'][0]['image'])))) / 255.
        height, width = int(image.shape[0]), int(image.shape[1])
        c2w = torch.from_numpy(np.float32(self.val_data['frames'][0]['transform_matrix']))

        i, j = torch.meshgrid(
                torch.arange(width, dtype=torch.float32),
                torch.arange(height, dtype=torch.float32),
                indexing = 'ij'
            )
    
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)

        directions = torch.stack(
            [
                (i - width * 0.5) / fx, -(j - height * 0.5) / fy, -torch.ones_like(i)
            ], 
            dim = -1
        )
        
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim = -1)
        rays_o = c2w[:3, -1].expand(rays_d.shape)

        rays_o, rays_d = rays_o.flatten(0, 1), rays_d.flatten(0, 1)
        self.eval_rays, self.eval_rgb = torch.cat([rays_o, rays_d], dim = -1), image.flatten(0, 1)

        self.height_eval, self.width_eval = height, width

        return

    def calc_gridsize(self, n_voxels, aabb):
        xyz_min, xyz_max = aabb.clone().detach()
        dim = len(xyz_min)
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
        return ((xyz_max - xyz_min) / voxel_size).long().tolist()
    
    def cal_n_samples(self):
        nSamples =  int(np.linalg.norm(self.model_config['gridSize']) / self.model_config['step_ratio'])
        return min(1e6, nSamples)

    def load_dataset(self):
        self.train_data = json.load(open(os.path.join(self.experiment_config['dataset_path'], 'train.json'), 'r'))
        self.val_data = json.load(open(os.path.join(self.experiment_config['dataset_path'], 'val.json'), 'r'))
        return self.train_data, self.val_data

    def save(self):
        self.tensorbase.save(os.path.join(self.experiment_config['dataset_path'], self.experiment_config['name']+ '.pth'))
        return
    
    def load(self):
        self.tensorbase.load(os.path.join(self.experiment_config['dataset_path'], self.experiment_config['name']+ '.pth'))
        return
    
    def get_train_batch_random(self, random = True):
        if random:
            idx = torch.randint(0, self.all_rays.shape[0], (self.training_config['batch_size'],))
        return self.all_rays[idx].to(self.device), self.all_rgb[idx].to(self.device)
    
    def get_train_batch_progressive(self):
        
        if self.current_frame == len(self.train_data['frames']) - 1:
            self.current_frame = 0

        r0 = torch.where(self.frame_ray_correspondence == self.current_frame)[0][0]
        r1 = torch.where(self.frame_ray_correspondence == self.current_frame)[0][-1]

        rays_all = self.all_rays[r0:r1]
        rgb_all = self.all_rgb[r0:r1]

        idx = torch.randint(0, rays_all.shape[0], (self.training_config['batch_size'],))

        self.current_frame += 1
        
        return rays_all[idx].to(self.device), rgb_all[idx].to(self.device)

    def train(self):
        self.n_samples = self.cal_n_samples()
        for step in range(self.training_config['n_iters']):

            if self.training_config['progressive']:
                rays, target = self.get_train_batch_progressive()
            else:
                rays, target = self.get_train_batch_random()
            
            loss = self.train_step(rays, target, step)

            if step + 1 in self.training_config['eval_on_iter']:
                self.eval(step + 1)
        
        self.save()
        return
    
    def train_step(self, rays, target, step):
        self.tensorbase.train()
        rgb, depth = self.tensorbase.forward(
            rays_chunk = rays, 
            white_bg = True,
            is_train = True,
            ndc_ray = False,
            N_samples = self.n_samples
        )

        loss = torch.nn.functional.mse_loss(rgb, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if step % 20 == 0:
            print(f"Step: {step}, Loss: {loss.item()}")

        for param_group in self.optim.param_groups:
            param_group['lr'] = param_group['lr'] * self.lr_factor

        if step in self.training_config['update_AlphaMask_list']:
            self.new_aabb = self.tensorbase.updateAlphaMask(tuple(self.model_config['gridSize']))

            if step == self.training_config['update_AlphaMask_list'][1]:
                mask = self.tensorbase.filtering_rays(self.all_rays, self.all_rgb)
                self.all_rays = self.all_rays[mask]
                self.all_rgb = self.all_rgb[mask]
                self.frame_ray_correspondence = self.frame_ray_correspondence[mask]

        if step in self.training_config['upsamp_list']:
            n_voxels = self.N_voxel_list.pop(0)
            self.model_config['gridSize'] = self.calc_gridsize(n_voxels, self.new_aabb)
            self.n_samples = self.cal_n_samples()
            self.tensorbase.upsample_volume_grid(self.model_config['gridSize'])
            if self.training_config['lr_upsample_reset']:
                lr_scale = 1
            else:
                lr_scale = self.training_config['lr_decay_target_ratio'] ** (step / self.training_config['n_iters'])
            self.optim = torch.optim.Adam(
                self.tensorbase.get_optparam_groups(
                    self.training_config['lr_init'] * lr_scale, self.training_config['lr_basis'] * lr_scale
                ), 
                betas = (0.9, 0.99)
            )
        
        return loss.item()

    def eval(self, step = None):
        if not os.path.exists(os.path.join(self.experiment_config['dataset_path'], 'eval')):
            os.mkdir(os.path.join(self.experiment_config['dataset_path'], 'eval'))

        rgb_out = np.zeros((self.eval_rays.shape[0], 3))
        for idx in range(0, self.eval_rays.shape[0], self.width_eval):
            rays = self.eval_rays[idx:idx + self.width_eval].to(self.device)
            rgb, depth = self.tensorbase.forward(
                rays_chunk = rays, 
                white_bg = True,
                is_train = False,
                ndc_ray = False,
                N_samples = -1
            )
            rgb = rgb.cpu().detach().numpy()
            rgb_out[idx:idx + self.width_eval] = rgb
        
        rgb_out = np.uint8(np.clip(rgb_out.reshape(self.height_eval, self.width_eval, 3) * 255, 0, 255))

        if step is None:
            imwrite(os.path.join(self.experiment_config['dataset_path'], self.experiment_config['name'] + '.png'), rgb_out)
        else:
            imwrite(os.path.join(self.experiment_config['dataset_path'], 'eval', f'{step}.png'), rgb_out)

        return

def main():
    parser = argparse.ArgumentParser(description="Train TensorRF-Debloated model")
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the experiment config JSON file"
    )
    args = parser.parse_args()

    # Load config and run
    config_path = args.config
    with open(config_path, 'r') as f:
        config = json.load(f)

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()