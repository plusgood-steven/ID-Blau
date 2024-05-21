import torch
import torch.nn as nn
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class PerceptualLoss():
	def __init__(self, features=34, device='cuda'):
		self.device = device
		self.features = features
		self.criterion = torch.nn.functional.l1_loss
		self.contentFunc = self.contentFunc()
		self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



	def contentFunc(self):
		cnn = torchvision.models.vgg19(pretrained=True).features
		model = nn.Sequential()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == self.features:
				break
		# No need to BP to variable
		for k, v in model.named_parameters():
			v.requires_grad = False

		return model.to(self.device).eval()
		
	def get_loss(self, fakeIm, realIm):
		fakeIm = self.transform(fakeIm)
		realIm = self.transform(realIm)
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss

	def __call__(self, fakeIm, realIm):
	    return self.get_loss(fakeIm, realIm)

class L1andPerceptualLoss(nn.Module):
    def __init__(self, gamma=0.1):
        super(L1andPerceptualLoss, self).__init__()
        self.preceptual = PerceptualLoss()
        self.l1 = CharbonnierLoss()
        self.gamma = gamma
    
    def forward(self, input, target):
        l1_loss = self.l1(input, target)
        preceptual_loss = self.preceptual(input, target)

        return l1_loss + self.gamma * preceptual_loss
    
class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
          super(GANLoss, self).__init__()
          self.real_label = target_real_label
          self.fake_label = target_fake_label
          self.gan_mode = gan_mode
    
    def get_zero_tensor(self, input):
        return torch.zeros_like(input).requires_grad_(False)

    def forward(self, input, target_is_real, for_discriminator=False):
        if self.gan_mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -input.mean()
        else:
            raise ValueError("'Unexpected gan_mode {}'.format(gan_mode)")
        
        return loss


def wgan_gp_loss(D, real_data, fake_data, batch_size, device):
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    alpha = alpha.expand_as(real_data)

    interpolated_data = alpha * real_data + (1 - alpha) * fake_data

    interpolated_data = interpolated_data.requires_grad_(True)

    interpolated_output = D(interpolated_data)

    gradients = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated_data,
        grad_outputs=torch.ones_like(interpolated_output).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradients_penalty






