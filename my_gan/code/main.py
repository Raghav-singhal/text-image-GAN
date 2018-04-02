import argparse
from PIL import Image
import torchvision.utils as vutils
import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from txt2image import TextData
from  gan_cls import generator_cls, discriminator_cls



parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument("--l1_coef", default=50, type=float)
parser.add_argument("--l2_coef", default=100, type=float)
parser.add_argument("--predict", default=False, action='store_true')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--epochs', default=100, type=int)

parser.add_argument('--outf', default='outputs', type=str)
parser.add_argument('--resume', default='checkpoints/', type=str)
parser.add_argument('--resume_epoch', default=0, type=int)
parser.add_argument('--resume_trained_disc', default=None)
parser.add_argument('--resume_trained_gen', default=None)

args = parser.parse_args()


generator = torch.nn.DataParallel(generator_cls().cuda())
discriminator = torch.nn.DataParallel(discriminator_cls().cuda())


outf = './flower_dataset/' + args.outf
if not os.path.exists(outf):
    os.makedirs(outf)

### flower dataset
dataset = TextData('./flower_dataset/flowers.hdf5')
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers)

optimD = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5,0.999))

optimG = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5,0.999))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if args.pre_trained_gen:
    generator.load_state_dict(torch.load(args.resume_trained_gen))
else:
    generator.apply(weights_init)

if args.pre_trained_disc :
    discriminator.load_state_dict(torch.load(args.resume_trained_disc))
else:
    discriminator.apply(weights_init)

#  make changes here
hidden_dim = 100
l1_coef = args.l1_coef
l2_coef = args.l2_coef
checkpoints_path = 'checkpoints'

def train():
    criterion = nn.BCELoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    i = 0

    for epoch in range(args.resume_epoch ,args.epochs):
        for sample in data_loader:
            # print(i)

            real_image = sample['real_images']
            real_embed = sample['real_embed']
            fake_image = sample['wrong_images']

            real_image = Variable(real_image.float()).cuda()
            real_embed = Variable(real_embed.float()).cuda()
            fake_image = Variable(fake_image.float()).cuda()

            real_label = torch.ones(real_image.size(0))
            fake_label = torch.zeros(real_image.size(0))

            smoothed_real_labels = torch.FloatTensor(real_label.numpy() - 0.1)

            smoothed_real_labels = Variable(smoothed_real_labels).cuda()

            real_label = Variable(real_label).cuda()
            fake_label = Variable(fake_label).cuda()

            ### Discriminator Training

            discriminator.zero_grad()
            outputs, real_activ = discriminator(real_image, real_embed)

            real_loss = criterion(outputs, smoothed_real_labels)
            # real_loss = criterion(outputs, real_label)
            real_score = outputs

            outputs, _ = discriminator(fake_image, real_embed)
            wrong_loss = criterion(outputs, fake_label)

            noise = Variable(torch.randn(real_image.size(0), hidden_dim)).cuda()
            noise = noise.view(noise.size(0), hidden_dim, 1, 1)
            fake_image = generator(real_embed, noise)
            outputs, _ = discriminator(fake_image, real_embed)
            fake_loss = criterion(outputs, fake_label)
            fake_score = outputs

            lossD = real_loss + fake_loss

            lossD = lossD + wrong_loss

            lossD.backward()
            optimD.step()

            ### Generator Training
            generator.zero_grad()
            noise = Variable(torch.randn(real_image.size(0), hidden_dim)).cuda()
            noise = noise.view(noise.size(0), hidden_dim, 1, 1)
            fake_image = generator(real_embed, noise)

            outputs, fake_activ = discriminator(fake_image, real_embed)
            _, real_activ = discriminator(real_image, real_embed)

            fake_activ = torch.mean(fake_activ, 0)
            real_activ = torch.mean(real_activ, 0)

            lossG = criterion(outputs, real_label) + l1_coef * l1_loss(fake_activ, real_activ.detach()) + l2_coef * l2_loss(fake_image,real_image)
            lossG.backward()
            optimG.step()

            if i % 5 == 0:
                print("Epoch: %d, lossD = %f, lossG= %f, D(X)= %f, D(G(z))= %f" % (epoch, lossD.data.cpu().mean(), lossG.data.cpu().mean(), real_score.data.cpu().mean(),fake_score.data.cpu().mean()))

            if i % 50 == 0:

                vutils.save_image(real_image.data, outf + '/real_samples.png',normalize=True)

                vutils.save_image(fake_image.data, outf + '/fake_samples_epoch_%03d_%s.png' %(epoch,i),normalize=True)
                print(i)

            i += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:

            torch.save(discriminator.state_dict(), '{0}/disc_{1}.pth'.format(checkpoints_path, epoch))
            torch.save(generator.state_dict(), '{0}/gen_{1}.pth'.format(checkpoints_path, epoch))

def predict():
    for i,sample in enumerate(data_loader):
        real_image = sample['right_images']
        real_embed = sample['right_embed']
        txt = sample['txt']

        if not os.path.exists('results/{0}'.format(outf)):os.makedirs('results/{0}'.format(outf))

        real_image = Variable(real_image.float()).cuda()
        real_embed = Variable(real_embed.float()).cuda()

        noise = Variable(torch.randn(real_image.size(0), hidden_dim)).cuda()
        noise = noise.view(noise.size(0), hidden_dim, 1, 1)
        fake_image = generator(real_embed, noise)

        for image, t in zip(fake_image, txt):
            im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('results/{0}/{1}.jpg'.format(outf, t.replace("/", "")[:100]))
            print(t)

def main():
    if not args.predict:
        train()
        print('prediction')
        predict()
    elif args.resume:
        train()
        print('prediction')
        predict()
    elif args.predict:
        print('prediction')
        predict()

    predict()

if __name__ == '__main__':
    main()
