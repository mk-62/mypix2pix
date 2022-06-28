import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.io import write_jpeg
from code.models import UNet
from code.models import PatchGAN
from tqdm import tqdm

def train_model(train_data, valid_data, epochs, batch_size,
                    log_dir, log_rate, log_samples, gf, df, noise, lr):
    #init device
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.empty_cache()        
    else:
        device = 'cpu'
    print('device: ',device)

    #clean log dir
    with os.scandir(log_dir) as it:
        for entry in it:
            islog = entry.is_file() and (
                entry.name.startswith('val_epoch_') and entry.name.endswith('.jpg') or
                entry.name.startswith('log.txt'))
            if islog:
                os.remove(os.path.join(log_dir,entry.name))

    #init generator
    generator = UNet(gf)
    print('UNet init with %d filters and %d parametrs' % (gf, generator.params()))
    generator.to(device)
    generator.init_weights()
    generator.train()

    #init discriminator
    discriminator = PatchGAN(df)
    print('PatchGAN init with %d filters and %d parametrs' % (df,discriminator.params()))
    discriminator.to(device)
    discriminator.init_weights()
    discriminator.train()

    #init data and configure
    losses_g = []
    losses_d = []
    adv_loss_fn = nn.BCEWithLogitsLoss()
    img_loss_fn = nn.L1Loss()
    lambda_value = 100.0
    train_load = DataLoader(train_data, batch_size, shuffle=True, num_workers=2) 
    valid_load = DataLoader(valid_data, log_samples, shuffle=True, num_workers=2)   

    #build optimizers
    optimizer_g  = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    print('batch_size=%d discriminator noise=%f lr=%f\n' % (batch_size, noise, lr))

    #train network
    for epoch in range(epochs):
        loss_d_per_epoch = []
        loss_g_per_epoch = []
        generator.train()

        for trainbatch in tqdm(train_load):
            outline = trainbatch['outline'].to(device)
            solid = trainbatch['solid'].to(device)

            #generator forward
            optimizer_d.zero_grad()
            generated = generator(outline)

            #train discriminator with generated fake images
            fgen_pair = torch.cat([outline,generated],1)
            fgen_d = discriminator(fgen_pair.detach())
            fgen_label = torch.randn(tuple(fgen_d.shape),device=device).abs()*noise
            loss_fgen_d = adv_loss_fn(fgen_d,fgen_label)

            #train discriminator with real images
            real_pair = torch.cat([outline,solid],1)
            real_d = discriminator(real_pair)
            real_label = torch.randn(tuple(real_d.shape),device=device).abs()*-noise+1.0
            loss_real_d = adv_loss_fn(real_d,real_label)

            #calculate common discriminator loss
            loss_d = (loss_fgen_d + loss_real_d) / 2
            loss_d_per_epoch.append(loss_d.item())
            loss_d.backward()
            optimizer_d.step()

            #train generator with real labels
            optimizer_g.zero_grad()
            loss_g_img = img_loss_fn(generated,solid)
            loss_g_adv = adv_loss_fn(discriminator(fgen_pair),real_label)
            loss_g = loss_g_img*lambda_value + loss_g_adv
            loss_g_per_epoch.append(loss_g.item())
            loss_g.backward()
            optimizer_g.step()
            
        # Record losses & scores
        losses_g.append(np.mean(loss_g_per_epoch))
        losses_d.append(np.mean(loss_d_per_epoch))

        #render samples log
        if epoch % log_rate == 0:
            generator.eval()
            with torch.no_grad():
                validbatch = next(iter(valid_load))
                outline, solid = validbatch['outline'].to(device), validbatch['solid'].to(device)
                images = torch.cat([generator(outline),solid],3)
                stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                transform = tt.ConvertImageDtype(torch.uint8)
                images = transform((images.cpu()*stats[1][0]+stats[0][0]).clamp(min=0, max=1))
                for i in range(images.shape[0]):
                    write_jpeg(images[i],os.path.join(log_dir,'val_epoch_%04d_%02d.jpg' % (epoch+1,i+1)))

        #noise_power = noise_power * 0.999    
        print("epoch %d from %d: loss_g=%f loss_d=%f" % (epoch+1, epochs, losses_g[-1], losses_d[-1]))

    #write losess log
    f = open(os.path.join(log_dir,'log.txt'),'w')
    f.write('UNet init with %d filters and %d parametrs\n' % (gf, generator.params()))
    f.write('PatchGAN init with %d filters and %d parametrs\n' % (df,discriminator.params()))
    f.write('batch_size=%d discriminator noise=%f lr=%f\n\n' % (batch_size, noise, lr))
    for i in range(len(losses_g)):
        f.write('epoch %d from %d: loss_g=%f loss_d=%f\n' % (i+1, epochs, losses_g[i], losses_d[i]))
    f.close()

    return generator.state_dict()
