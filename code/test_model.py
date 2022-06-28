import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.io import write_jpeg
from code.models import UNet
from tqdm import tqdm

@torch.no_grad()
def test_model(model_state, valid_data, batch_size, shuffle, test_dir, test_count):
    #init device
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.empty_cache()        
    else:
        device = 'cpu'
    print('device: ',device)

    #clean log dir
    with os.scandir(test_dir) as it:
        for entry in it:
             if entry.is_file() and entry.name.startswith('val_') and entry.name.endswith('.jpg'):
                os.remove(os.path.join(test_dir,entry.name))

    #init generator
    gf = UNet.filters(model_state)
    generator = UNet(gf)
    generator.load_state_dict(model_state)
    generator.to(device)
    generator.eval()
    print('UNet init with %d filters and %d parametrs' % (gf, generator.params()))

    #input data and configure
    valid_load = DataLoader(valid_data, batch_size, shuffle=shuffle, num_workers=2)
    test_count = len(valid_data) if test_count<=0 else min(len(valid_data),test_count)
    count = 0

    #print test pictures
    with tqdm(total=test_count) as bar:
        for validbatch in valid_load:
            outline = validbatch['outline'].to(device)
            solid = validbatch['solid'].to(device)

            images = torch.cat([generator(outline),solid],3)
            stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            transform = tt.ConvertImageDtype(torch.uint8)
            images = transform((images.cpu()*stats[1][0]+stats[0][0]).clamp(min=0, max=1))
            batch_count = min(images.shape[0],test_count-count)
            for i in range(batch_count):
                write_jpeg(images[i],os.path.join(test_dir,'val_%06d.jpg' % (i+count+1)))
            count += batch_count
            bar.update(batch_count)
            if count==test_count:
                break

    print('%d images completed at path %s' % (count, test_dir))