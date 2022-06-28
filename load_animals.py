import os
import torch
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.io import read_image
from torchvision.io import write_jpeg
import torchvision.transforms as tt

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default = os.path.join('.', 'data'))
    p.add_argument('--crop_count', default=2, type=int)
    p.add_argument('--crop_size', default=512, type=int)
    p.add_argument('--train_size', default=0.8, type=float)
    p.add_argument('--gpu', default=0, type=int)
    args = p.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    data_dir = os.path.join(args.data_root, 'animals')
    if not os.path.exists(data_dir):
        print("images not found, donwloading...")
        archive = 'awa2.tmp.zip'
        os.system("wget http://cvml.ist.ac.at/AwA2/AwA2-data.zip -O %s" % archive)
        print("extracting...")
        os.system("unzip -q %s" % archive)
        os.remove(archive)
        tmp_dir = os.path.join('.','Animals_with_Attributes2')
        assert os.path.exists(tmp_dir)

        #scan temorary foldiers
        data_items = []
        for root, dirs, files in os.walk(tmp_dir):
            for name in files:
                if name.endswith('.jpg'):
                    data_items.append(os.path.join(root,name))

        #make directory structure
        os.mkdir(data_dir)
        train_items, valid_items = train_test_split(data_items, train_size = args.train_size)

        #render train images
        os.mkdir(os.path.join(data_dir,'train'))
        print('render train images x %d:' % args.crop_count)
        for i in tqdm(range(len(train_items))):
            image = read_image(train_items[i])
            transform = tt.Compose([tt.RandomCrop(min(image.shape[1],image.shape[2],args.crop_size)), tt.Resize(256)])
            for j in range(args.crop_count):
                color_image = transform(image) if image.shape[0]==3 else transform(torch.cat([image,image,image],0))            
                pair_images = torch.cat([tt.Grayscale(3)(color_image),color_image],2)
                write_jpeg(pair_images,os.path.join(data_dir,'train','animal_%06d_%02d.jpg' % (i+1,j+1)))
            os.remove(train_items[i])

        #render validation images
        os.mkdir(os.path.join(data_dir,'val'))
        print('render validation images:')
        for i in tqdm(range(len(valid_items))):
            image = read_image(valid_items[i])
            transform = tt.Compose([tt.RandomCrop(min(image.shape[1],image.shape[2],args.crop_size)), tt.Resize(256)])
            color_image = transform(image) if image.shape[0]==3 else transform(torch.cat([image,image,image],0))
            pair_images = torch.cat([tt.Grayscale(3)(color_image),color_image],2)
            write_jpeg(pair_images,os.path.join(data_dir,'val','animal_%06d.jpg' % (i+1)))
            os.remove(valid_items[i])

        #clean and remove temporary directory
        for root, dirs, files in os.walk(tmp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root,name))
            for name in dirs:
                os.rmdir(os.path.join(root,name))
        os.rmdir(tmp_dir)

        #print done
        print("done")