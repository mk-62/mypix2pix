import os
import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default = os.path.join('.', 'data'))
    args = p.parse_args()

    data_dir = os.path.join(args.data_root, 'edges2shoes')
    if not os.path.exists(data_dir):
        print("images not found, donwloading...")
        archive = 'edges2shoes.tmp.tgz'
        os.system("wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz -O %s" % archive)
        print("extracting...")
        os.system("tar -C %s -xvzf %s" % (args.data_root, archive))
        os.remove(archive)
        assert os.path.exists(data_dir)

        #print done
        print("done")