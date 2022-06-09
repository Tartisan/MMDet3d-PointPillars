import numpy as np 
import numpy as np
from open3d_vis import show_pts_boxes
import open3d as o3d
import yaml

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
        merge_new_config(config=config, new_config=new_config)
    return config


def dataloader(cloud_path , boxes_path):
    cloud = np.fromfile(cloud_path, dtype=np.float32, count=-1).reshape([-1, 6])[:, :5]
    boxes = np.loadtxt(boxes_path).reshape(-1,7)
    return cloud , boxes 

def main():
    import yaml
    with open("bootstrap.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cloud ,boxes = dataloader(config['InputFile'], config['OutputFile'])
    print(cloud.shape)

    show_pts_boxes(cloud, boxes)


if __name__ == "__main__":
    main()