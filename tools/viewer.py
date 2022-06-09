import numpy as np 
import numpy as np
from open3d_vis import show_pts_boxes
import open3d as o3d
import yaml

def dataloader(cloud_path , boxes_path):
    cloud = np.fromfile(cloud_path, dtype=np.float32, count=-1).reshape([-1, 6])[:, :5]
    boxes = np.loadtxt(boxes_path).reshape(-1,7)
    return cloud , boxes 

def main():
    with open("bootstrap.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cloud ,boxes = dataloader(config['InputFile'], config['OutputFile'])
    print(cloud.shape)

    show_pts_boxes(cloud, boxes)


if __name__ == "__main__":
    main()