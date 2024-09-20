import os
import numpy as np
import torch
import warnings
import cv2
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def plot_scanpath(img_path,scanpaths,save_path="",img_height=320,img_width=512):
    image = cv2.resize(matplotlib.image.imread(img_path), (img_width, img_height))

    fig, ax = plt.subplots()
    ax.imshow(image)

    xs = [x*2 for x in scanpaths['X']]
    ys = [x*2 for x in scanpaths['Y']]
    ts = [x*2 for x in scanpaths['T']]

    cir_rad_min, cir_rad_max = 10,18
    min_T, max_T = np.min(ts), np.max(ts)
    rad_per_T = (cir_rad_max - cir_rad_min) / float(max_T - min_T)

    linewidth = 2
    for i in range(len(xs)):
        if i > 0:
            plt.plot([xs[i], xs[i - 1]], [ys[i], ys[i - 1]], color='red',linewidth=linewidth, alpha=0.35)

    for i in range(len(xs)):
        cir_rad = int(14 + rad_per_T * (ts[i] - min_T))
        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            facecolor='yellow',
                            alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(
            i+1), xy=(xs[i], ys[i]+3), fontsize=10, ha="center", va="center")

    ax.axis('off')
    if not save_path:
        plt.show(bbox_inches='tight', pad_inches=-0.1)
    else:
        parent_dir = os.path.dirname(save_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        plt.savefig(str(save_path), bbox_inches='tight', pad_inches=-0.1, dpi=2000)
    plt.cla()

if __name__=="__main__":
    image_dir="/data/lyt/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/images/"
    scanpath_dir = "/data/lyt/02-Results/01-ScanPath/ClipGaze/compare_results/ours/"
    save_dir = "/data/lyt/02-Results/01-ScanPath/ClipGaze/compare_results_plot/ours/"
    for cartogories in os.listdir(scanpath_dir):
        car_path=scanpath_dir+cartogories+'/'
        for file in os.listdir(car_path):
            path = car_path + file
            image_path = image_dir + cartogories + '/' + file.split('.')[0]+'.jpg'
            scanpath = torch.load(path)
            save_path = save_dir + cartogories + '/' + file.split('.')[0]+'.jpg'
            # save_path = ""
            plot_scanpath(image_path, scanpath, save_path=save_path, img_height=640, img_width=1024)

        print('done!')
    print('finished')



