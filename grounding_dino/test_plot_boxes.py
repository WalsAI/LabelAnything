import matplotlib.pyplot as plt 
import pandas as pd
import cv2
import matplotlib.patches as patches
import numpy as np

def try_plot_boxes(dataframe, image):
    img_csv = pd.read_csv(dataframe)
    image_read = cv2.imread(image)
    image_read = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    
    ax.imshow(image_read)

    b_box = img_csv["Boxes"][0]
    b_box = b_box.replace('[[',"").replace("]]","").replace("[","").replace("]","").replace("  ", " ")



    b_box_list = b_box.strip().split(' ')
    
    for i in range (len(b_box_list)):
        b_box_list[i] = b_box_list[i].strip()
        
    start_pos = 0
    list_bounding_boxes = []
    for i in range((len(b_box_list) // 4)):
        b_box_1 = b_box_list[start_pos:((i+1)*4)]
        start_pos = ((i+1)*4)

        list_bounding_boxes.append(b_box_1)

    print(list_bounding_boxes)

    b_box = np.array(list_bounding_boxes, dtype=float)

    for i in range(len(b_box)):
        rect = patches.Rectangle((b_box[i][0]*image_read.shape[1], b_box[i][1]*image_read.shape[0]), b_box[i][2]*image_read.shape[1]-b_box[i][0]*image_read.shape[1], b_box[i][3]*image_read.shape[0] -b_box[i][1]*image_read.shape[0], linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)
    plt.show()
    plt.savefig("/root/LabelAnythingWeights/images/imageshown.png")




if __name__ == "__main__":
    try_plot_boxes('/root/LabelAnything/Run/b.csv', "/root/LabelAnythingWeights/images/pothole.jpeg")