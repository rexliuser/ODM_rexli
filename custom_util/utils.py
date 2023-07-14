import os
import glob

def generate_split(img_folder, root_folder):
    #lst = os.listdir(img_folder)
    lst = glob.glob(os.path.join(img_folder,'*.jpg'))
    lst = [os.path.basename(file) for file in lst]
    print(lst)
    groups = [f.split('_')[1][0] for f in lst]
    print(groups)
    with open(os.path.join(root_folder, 'image_groups.txt'), 'w') as f:
        for i in range(len(lst)):
            text = f"{lst[i]} {chr(int(groups[i])+65)}\n"
            f.write(text)
    