import cv2
import os


folder = '<path to your downloaded DF2K dataset>'
dest_folder = '<path to your output cropped training dataset>'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

patchsize = 256
stride = 256

count = 1

for img_n in sorted(os.listdir(folder)):
    img = cv2.imread(os.path.join(folder, img_n))
    h, w, _ = img.shape
    h_number = h // patchsize
    w_number = w // patchsize
    for i in range(h_number):
        for j in range(w_number):
            start_ij_l = j * stride
            start_ij_u = i * stride
            end_ij_l = start_ij_l + stride
            end_ij_u = start_ij_u + stride
            img_crop = img[start_ij_u:end_ij_u, start_ij_l:end_ij_l]
            cv2.imwrite(os.path.join(dest_folder, '{:0>6d}.png'.format(count)), img_crop)
            count += 1
print("{} done!".format(folder))
