import torch,math,glob,re,cv2,os

from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from utils import *

def tryint(s):                       #//将元素中的数字转换为int后再排序
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):               # //将元素中的字符串和数字分割开
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sort_humanly(v_list):    #//以分割后的list为单位进行排序
    return sorted(v_list, key=str2int)
def transform(input,angle_nor,name):
    angle = 90 - ((angle_nor) * 90 + 45)
    alpha = math.radians(angle)
    image = input
    (h, w) = image.shape[:2]
    image = Image.fromarray(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    img = image
    N, C, h1, w1 = img.size()

    theta = torch.tensor([
        [math.sin(-alpha), math.cos(alpha), 0],
        [math.cos(alpha), math.sin(alpha), 0]
    ], dtype=torch.float)
    theta = theta.unsqueeze(0)
    nW = math.ceil(h*math.fabs(math.sin(alpha))+w*math.cos(alpha))
    nH = math.ceil(h*math.cos(alpha)+w*math.fabs(math.sin(alpha)))

    g = AffineGridGen(nH, nW, aux_loss=True)
    grid_out, aux = g(theta)
    grid_out[:,:,:,0] = grid_out[:,:,:,0]*nW / w
    grid_out[:, :, :, 1] = grid_out[:, :, :, 1] * nW/h
    out = F.grid_sample(img, grid_out)

    out1 = out.cpu().squeeze(0).numpy().transpose(1,2,0)*255
    out1 = out1.astype('uint8')
    return out1

if __name__ == "__main__":
    img_path = "K:/YCF_Project/data/real_rain_streak"
    save_path = "output_patch2"
    imgs_list = glob.glob(img_path+"/*.png")
    imgs_list = sort_humanly(imgs_list)
    size_input = 256
    count = 0
    for i in range(len(imgs_list)):
        image = cv2.imread(imgs_list[i])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        name = imgs_list[i].split('\\')[-1]
        name = name.split('.')[0]
        for n in range (0,np.shape(image)[0],size_input//2):
            if n+size_input > np.shape(image)[0]:
                if n + 20 > np.shape(image)[0]:
                    break
                else:
                    temp_n = np.shape(image)[0] - size_input
            else:
                temp_n = n
            for m in range(0,np.shape(image)[0],size_input//2):
                if m + size_input > np.shape(image)[1]:
                    if m + 20 > np.shape(image)[1]:
                        break
                    else:
                        temp_n = np.shape(image)[0] - size_input
                else:
                    temp_m = m
                image_crop = image[temp_n:temp_n+size_input,temp_m:temp_m+size_input]

                angle = 0.5
                while True:
                    out = transform(image_crop,angle,name)
                    cv2.namedWindow("test",0)
                    cv2.imshow("test", out)
                    key = cv2.waitKey(0)
                    transformed_image_path = save_path + '/transformed_image'
                    angle_save_path = save_path + '/angle'
                    img_save_path = save_path + '/original_image'
                    Clean_save_path = save_path + '/clean_image'
                    if not os.path.exists(img_save_path):
                        os.makedirs(img_save_path)
                        os.makedirs(angle_save_path)
                        os.makedirs(transformed_image_path)
                        os.makedirs(Clean_save_path)
                    original_img_save_dir = img_save_path + '/'+name + '_{}.png'.format(count)
                    transform_img_save_dir = transformed_image_path + '/' + name + '_{}.png'.format(count)
                    Clean_img_save_dir = Clean_save_path + '/' + name + '_{}.png'.format(count)
                    angle_save_dir = angle_save_path + '/'+ name.replace(".png",".npy")
                    if key&0xff == ord('s'):
                        cv2.imwrite(transform_img_save_dir,out)
                        cv2.imwrite(original_img_save_dir,image_crop)
                        np.save(angle_save_dir,angle)
                        count += 1
                        break
                    elif key&0xff == ord('a'):
                        angle = angle - 0.01
                    elif key&0xff == ord('o'):
                        cv2.imwrite(Clean_img_save_dir, image_crop)
                        count += 1
                    elif key&0xff == ord('d'):
                        angle = angle + 0.01
                    elif key&0xff == ord('q') or angle >1 or angle<0:
                        break

