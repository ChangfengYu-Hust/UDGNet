import cv2
import os
import argparse
import glob

import torch
import torchvision as tv
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from lib.utils import *
from lib.dataset import *
import time
from lib.model import *
from lib.mean_patch import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device_ids = [0]

parser = argparse.ArgumentParser(description="UDGNet_Test")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--logdir", type=str, default='output/train', help='path of log files')
parser.add_argument("--rain_path",type=str,default="./dataset/test/rain",help='path of rain dataset')
parser.add_argument("--angle_path",type=str,default="./dataset/train/angle",help='path of angle path')
parser.add_argument("--clean_path",type=str,default="./dataset/train/clean",help='path of clean path')
parser.add_argument("--weight_path",tyrp=str,default="./output/real_model/generator_backup.pth",help='path of weight')
parser.add_argument("--outf", type=str, default="output",help='path of log files')
parser.add_argument("--test_type", type=str, default="Whole", help='path of dataset')
# parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

transform = tv.transforms.Compose(
    [  # tv.transforms.Resize(128),
        tv.transforms.ToTensor()
        # tv.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])

def test(weight_path, weight_name):
    print('Loading dataset ...\n')
    dataset_val = MyData_cityscape(opt.rain_path,opt.clean_path,opt.angle_path, transforms=transform, patch_size=0,batch_size=1,repeat=0,channel=3)
    loader_val = DataLoader(dataset=dataset_val, num_workers=16, batch_size=1, shuffle=False)
    print("# of training samples: %d/n" % int(len(dataset_val)))

    # Build model
    generator = UDGNet(inchannel=3)
    generator.apply(weights_init_kaiming)
    model = nn.DataParallel(generator, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    angle_model = angle_estimation(in_channels=3)
    angle_model.apply(weights_init_kaiming)
    angle_model = nn.DataParallel(angle_model, device_ids=device_ids).cuda()
    angle_weight = weight_path.replace("generator", "angle_model")
    angle_model.load_state_dict(torch.load(angle_weight))
    angle_model.eval()
    # test info
    psnr_test = 0
    AVGNETtime = 0
    AVGWHOLEtime = 0
    count = 0
    output_folder = opt.outf + '/result/train_Decomposition_angle_real_test/' + weight_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, 'output'))
        os.makedirs(os.path.join(output_folder, 'visualize'))
        os.makedirs(os.path.join(output_folder, 'label'))
        os.makedirs(os.path.join(output_folder, 'rain'))

    writer = SummaryWriter(output_folder)
    step = 0
    WHOLEstart = time.time()
    with open(output_folder + '/results.txt', 'w') as result_file:
        for i, data in enumerate(loader_val):
            rain, label,angle_label,file_name = data['rain_data'].type(torch.FloatTensor).cuda(), \
                                                data['label'].type(torch.FloatTensor).cuda(), \
                                                data['angle_label'].type(torch.FloatTensor).cuda(), \
                                                data['filename']
            (H, W) = rain.shape[2:]
            if opt.test_type == 'mean_patch':
                input = rain.cpu().numpy()
                im_patch = crop_patch(input, [128, 128], [64, 64])
                im_patch = torch.Tensor(im_patch).cuda()

                gt = label.cpu().numpy()
                gt_patch = crop_patch(gt, [128, 128], [64, 64])
                gt_patch = torch.tensor(gt_patch).cuda()

                gt_patches = torch.zeros_like(gt_patch)
                out_patches = torch.zeros_like(im_patch)
                with torch.no_grad():  # this can save much memory
                    NETstart = time.time()
                    for i in range(im_patch.shape[0]):
                        img_input = im_patch[i].unsqueeze(0)
                        (h, w) = img_input.shape[2:]  # original patch size

                        # broad the rain image Border
                        img_temp = np.transpose(im_patch[i].cpu().numpy(), [1, 2, 0])
                        img_temp = cv2.copyMakeBorder(img_temp, 20, 20, 20, 20, cv2.BORDER_REFLECT)
                        img_temp = np.transpose(img_temp, [2, 0, 1])
                        # img_temp = img_temp[np.newaxis,:,:]
                        # broad the label image Border
                        gt_temp = np.transpose(gt_patch[i].cpu().numpy(), [1, 2, 0])
                        gt_temp = cv2.copyMakeBorder(gt_temp, 20, 20, 20, 20, cv2.BORDER_REFLECT)
                        # gt_temp = gt_temp[np.newaxis, :, :]
                        gt_temp = np.transpose(gt_temp, [2, 0, 1])
                        gt_tensor = torch.Tensor(gt_temp).unsqueeze(0).cuda()

                        # estimate the angle by original patch
                        rain_patch = torch.Tensor(img_temp).unsqueeze(0).cuda()
                        out_angle = angle_model(img_input)

                        # transform the broaded gt patch
                        transform_gt = transform_no_crop(gt_tensor, out_angle).cuda()

                        # transform the broaded rain patch and derain
                        transform_rain = transform_no_crop(rain_patch, out_angle).cuda()

                        visual_rain = np.transpose(transform_rain.cpu().squeeze(0).numpy(), [1, 2, 0]) * 255
                        visual_rain = visual_rain.astype('uint8')
                        cv2.namedWindow("test",0)
                        cv2.imshow("test",visual_rain)
                        cv2.waitKey(0)



                        out_put = model(transform_rain)
                        out_img, Out_R = out_put['out_clean'], \
                                         out_put['rain_streak']

                        # Turn right the output
                        out_img_large = transform_no_crop(out_img, 1 - out_angle)
                        (nH, nW) = out_img_large.shape[2:]
                        newH = int((nH - h) / 2)
                        newW = int((nW - w) / 2)
                        out_img = out_img_large[:, :, newH:newH + 128, newW:newW + 128]
                        out_img = torch.clamp(out_img, 0., 1.)
                        out_patches[i, :, :, :] = out_img[0, :, :, :]

                        # Turn right the gt
                        gt_img_large = transform_no_crop(transform_gt, 1 - out_angle)
                        (nH, nW) = gt_img_large.shape[2:]
                        newH = int((nH - h) / 2)
                        newW = int((nW - w) / 2)
                        gt_img = gt_img_large[:, :, newH:newH + 128, newW:newW + 128]
                        gt_img = torch.clamp(gt_img, 0., 1.)
                        gt_patches[i, :, :, :] = gt_img[0, :, :, :]

                        visual_rain_large = transform_no_crop(transform_rain, 1 - out_angle)
                        visual_rain = visual_rain_large[:, :, newH:newH + 128, newW:newW + 128]

                        # visual_rain = np.transpose(visual_rain.cpu().squeeze(0).numpy(), [1, 2, 0]) * 255
                        # visual_rain = visual_rain.astype('uint8')
                        # visual = np.transpose(out_img.cpu().squeeze(0).numpy(),[1,2,0])*255
                        # visual = visual.astype('uint8')
                        # imgs = np.hstack([visual_rain,visual])
                        # cv2.namedWindow("test",0)
                        # cv2.imshow("test",imgs)
                        # cv2.waitKey(0)
                    out_patches = out_patches.cpu().numpy()
                    out_Whole = mean_patch(out_patches, [H, W], [64, 64])
                    out_Whole = torch.Tensor(out_Whole)

                    gt_patches = gt_patches.cpu().numpy()
                    gt_Whole = mean_patch(gt_patches, [H, W], [64, 64])
                    gt_Whole = torch.Tensor(gt_Whole)
                    NETend = time.time()
                    temp_time = NETend - NETstart
                    filename = file_name[0]
                    print("\n%s" % (filename))
                    rain_images = np.transpose(rain.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255
                    rain_images = rain_images.astype('uint8')
                    label_images = np.transpose(gt_Whole.cpu().numpy(), (1, 2, 0)) * 255
                    # label_images = np.transpose(gt_Whole.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255
                    label_images = label_images.astype('uint8')
                    model_output2 = np.transpose(out_Whole.cpu().numpy(), (1, 2, 0)) * 255
                    model_output2 = model_output2.astype('uint8')
                # sample_model_out1_mask = model_out1_mask[index]
                # sample_model_out2_mask = model_out2_mask[index]

                # sample_model_out_all_mask = model_out_all_mask[index]

                rain_filename = filename.split(".")[0]
                rain_filename = rain_filename + "_rain.png"

                imgs1 = np.row_stack([rain_images, model_output2])
                cv2.namedWindow("test", 0)
                cv2.imshow("test", imgs1)
                cv2.waitKey(0)

                cv2.imwrite(output_folder + '/label/' + filename, label_images)
                cv2.imwrite(output_folder + '/output/' + rain_filename, model_output2)
                cv2.imwrite(output_folder + '/visualize/' + 'img_' + filename, imgs1)
            else:
                with torch.no_grad():
                    NETstart = time.time()
                    #estimate angle
                    input = rain.cpu().numpy()
                    im_patch = crop_patch(input, [128, 128], [64, 64])
                    im_patch = torch.Tensor(im_patch).cuda()
                    out_angle = angle_model(im_patch)
                    mean_angle = torch.mean(out_angle)
                    mean_angle = mean_angle.reshape(-1,1,1,1)
                    # mean_angle = angle_label
                    input = np.transpose(rain.squeeze(0).cpu().numpy(), [1, 2, 0])

                    img_temp = cv2.copyMakeBorder(input, 20, 20, 20, 20, cv2.BORDER_REFLECT)
                    img_temp = np.transpose(img_temp, [2, 0, 1])
                    Broad_rain = torch.Tensor(img_temp).unsqueeze(0).cuda()

                    clean = np.transpose(label.squeeze(0).cpu().numpy(), [1, 2, 0])
                    clean_temp = cv2.copyMakeBorder(clean, 20, 20, 20, 20, cv2.BORDER_REFLECT)
                    clean_temp = np.transpose(clean_temp, [2, 0, 1])
                    Broad_label = torch.Tensor(clean_temp).unsqueeze(0).cuda()

                    transform_rain = transform_no_crop(Broad_rain, mean_angle).cuda()
                    transform_clean = transform_no_crop(Broad_label, mean_angle).cuda()


                    out_put = model(transform_rain)
                    out_img, Out_R = out_put['out_clean'], \
                                               out_put['rain_streak']
                    out_img_large = transform_no_crop(out_img, 1 - mean_angle)
                    (nH, nW) = out_img_large.shape[2:]
                    out_img = out_img_large[:, :, int((nH - H) / 2):int((nH + H) / 2), int((nW - W) / 2):int((nW + W) / 2)]
                    out_img = torch.clamp(out_img, 0., 1.)
                    clean_img_large = transform_no_crop(transform_clean, 1 - mean_angle)
                    real_image2 = clean_img_large[:, :, int((nH - H) / 2):int(int((nH + H) / 2)),
                                  int((nW - W) / 2):int((nW + W) / 2)]

                    NETend = time.time()
                    temp_time = NETend - NETstart
                    filename = file_name[0]
                    Out2 = torch.clamp(out_img, 0., 1.)
                    rain_images = np.transpose(rain.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255
                    label_images = np.transpose(real_image2.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255
                    model_output2 = np.transpose(Out2.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255
                    imgs = np.row_stack([rain_images,  model_output2])
                    model_output2 = model_output2.astype('uint8')
                    cv2.namedWindow("test",0)
                    cv2.imshow("test",model_output2)
                    cv2.waitKey(0)
                    cv2.imwrite(output_folder + '/label/' + filename, label_images)
                    cv2.imwrite(output_folder + '/output/' + filename, model_output2)
                    cv2.imwrite(output_folder + '/visualize/' + filename, imgs)

        WHOLEend = time.time()
        AVGWHOLEtime += WHOLEend - WHOLEstart

        psnr_test /= step
        AVGNETtime /= step
        avgNETFPS = 1 / AVGNETtime
        AVGWHOLEtime /= step
        avgWHOLEFPS = 1 / AVGWHOLEtime
        print("\nPSNR on test data %f" % psnr_test)
        print("\nAVG NET Test Time on test data %f" % AVGNETtime)
        print("\nAVG NET FPS on test data %f" % avgNETFPS)
        print("\nAVG WHOLE Test Time on test data %f" % AVGWHOLEtime)
        print("\nAVG WHOLE FPS on test data %f" % avgWHOLEFPS)

        result_file.writelines("\nPSNR on test data %f" % psnr_test)
        result_file.writelines("\nAVG NET Test Time on test data %f" % AVGNETtime)
        result_file.writelines("\nAVG NET FPS on test data %f" % avgNETFPS)
        result_file.writelines("\nAVG WHOLE Test Time on test data %f" % AVGWHOLEtime)
        result_file.writelines("\nAVG WHOLE FPS on test data %f" % avgWHOLEFPS)
        result_file.close()
    # writer.add_scalar('Avg PSNR on test data', psnr_test, 0)
    # writer.add_scalar('AVG NET Test Time on test data', AVGNETtime, 0)
    # writer.add_scalar('AVG NET FPS on test data', avgNETFPS, 0)
    # writer.add_scalar('AVG WHOLE Test Time on test data', AVGWHOLEtime, 0)
    # writer.add_scalar('AVG WHOLE FPS on test data', avgWHOLEFPS, 0)


def computePSNR(im1, im2):
    diff = im1 - im2
    mse = np.mean(np.square(diff.detach().cpu().numpy()))
    psnr = 10 * np.log10(1 * 1 / mse)
    return psnr


if __name__ == "__main__":
    # file_list = glob.glob(os.path.join(opt.logdir, '*.pth'))
    # file_list.sort()
    file_list = [opt.weight_path]
    for weight in file_list:
        weight_name = weight.split('/')[-1].split('.')[0]
        print(weight_name)
        test(weight_path=weight, weight_name=weight_name)
    print('test done！')
