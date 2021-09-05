import argparse
import torchvision as tv
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from lib.model import *

from lib.dataset import MyData_cityscape
from lib.utils import *
from lib.loss_function import *
from matplotlib import pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]

parser = argparse.ArgumentParser(description="UDGNet")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
parser.add_argument("--milestone", type=list, default=[30, 60, 80], help="When to decay learning rate;")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="output", help='path of log files')
parser.add_argument("--rain_path",type=str,default="./dataset/test/rain",help='path of rain dataset')
parser.add_argument("--angle_path",type=str,default="./dataset/train/angle",help='path of angle path')
parser.add_argument("--clean_path",type=str,default="./dataset/train/clean",help='path of clean path')
parser.add_argument("--reset", type=int, default=1, help='path of dataset')
opt = parser.parse_args()



transform = tv.transforms.Compose(
    [#tv.transforms.Resize(128),
        tv.transforms.ToTensor()
        # tv.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def main():
    # Load dataset
    print('Loading dataset .../n')
    repeat = 2e3
    dataset_train = MyData_cityscape(opt.rain_path,opt.clean_path,opt.angle_path,transforms=transform,patch_size=128,batch_size=opt.batchSize,repeat=repeat,channel=3)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d/n" % int(len(dataset_train)))
    # 定义参数
    generator = UDGNet(inchannel=3)
    angle_model = angle_estimation(in_channels=3)
    discriminator = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d)

    generator.apply(weights_init_kaiming)
    angle_model.apply(weights_init_kaiming)
    discriminator.apply(weights_init_kaiming)

    generator = nn.DataParallel(generator, device_ids=device_ids).cuda()
    angle_model = nn.DataParallel(angle_model, device_ids=device_ids).cuda()
    discriminator = nn.DataParallel(discriminator, device_ids=device_ids).cuda()
    if opt.reset == 1:
        generator = init_model(generator)
        angle_model = init_model(angle_model)
        discriminator = init_model(discriminator)
    else:
        generator.load_state_dict(torch.load('/home/ubuntu/2TB/YCF/2021YCF_ACM/Decomposition_UTV/output/model2_single_streak/generator_backup.pth'))
        angle_model.load_state_dict(torch.load('/home/ubuntu/2TB/YCF/2021YCF_ACM/UTV_Decomposition/output/train_Decomposition_angle_model/angle_model_backup.pth'))
        discriminator.load_state_dict(torch.load('/home/ubuntu/2TB/YCF/2021YCF_ACM/Decomposition_UTV/output/model2_single_streak/discriminator_backup.pth'))

    fake_pool = ImagePool(50)

    class_loss = GANLoss('lsgan')
    Prior_criterion1 = nn.L1Loss(reduction='sum')
    Prior_criterion2 = TV_Loss()
    data_criterion2 = nn.MSELoss(size_average=False)

    Prior_criterion1.cuda()
    Prior_criterion2.cuda()
    data_criterion2.cuda()
    class_loss.cuda()

    # training
    output_folder = opt.outf + '/train_Decomposition_angle_model_cityscape'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    writer = SummaryWriter(output_folder)
    step = 0
    current_lr = opt.lr

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lr,eps=1e-4,amsgrad=True)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, eps=1e-4, amsgrad=True)
    angle_optimizer = torch.optim.Adam(angle_model.parameters(), lr=opt.lr, eps=1e-4, amsgrad=True)

    d_schedulr = torch.optim.lr_scheduler.MultiStepLR(d_optimizer,milestones=[50,70,100],gamma=0.1,last_epoch = -1)
    g_schedulr = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=[50, 70, 100], gamma=0.1, last_epoch=-1)
    angle_schedulr = torch.optim.lr_scheduler.MultiStepLR(angle_optimizer, milestones=[10, 40, 50], gamma=0.1, last_epoch=-1)
    discriminator_total_loss = 0
    for epoch in range(opt.epochs):
        D_loss = 0
        G_loss = 0
        A_loss = 0
        for i, data in enumerate(loader_train):  # 可同时获得索引和值

            #angle model train
            rain2, real_image,angle_label = data['rain_data'].type(torch.FloatTensor).cuda(), \
                                data['clean_data'].type(torch.FloatTensor).cuda(),\
                                data['angle_label'].type(torch.FloatTensor).cuda()

            out_angle = angle_model(rain2)
            angle_loss = data_criterion2(out_angle,angle_label)
            angle_optimizer.zero_grad()
            angle_loss.backward()
            angle_optimizer.step()
            if epoch < 20:
                if step % 30 == 0 and step != 0:
                    # print("[epoch %d][%d/%d] loss: %.4f   loss_angle: %.4f" %
                    #       (epoch + 1, i + 1, len(loader_train), loss.item(), loss_angle.item()))
                    # print("Iter: {}, D: {:.4}, G:{:.4}".format(iter_count, d_total_error.item(), g_error.item()))
                    print("[epoch %d][%d/%d], angle_loss: %f " %
                        (epoch + 1, i + 1, len(loader_train), angle_loss.item()))
                    writer.add_scalar('angle loss', angle_loss.item(), step)

                    if step % 300 == 0:
                        #     # results
                        with torch.no_grad():
                            out_angle_transform = transform_crop(rain2,out_angle)
                            label_angle_transform = transform_crop(rain2,angle_label)
                            Img_out_angle = utils.make_grid(out_angle_transform.data, nrow=8, normalize=True, scale_each=True)
                            Img_label_angle = utils.make_grid(label_angle_transform.data, nrow=8, normalize=True, scale_each=True)

                            writer.add_image('label angle image', Img_label_angle, epoch * repeat + step)
                            writer.add_image('out angle image', Img_out_angle, epoch * repeat + step)
            else:
                """generator train"""
                generator.train()
                set_requires_grad([discriminator],False)
                transform_rain = transform_crop(rain2,out_angle)
                out_put = generator(transform_rain)
                fake_images, Out_R = out_put['out_clean'],\
                                            out_put['rain_streak']
                reconstruct_image = fake_images + Out_R
                w_x = fake_images.size()[3]
                h_x = fake_images.size()[2]

                #reconstruct loss
                w1 = 0.1
                consistencey_loss = data_criterion2(reconstruct_image,transform_rain)

                # rain UTV Loss
                w2 = 1.5
                w3 = 1
                w4 = 0.1
                data_zero = torch.zeros_like(Out_R)
                Gradient_y_OutR = gradient_y(Out_R)
                gradient_zero = torch.zeros_like(Gradient_y_OutR)
                R_prior1 = Prior_criterion1(gradient_y(Out_R),gradient_zero)  #雨条层竖直方向梯度尽可能小
                R_prior2 = Prior_criterion1(gradient_x(transform_rain),gradient_x(Out_R))   #雨条层的垂直方向梯度与退化图像的垂直方向梯度尽可能相似
                R_prior3 = Prior_criterion1(Out_R,data_zero)
                streak_total_loss = w2*R_prior1 + w3*R_prior2 + w4*R_prior3

                #Clean image loss
                w5 = 0.01
                w6 = 0.01
                w7 = 4e3
                Clean_prior = data_criterion2(fake_images,transform_rain)
                Tv_loss = Prior_criterion2(fake_images)
                #adv loss
                gen_logits_fake = discriminator(fake_images)
                GAN_loss = class_loss(gen_logits_fake, True)
                Image_total_loss = w5*Clean_prior + w6*Tv_loss + w7*GAN_loss

                generator_total_loss = w1*consistencey_loss + 0.1*streak_total_loss + 0.1*Image_total_loss
                g_optimizer.zero_grad()
                generator_total_loss.backward()
                g_optimizer.step()

                """discriminator train"""
                # real data
                transform_real_image = transform_crop(real_image,out_angle)
                set_requires_grad([discriminator], True)
                real_data = Variable(transform_real_image)
                # transform_real_data = transform(real_data)
                logits_real = discriminator(real_data)

                # fake data
                # fake_images = generator(rain2)
                fake = fake_pool.query(fake_images)
                logits_fake = discriminator(fake.detach())


                #discriminator loss
                discriminator_total_loss = class_loss(logits_real, True) + class_loss(logits_fake, False)
                d_optimizer.zero_grad()
                discriminator_total_loss.backward()
                d_optimizer.step()




                if step % 30 == 0 and step != 0:
                    # print("[epoch %d][%d/%d] loss: %.4f   loss_angle: %.4f" %
                    #       (epoch + 1, i + 1, len(loader_train), loss.item(), loss_angle.item()))
                    # print("Iter: {}, D: {:.4}, G:{:.4}".format(iter_count, d_total_error.item(), g_error.item()))
                    print("[epoch %d][%d/%d], angle_loss:%f,,D_loss: %f, consistencey_loss:%f,streak_total_loss:%f, Image_total_loss:%f,generator_total_loss:%f " %
                          (epoch + 1, i + 1, len(loader_train),\
                           angle_loss.item(),\
                           discriminator_total_loss.item(),\
                           consistencey_loss.item(),\
                           streak_total_loss.item(),\
                           Image_total_loss.item(),\
                           generator_total_loss.item()))
                    writer.add_scalar('discriminator_total_loss', discriminator_total_loss.item(), step)
                    writer.add_scalar('angle loss', angle_loss.item(), step)
                    writer.add_scalar('R_prior1_loss', R_prior1.item(), step)
                    writer.add_scalar('R_prior2_loss', R_prior2.item(), step)
                    writer.add_scalar('R_prior3_loss', R_prior2.item(), step)
                    writer.add_scalar('Clean_prior_loss', Clean_prior.item(), step)
                    writer.add_scalar('Tv_loss', Tv_loss.item(), step)
                    writer.add_scalar('GAN_loss', GAN_loss.item(), step)
                    writer.add_scalar('consistencey_loss', consistencey_loss.item(), step)
                    writer.add_scalar('streak_total_loss', streak_total_loss.item(), step)
                    writer.add_scalar('Image_total_loss', Image_total_loss.item(), step)
                    writer.add_scalar('generator_total_loss', generator_total_loss.item(), step)
                    if step % 300 == 0:
                        #     # results
                        with torch.no_grad():
                            out2_train = torch.clamp(fake_images, 0., 1.)
                            out_mask_all = torch.clamp(Out_R, 0., 1.)
                            Img_rain = utils.make_grid(rain2.data, nrow=8, normalize=True, scale_each=True)
                            Img_clean = utils.make_grid(real_data.data, nrow=8, normalize=True, scale_each=True)
                            Img_out_img_all = utils.make_grid(out2_train.data, nrow=8, normalize=True, scale_each=True)
                            Img_out_mask_all = utils.make_grid(out_mask_all.data, nrow=8, normalize=True, scale_each=True)

                            writer.add_image('noisy image', Img_rain, epoch * repeat + step)
                            writer.add_image('clean image', Img_clean, epoch*repeat+step)
                            writer.add_image('reconstructed image2', Img_out_img_all, epoch * repeat + step)
                            writer.add_image('reconstructed all mask', Img_out_mask_all, epoch * repeat + step)
                        # torch.cuda.empty_cache()
                # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]

            step += 1

        d_schedulr.step()
        g_schedulr.step()
        angle_schedulr.step()

        if epoch % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(output_folder, 'generator_epoch_' + str(epoch) + '.pth'))
            torch.save(discriminator.state_dict(), os.path.join(output_folder, 'discriminator_epoch_' + str(epoch) + '.pth'))
            torch.save(angle_model.state_dict(),os.path.join(output_folder, 'angle_model_epoch_' + str(epoch) + '.pth'))
        torch.save(generator.state_dict(), os.path.join(output_folder, 'generator_backup.pth'))
        torch.save(discriminator.state_dict(), os.path.join(output_folder, 'discriminator_backup.pth'))
        torch.save(angle_model.state_dict(), os.path.join(output_folder, 'angle_model_backup.pth'))
if __name__ == "__main__":
    main()