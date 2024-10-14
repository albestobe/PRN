import os

from PIL import Image
from torch.utils import data
from torchvision import transforms
from models.common import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])


class MSRS_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得TNO数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'bright':
                self.bright_path = temp_path  # 获得红外路径
            if sub_dir == 'cartoonresult':
                self.cartoon_path = temp_path
            if sub_dir == 'originalmask':
                self.mask_path = temp_path
            if sub_dir == 'originalimage':
                self.gt_path = temp_path


        self.name_list = os.listdir(self.bright_path )  # 获得子目录下的图片的名称
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        bright_image = Image.open(os.path.join(self.bright_path, name))  # 获取红外图像
        cartoon_image = Image.open(os.path.join(self.cartoon_path, name))
        mask=Image.open(os.path.join(self.mask_path, name))
        gt_image=Image.open(os.path.join(self.gt_path, name))

        bright_image= self.transform(bright_image)
        cartoon_image  = self.transform(cartoon_image )
        mask=self.transform(mask)
        mask=mask[0:1,:,:]
        gt_image=self.transform(gt_image)


        # vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return bright_image, cartoon_image, mask, gt_image ,name

    def __len__(self):
        return len(self.name_list)
