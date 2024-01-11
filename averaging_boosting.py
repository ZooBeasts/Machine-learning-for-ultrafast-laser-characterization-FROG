from Resnet import *
import pandas as pd
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import matplotlib.pyplot as plt

'''
this is the python file for prediction of concatenated data
averaging method to predict the real and imag data from SHG and THG frog images
step 1: load the model, SHG and THG for real and imag data
step 2: load each shg_real and shg_imag, thg_real and thg_imag model
step 3: predict the real and imag data from each model 
step 4: average the real and imag data from each model
 such as (shg_real,thg_real)/2 and (shg_imag,thg_imag)/2
step 5: save the data to csv file
'''

shg_real_path = 'E:/FROGS/logs/shg240.pt'
shg_imag_path = 'E:/FROGS/logs/shg_imag170.pt'
thg_real_path = 'E:/FROGS/logs/thg340.pt'
thg_imag_path = 'E:/FROGS/logs/thg_imag290.pt'
image_path_shg = 'E:/FROGS/test_shg_test/'
image_path_thg = 'E:/FROGS/test_thg_test/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class averaging_boosting:
    def __init__(self):
        self.shg_real_path = shg_real_path
        self.thg_real_path = thg_real_path
        self.shg_imag_path = shg_imag_path
        self.thg_imag_path = thg_imag_path
        self.image_path_shg = image_path_shg
        self.image_path_thg = image_path_thg
        self.transform = transforms
        self.device = device
        self.shg_real_model = self.load_shg_real_model()
        self.thg_real_model = self.load_thg_real_model()
        self.shg_imag_model = self.load_shg_imag_model()
        self.thg_imag_model = self.load_thg_imag_model()
        self.shg_real, self.shg_imag = self.shg_outputs()
        self.thg_real, self.thg_imag = self.thg_outputs()

    print('Starting load model')
    '''
     here we load the model for shg_real and thg_real, thg_imag and shg_imag model end with .pt
        param model_path: os.path
        return model to(device)
    '''

    def load_shg_real_model(self):
        if self.shg_real_path.endswith('.pt'):
            return torch.load(self.shg_real_path).to(self.device)
        else:
            raise ValueError('Unknown shg real model file: {}'.format(self.shg_real_path))

    def load_thg_real_model(self):
        if self.thg_real_path.endswith('.pt'):
            return torch.load(self.thg_real_path).to(self.device)
        else:
            raise ValueError('Unknown thg real model file: {}'.format(self.thg_real_path))

    def load_shg_imag_model(self):
        if self.shg_imag_path.endswith('.pt'):
            return torch.load(self.shg_imag_path).to(self.device)
        else:
            raise ValueError('Unknown shg imag model file: {}'.format(self.shg_imag_path))

    def load_thg_imag_model(self):
        if self.thg_imag_path.endswith('.pt'):
            return torch.load(self.thg_imag_path).to(self.device)
        else:
            raise ValueError('Unknown thg imag model file: {}'.format(self.thg_imag_path))

    print('-------------------')
    print('Four models loaded')
    print('-------------------')

    @staticmethod
    def _process_image_shg(image_path_shg):
        img = Image.open(image_path_shg).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        img_SHG = img.unsqueeze(0)
        return img_SHG

    @staticmethod
    def _process_image_thg(image_path_thg):
        img = Image.open(image_path_thg).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        img_THG = img.unsqueeze(0)
        return img_THG

    def shg_outputs(self):

        for filename in os.listdir(self.image_path_shg):
            file_path = os.path.join(self.image_path_shg, filename)
            image_tensor = averaging_boosting._process_image_shg(file_path).to(device)
            with torch.no_grad():
                shg_real = self.shg_real_model.eval()(image_tensor).squeeze(0).cpu().detach().numpy()
                shg_imag = self.shg_imag_model.eval()(image_tensor).squeeze(0).cpu().detach().numpy()
                return shg_real, shg_imag

    def thg_outputs(self):

        for filename in os.listdir(self.image_path_thg):
            file_path = os.path.join(self.image_path_thg, filename)
            image_tensor = averaging_boosting._process_image_thg(file_path).to(device)
            with torch.no_grad():
                thg_real = self.thg_real_model.eval()(image_tensor).squeeze(0).cpu().detach().numpy()
                thg_imag = self.thg_imag_model.eval()(image_tensor).squeeze(0).cpu().detach().numpy()
                return thg_real, thg_imag

    def averaging(self):
        shg_real, shg_imag = self.shg_outputs()
        thg_real, thg_imag = self.thg_outputs()
        real = 0.5 * (shg_real + thg_real)
        imag = 0.5 * (shg_imag + thg_imag)
        return real, imag

    def plot_averaged_output(self, plot=False, save=False):

        real, imag = self.averaging()
        wl = np.arange(1, 201, 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        original_real = pd.read_csv('dataset/concatenated_Ereal.csv', header=None).values[9000:9001, 155:355]
        original_imag = pd.read_csv('dataset/concatenated_Eimag.csv', header=None).values[9000:9001, 155:355]
        ax1.plot(wl, real.reshape(200, 1), '--r', label='pred')
        ax1.plot(wl, original_real.reshape(200, 1), 'b', label='real')
        ax1.set_xlabel('Wavelength nm')
        ax1.set_ylabel('real value')
        ax1.set_title('Prediction vs Real - Real Part')
        ax1.legend()

        ax2.plot(wl, imag.reshape(200, 1), '--r', label='pred')
        ax2.plot(wl, original_imag.reshape(200, 1), 'b', label='imag')
        ax2.set_xlabel('Wavelength nm')
        ax2.set_ylabel('imaginary value')
        ax2.set_title('Prediction vs Real - Imaginary Part')
        ax2.legend()

        if plot:
            plt.show()

        if save:
            fig.savefig('averaged_output.png')
            real, imag = self.averaging()
            real, imag = pd.DataFrame(real), pd.DataFrame(imag)
            real.to_csv('averaged_real.csv', index=False, header=False)
            imag.to_csv('averaged_imag.csv', index=False, header=False)

    print('Prediction finished')
    print('Hold on for plotting')
    print('averaged_output.png saved')
    print('averaged_real.csv and averaged_imag.csv saved')


if __name__ == '__main__':
    averaging_boosting().plot_averaged_output(plot=True, save=False)
