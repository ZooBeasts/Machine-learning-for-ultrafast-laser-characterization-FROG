import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Resnet import *
from averaging_boosting import averaging_boosting
from sklearn.metrics import r2_score, confusion_matrix


all_shg_image_path = 'shg_test'
all_thg_image_path = 'thg_test'
all_shg_real_data = 'dataset/shg_real_test.csv'
all_thg_real_data = 'dataset/thg_real_test.csv'
all_shg_imag_data = 'dataset/shg_imag_test.csv'
all_thg_imag_data = 'dataset/thg_imag_test.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class output_statistics(averaging_boosting):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (self.eval_shg_imag_model, self.eval_shg_real_model,
         self.eval_thg_imag_model, self.eval_thg_real_model) = self.__load_model_eval()
        self.all_shg_image_path = all_shg_image_path
        self.all_thg_image_path = all_thg_image_path
        self.all_shg_real_data = all_shg_real_data
        self.all_thg_real_data = all_thg_real_data
        self.all_shg_imag_data = all_shg_imag_data
        self.all_thg_imag_data = all_thg_imag_data
        self.selected_shg_imag_predictions, self.shg_imag_values = self._get_shg_imag_outputs()
        self.selected_shg_real_predictions, self.shg_real_values = self._get_shg_real_outputs()
        self.selected_thg_imag_predictions, self.thg_imag_values = self._get_thg_imag_outputs()
        self.selected_thg_real_predictions, self.thg_real_values = self._get_thg_real_outputs()
        self.wl = np.arange(1, 201, 1)

    def __load_model_eval(self):
        self.eval_shg_imag_model = self.load_shg_imag_model().eval()
        self.eval_shg_real_model = self.load_shg_real_model().eval()
        self.eval_thg_imag_model = self.load_thg_imag_model().eval()
        self.eval_thg_real_model = self.load_thg_real_model().eval()
        return self.eval_shg_imag_model, self.eval_shg_real_model, self.eval_thg_imag_model, self.eval_thg_real_model

    def _get_shg_real_outputs(self, plot=False):
        shg_predictions = []
        shg_file_names = []
        shg_real_values = []
        real_values_array = pd.read_csv(all_shg_real_data, header=None)
        real_values_all = real_values_array.iloc[:, 155:355]

        for filename in os.listdir(all_shg_image_path):
            file_path = os.path.join(all_shg_image_path, filename)
            image_tensor = averaging_boosting._process_image_shg(file_path).to(device)
            with torch.no_grad():
                output = self.eval_shg_real_model(image_tensor).squeeze(0).cpu().detach().numpy()
                shg_predictions.append(output)
                shg_file_names.append(filename)
                shg_real_values.append(real_values_all.values)

        selected_shg_real_predictions = []
        selected_file_names = []

        for i, (prediction, real_value) in enumerate(zip(shg_predictions, shg_real_values)):
            if (prediction <= real_value * 0.8).any():
                selected_shg_real_predictions.append(prediction)
                selected_file_names.append(shg_file_names[i])
            # print(selected_shg_real_predictions)


        if plot:
            for i, prediction in enumerate(selected_shg_real_predictions):
                plt.plot(self.wl, prediction, '--', label=f'Selected Prediction for {selected_file_names[i]}')
                print(selected_file_names[i])
                for j, real in enumerate(shg_real_values):
                    if selected_file_names[i] == selected_file_names[j]:
                        plt.plot(self.wl, real[i], label=f'Real Value for {selected_file_names[j]}')
                        print(selected_file_names[j])
                        plt.legend()
                        plt.show()
        return selected_shg_real_predictions, shg_real_values

    # selected_file_names,

    def _get_shg_imag_outputs(self, plot=False):
        shg_predictions = []
        shg_file_names = []
        shg_imag_values = []
        real_values_array = pd.read_csv(all_shg_imag_data, header=None)
        real_values_all = real_values_array.iloc[:, 155:355]

        for filename in os.listdir(all_shg_image_path):
            file_path = os.path.join(all_shg_image_path, filename)
            image_tensor = averaging_boosting._process_image_shg(file_path).to(device)
            with torch.no_grad():
                output = self.eval_shg_imag_model(image_tensor).squeeze(0).cpu().detach().numpy()
                shg_predictions.append(output)
                shg_file_names.append(filename)
                shg_imag_values.append(real_values_all.values)

        selected_shg_imag_predictions = []
        selected_shg_imag_file_names = []

        for i, (prediction, real_value) in enumerate(zip(shg_predictions, shg_imag_values)):
            if (prediction <= real_value * 0.8).any():
                selected_shg_imag_predictions.append(prediction)
                selected_shg_imag_file_names.append(shg_file_names[i])
            # print(selected_shg_real_predictions)

        if plot:
            for i, prediction in enumerate(selected_shg_imag_predictions):
                plt.plot(self.wl, prediction, '--', label=f'Selected Prediction for {selected_shg_imag_file_names[i]}')
                print(selected_shg_imag_file_names[i])
                for j, real in enumerate(shg_imag_values):
                    if selected_shg_imag_file_names[i] == selected_shg_imag_file_names[j]:
                        plt.plot(self.wl, real[i], label=f'Real Value for {selected_shg_imag_file_names[j]}')
                        print(selected_shg_imag_file_names[j])
                        plt.legend()
                        plt.show()

        return selected_shg_imag_predictions, shg_imag_values

    # selected_shg_imag_file_names,

    def _get_thg_real_outputs(self, plot=False):
        thg_predictions = []
        thg_file_names = []
        thg_real_values = []
        real_values_array = pd.read_csv(all_thg_real_data, header=None)
        real_values_all = real_values_array.iloc[:, 155:355]

        for filename in os.listdir(all_thg_image_path):
            file_path = os.path.join(all_thg_image_path, filename)
            image_tensor = averaging_boosting._process_image_shg(file_path).to(device)
            with torch.no_grad():
                output = self.eval_thg_real_model(image_tensor).squeeze(0).cpu().detach().numpy()
                thg_predictions.append(output)
                thg_file_names.append(filename)
                thg_real_values.append(real_values_all.values)

        selected_thg_real_predictions = []
        selected_thg_real_file_names = []

        for i, (prediction, real_value) in enumerate(zip(thg_predictions, thg_real_values)):
            if (prediction <= real_value * 0.8).any():
                selected_thg_real_predictions.append(prediction)
                selected_thg_real_file_names.append(thg_file_names[i])
            # print(selected_shg_real_predictions)

        if plot:
            for i, prediction in enumerate(selected_thg_real_predictions):
                plt.plot(self.wl, prediction, '--', label=f'Selected Prediction for {selected_thg_real_file_names[i]}')
                print(selected_thg_real_file_names[i])
                for j, real in enumerate(thg_real_values):
                    if selected_thg_real_file_names[i] == selected_thg_real_file_names[j]:
                        plt.plot(self.wl, real[i], label=f'Real Value for {selected_thg_real_file_names[j]}')
                        print(selected_thg_real_file_names[j])
                        plt.legend()
                        plt.show()

        return selected_thg_real_predictions, thg_real_values

    # selected_thg_real_file_names,

    def _get_thg_imag_outputs(self, plot=False):
        thg_predictions = []
        thg_file_names = []
        thg_imag_values = []
        real_values_array = pd.read_csv(all_thg_imag_data, header=None)
        real_values_all = real_values_array.iloc[:, 155:355]

        for filename in os.listdir(all_thg_image_path):
            file_path = os.path.join(all_thg_image_path, filename)
            image_tensor = averaging_boosting._process_image_shg(file_path).to(device)
            with torch.no_grad():
                output = self.eval_thg_imag_model(image_tensor).squeeze(0).cpu().detach().numpy()
                thg_predictions.append(output)
                thg_file_names.append(filename)
                thg_imag_values.append(real_values_all.values)

        selected_thg_imag_predictions = []
        selected_thg_imag_file_names = []

        for i, (prediction, real_value) in enumerate(zip(thg_predictions, thg_imag_values)):
            if (prediction <= real_value * 0.8).any():
                selected_thg_imag_predictions.append(prediction)
                selected_thg_imag_file_names.append(thg_file_names[i])
            # print(selected_shg_real_predictions)

        if plot:
            for i, prediction in enumerate(selected_thg_imag_predictions):
                plt.plot(self.wl, prediction, '--', label=f'Selected Prediction for {selected_thg_imag_file_names[i]}')
                print(selected_thg_imag_file_names[i])
                for j, real in enumerate(thg_imag_values):
                    if selected_thg_imag_file_names[i] == selected_thg_imag_file_names[j]:
                        plt.plot(self.wl, real[i], label=f'Real Value for {selected_thg_imag_file_names[j]}')
                        print(selected_thg_imag_file_names[j])
                        plt.legend()
                        plt.show()

        return selected_thg_imag_predictions, thg_imag_values

    # selected_shg_imag_file_names,

    def get_mse(self):

        shg_real_predictions = np.array(self.selected_shg_real_predictions)
        shg_real_values = np.array(self.shg_real_values)
        mse1 = np.mean(np.square(shg_real_predictions - shg_real_values))
        mae1 = np.mean(np.abs(shg_real_predictions - shg_real_values))
        print(f"Mean Squared Error (MSE) of shg_real is: {mse1}")
        print(f"Mean Absolute Error (MAE)of shg_real is: {mae1}")
        print('-----------------------------------------------')

        shg_imag_predictions = np.array(self.selected_shg_imag_predictions)
        shg_imag_values = np.array(self.shg_imag_values)
        mse2 = np.mean(np.square(shg_imag_predictions - shg_imag_values))
        mae2 = np.mean(np.abs(shg_imag_predictions - shg_imag_values))

        print(f"Mean Squared Error (MSE) of shg_imag is: {mse2}")
        print(f"Mean Absolute Error (MAE)of shg_imag is: {mae2}")
        print('-----------------------------------------------')

        thg_real_predictions = np.array(self.selected_thg_real_predictions)
        thg_real_values = np.array(self.thg_real_values)
        mse3 = np.mean(np.square(thg_real_predictions - thg_real_values))
        mae3 = np.mean(np.abs(thg_real_predictions - thg_real_values))

        print(f"Mean Squared Error (MSE) of thg_real is: {mse3}")
        print(f"Mean Absolute Error (MAE)of thg_real is: {mae3}")
        print('-----------------------------------------------')

        thg_imag_predictions = np.array(self.selected_thg_imag_predictions)
        thg_imag_values = np.array(self.thg_imag_values)
        mse4 = np.mean(np.square(thg_imag_predictions - thg_imag_values))
        mae4 = np.mean(np.abs(thg_imag_predictions - thg_imag_values))

        print(f"Mean Squared Error (MSE) of thg_imag is: {mse4}")
        print(f"Mean Absolute Error (MAE)of thg_imag is: {mae4}")
        print('-----------------------------------------------')

        print(f"Mean Squared Error (MSE) of Aver_Boosting real is: {0.5 * (mse1 + mse3)}")
        print(f"Mean Absolute Error (MAE)of Aver_Boosting real is: {0.5 * (mae1 + mae3)}")
        print('-----------------------------------------------')
        print(f"Mean Squared Error (MSE) of Aver_Boosting imag is: {0.5 * (mse2 + mse4)}")
        print(f"Mean Absolute Error (MAE)of Aver_Boosting imag is: {0.5 * (mae2 + mae4)}")
        print('-----------------------------------------------')

    def get_r2_score(self):
        shg_real_predictions = np.array(self.selected_shg_real_predictions).flatten()
        shg_real_values = np.array(self.shg_real_values).flatten()

        shg_imag_predictions = np.array(self.selected_shg_imag_predictions).flatten()
        shg_imag_values = np.array(self.shg_imag_values).flatten()

        thg_real_predictions = np.array(self.selected_thg_real_predictions).flatten()
        thg_real_values = np.array(self.thg_real_values).flatten()

        thg_imag_predictions = np.array(self.selected_thg_imag_predictions).flatten()
        thg_imag_values = np.array(self.thg_imag_values).flatten()

        r2_shg_real = r2_score(shg_real_values, shg_real_predictions)
        r2_shg_imag = r2_score(shg_imag_values, shg_imag_predictions)
        r2_thg_real = r2_score(thg_real_values, thg_real_predictions)
        r2_thg_imag = r2_score(thg_imag_values, thg_imag_predictions)
        r2_aver_real = r2_score(0.5 * (shg_real_values + thg_real_values), 0.5 * (shg_real_predictions + thg_real_predictions))
        r2_aver_imag = r2_score(0.5 * (shg_imag_values + thg_imag_values), 0.5 * (shg_imag_predictions + thg_imag_predictions))

        print(f"R2 Score for SHG Real: {r2_shg_real}")
        print(f"R2 Score for SHG Imag: {r2_shg_imag}")
        print(f"R2 Score for THG Real: {r2_thg_real}")
        print(f"R2 Score for THG Imag: {r2_thg_imag}")
        print(f"R2 Score for Aver_Boosting Real: {r2_aver_real}")
        print(f"R2 Score for Aver_Boosting Imag: {r2_aver_imag}")

    def get_confusion_matrix(self, num_bins=5):
        # Example for SHG real predictions
        shg_real_predictions = np.array(self.selected_shg_real_predictions).flatten()
        shg_real_values = np.array(self.shg_real_values).flatten()

        # Binning the data
        bins = np.linspace(min(shg_real_values.min(), shg_real_predictions.min()),
                           max(shg_real_values.max(), shg_real_predictions.max()),
                           num_bins)
        binned_real_values = np.digitize(shg_real_values, bins)
        binned_predictions = np.digitize(shg_real_predictions, bins)

        conf_matrix = confusion_matrix(binned_real_values, binned_predictions)
        print("Confusion Matrix for SHG Real:\n", conf_matrix)
        print('\n')

        shg_imag_predictions = np.array(self.selected_shg_imag_predictions).flatten()
        shg_imag_values = np.array(self.shg_imag_values).flatten()

        # Binning the data
        bins = np.linspace(min(shg_imag_values.min(), shg_imag_predictions.min()),
                           max(shg_imag_values.max(), shg_imag_predictions.max()),
                           num_bins)
        binned_shg_imag_values = np.digitize(shg_imag_values, bins)
        binned_shg_imag_predictions = np.digitize(shg_imag_predictions, bins)

        conf_matrix2 = confusion_matrix(binned_shg_imag_values, binned_shg_imag_predictions)
        print("Confusion Matrix for SHG IMAG:\n", conf_matrix2)
        print('\n')

        thg_real_predictions = np.array(self.selected_thg_real_predictions).flatten()
        thg_real_values = np.array(self.thg_real_values).flatten()

        # Binning the data
        bins = np.linspace(min(thg_real_values.min(), thg_real_predictions.min()),
                           max(thg_real_values.max(), thg_real_predictions.max()),
                           num_bins)
        binned_thg_real_values = np.digitize(thg_real_values, bins)
        binned_thg_real_predictions = np.digitize(thg_real_predictions, bins)

        conf_matrix3 = confusion_matrix(binned_thg_real_values, binned_thg_real_predictions)
        print("Confusion Matrix for THG Real:\n", conf_matrix3)
        print('\n')

        thg_imag_predictions = np.array(self.selected_thg_imag_predictions).flatten()
        thg_imag_values = np.array(self.thg_imag_values).flatten()

        # Binning the data
        bins = np.linspace(min(thg_imag_values.min(), thg_imag_predictions.min()),
                           max(thg_imag_values.max(), thg_imag_predictions.max()),
                           num_bins)
        binned_thg_imag_values = np.digitize(thg_imag_values, bins)
        binned_thg_imag_predictions = np.digitize(thg_imag_predictions, bins)

        conf_matrix4 = confusion_matrix(binned_thg_imag_values, binned_thg_imag_predictions)
        print("Confusion Matrix for THG IMAG:\n", conf_matrix4)
        print('\n')


if __name__ == '__main__':
    output_statistics().get_mse()
    output_statistics().get_r2_score()
    output_statistics().get_confusion_matrix()

#
# model_path = 'E:/FROGS/logs/test350.pt'

# def load_model(model_path):
#     print('Starting load model')
#     '''
#     if the save model end with pth, use model load_state_dict
#     if the save model end with pt, use torch.load
#     :param model_path:
#     :return: Model to(device)
#     '''
#     if model_path.endswith('.pth'):
#         Model = ResNet(3, 18, BasicBlock)
#         Model.load_state_dict(torch.load(model_path)).to(device)
#     elif model_path.endswith('.pt'):
#         Model = torch.load(model_path).to(device)
#         print('Model Loaded')
#     else:
#         raise ValueError('Unknown model file: {}'.format(model_path))
#     return Model
#
#
# Model = load_model(model_path).eval()
#
# class plot_image:
#     def __init__(self, image_path = 'tests'):
#         self.image_path = image_path
#     @staticmethod
#     def process_image(image_path):
#         # Open the image
#         img = Image.open(image_path).convert('RGB')
#         transform = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         img = transform(img)
#         img = img.unsqueeze(0)
#         return img
#
#     def output(self, plot=True):
#         # outputs = []
#         for filename in os.listdir(self.image_path):
#             file_path = os.path.join(self.image_path, filename)
#             image_tensor = plot_image.process_image(file_path).to(device)
#             with torch.no_grad():
#                 output = Model(image_tensor).squeeze(0).cpu().detach().numpy()
#             # outputs.append(output)
#
#         if plot:
#             wl = np.arange(1, 201, 1)
#             op1 = pd.read_csv('dataset/thg_imag_test.csv', header=None).values[0:1, 155:355]
#             plt.plot(wl, output.reshape(200, 1), '--r', label='pred')
#             plt.plot(wl, op1.reshape(200, 1), 'b', label='real')
#             plt.xlabel('Wavelength nm')
#             plt.ylabel('Transmission')
#             plt.title('Prediction vs Real')
#             plt.legend()
#             plt.show()
#


