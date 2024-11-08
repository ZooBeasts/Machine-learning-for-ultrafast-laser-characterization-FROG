import pandas as pd
from torch.utils.data import DataLoader
from Dataholder import FROGdata
from Model import resnet9_with_attention
import torch
import os

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
try:
    print("CUDA is available: {}".format(torch.cuda.is_available()))
    print("CUDA Device Count: {}".format(torch.cuda.device_count()))
    print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))
except Exception as E:
    print('Error with Cuda', E)

num_epochs = 1000
num_output = 128
batch_size = 64
learning_rate = 1e-3

save_dir = 'logs_re_11/'
os.makedirs(save_dir) if not os.path.exists(save_dir) else print("Directory exists")


training_set = FROGdata('Ere_20k.csv',
                        'SHG/',
                        'THG/',
                        train=True)

test_set = FROGdata('Ere_20k.csv',
                    'SHG/',
                    'THG/',
                    train=False)

train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4) # windows using num_workers=None
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

model = resnet9_with_attention(6, 64, num_output).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(num_epochs, model, train_loader, test_loader, criterion, optimizer, save_dir):
    patience = 40
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_labels,_ in test_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'bestnet_test.pt'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.6f}")
            break

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

        if epoch >= 30 and epoch % 10 == 0:
            torch.save(model.state_dict(), save_dir + f'Resnet_epoch_{epoch + 1}.pt')

        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), save_dir + f'Resnet_last_epoch_{epoch}.pt')

    training_loss_df = pd.DataFrame(train_losses)
    validation_loss_df = pd.DataFrame(val_losses)
    training_loss_df.to_csv(os.path.join(save_dir, 'training_loss.csv'), index=False)
    validation_loss_df.to_csv(os.path.join(save_dir, 'validation_loss.csv'), index=False)



if __name__ == '__main__':
    train(num_epochs, model, train_loader, test_loader, criterion, optimizer, save_dir)
