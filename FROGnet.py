from torch.utils.data import DataLoader
from torchvision import transforms
from Resnet import *
from norm_dataset import frogdata
from itertools import zip_longest
import pandas as pd

# check the cuda and GPU is available,if not using CPU
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
try:
    torch.cuda.empty_cache()
    print("CUDA is available: {}".format(torch.cuda.is_available()))
    print("CUDA Device Count: {}".format(torch.cuda.device_count()))
    print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))
except Exception as E:
    print('There is no GPU available, using CPU instead.')
    print('Ignoring this error and comment out torch device line.')

# hyperparameters here for how many epochs and classes for regression
num_epochs = 530
number_of_classes = 160
batchsize = 32
# load the dataset train and test
dataset = frogdata('dataset/shg_real.csv',
                   'SHG/',
                   transform=transforms
                   )

test_set = frogdata('dataset/shg_real_test.csv',
                   'shg_test/',
                    transform=transforms,
                    )

train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=False)

model = ResNet(3, 18, block=BasicBlock).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
save_dir = 'E:/FROGS/logs/shg_real/'

patience = 300
best_val_loss = float('inf')
current_patience = 0
best_model_state = None
early_stop_counter = 0
# Train the model
# for epoch in range(num_epochs):
#     model.train()
#     for train_batch, val_batch in zip_longest(train_loader, test_loader):
#         if train_batch is not None:
#             train_inputs, train_labels = train_batch
#             train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
#             # Training steps
#             optimizer.zero_grad()
#             train_outputs = model(train_inputs)
#             train_loss = criterion(train_outputs, train_labels)
#             train_loss.backward()
#             optimizer.step()
#
#         if val_batch is not None:
#             val_inputs, val_labels = val_batch
#             val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
#             # Validation steps (not for evaluation purposes)
#             val_outputs = model(val_inputs)
#             val_loss = criterion(val_outputs, val_labels)
#
#     avg_val_loss = val_loss.item()
#
#     # Check for improvement in validation loss
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         early_stop_counter = 0
#     else:
#         early_stop_counter += 1
#         if early_stop_counter >= patience:
#             print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4f}")
#             break
#
#     print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss.item():.4f} - Val Loss: {val_loss.item():.4f}")
#
#     if epoch % 10 == 0 and best_model_state is not None:
#         torch.save(best_model_state, save_dir + 'best_test' + str(epoch) + '.pt')
#
#     if epoch % 10 == 0:
#         torch.save(model, save_dir + 'test' + str(epoch) + '.pt')

# #
# data = data.to(device)
# targets = targets.to(device)
# optimizer.zero_grad()
# outputs = model(data)
# loss = criterion(outputs, targets)
# loss.backward()
# optimizer.step()
#
#     print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.6f}'.format(epoch + 1, num_epochs, batch_idx + 1,
#                                                               len(dataloader), loss.item()))

# second method to train the model
for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    train_losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


        if batch_idx % 30 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.6f}'.format(epoch + 1, num_epochs, batch_idx + 1,
                                                                      len(train_loader), loss.item()))
        if epoch > 130 and epoch % 10 == 0:
            torch.save(model, save_dir + 'thg_512_real' + str(epoch) + '.pt')

        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_train_loss:.6f}')
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    avg_val_losses =[]
    with torch.no_grad():
        for val_batch_idx, (val_data, val_targets) in enumerate(test_loader):
            val_data = val_data.to(device)
            val_targets = val_targets.to(device)
            val_outputs = model(val_data)
            val_loss += criterion(val_outputs, val_targets).item()

    avg_val_loss = val_loss / len(test_loader)
    avg_val_losses.append(avg_val_loss)

    # Check for improvement in validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        # Save the model state when validation loss improves
        torch.save(model.state_dict(), 'best_model_real_512.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4f}")
            break

    trainloss = pd.DataFrame(train_losses)
    valloss = pd.DataFrame(avg_val_losses)
    trainloss.to_csv('shg_real_trainloss.csv', index=False, header=False)
    valloss.to_csv('shg_real_valloss.csv', index=False, header=False)

# print(f"Best model found at epoch {best_epoch}. Best validation loss: {best_val_loss:.4f}")
