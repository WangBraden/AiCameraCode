import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import  SubsetRandomSampler
import matplotlib.pyplot as plt

from thermal_dataloader import ThermalImageDataset
import models

# the normalization factors mean and std should be calculated by mean.py
txfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.44,),(0.1,),)])
#txfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0297,),(0.1375,),)])

class0 = ThermalImageDataset('labels.txt', 'Photos', transform=txfm, classes=0)
train_size = int(0.6 * len(class0))
valid_size = int(0.2 * len(class0))
test_size = len(class0) - train_size - valid_size
train0, valid0, test0 = torch.utils.data.random_split(class0, [train_size, valid_size, test_size])

class1 = ThermalImageDataset('labels.txt', 'Photos', transform=txfm, classes=1)
train_size = int(0.6 * len(class1))
valid_size = int(0.2 * len(class1))
test_size = len(class1) - train_size - valid_size
train1, valid1, test1 = torch.utils.data.random_split(class1, [train_size, valid_size, test_size])

train_set = torch.utils.data.ConcatDataset([train0, train1])
valid_set = torch.utils.data.ConcatDataset([valid0, valid1])
test_set = torch.utils.data.ConcatDataset([test0, test1])

trainloader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=4)
validloader = torch.utils.data.DataLoader(valid_set, shuffle=True, batch_size=4)
testloader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=16)

"""
dataset = ThermalImageDataset('labels.txt', 'Photos', transform=txfm)

#Preparing for validaion test
indices = list(range(len(dataset)))
np.random.shuffle(indices)

# split the samples into train (60%), validate (20%) and test sets (20%)
valid_pct = int(np.floor(0.2 * len(dataset)))
train_split = int(np.floor(0.6 * len(dataset)))
valid_end = train_split + valid_pct
train_sample = SubsetRandomSampler(indices[:train_split])
valid_sample = SubsetRandomSampler(indices[train_split:valid_end])
test_sample = SubsetRandomSampler(indices[valid_end:])

trainloader = torch.utils.data.DataLoader(dataset, sampler=train_sample, batch_size=4)
validloader = torch.utils.data.DataLoader(dataset, sampler=valid_sample, batch_size=4)
testloader = torch.utils.data.DataLoader(dataset, sampler=test_sample, batch_size=16)
"""

def show_pics():
    dataiter = iter(trainloader)
    print(dataiter)
    images, labels = dataiter.next()

    fig = plt.figure(figsize=(15,5))
    for idx in np.arange(10):
        # xticks=[], yticks=[] is empty to print the images without any ticks around them
        #np.sqeeze : Remove single-dimensional entries from the shape of an array.
        ax = fig.add_subplot(4, 20//4, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        # .item() gets the value contained in a Tensor
        ax.set_title(labels[idx].item())
    fig.tight_layout()


def train(model, loader, f_loss, optimizer, device):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()

    for images, labels in loader:
        # We need to copy the data on the GPU if we use one
        inputs, targets = images.to(device), labels.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

def test(model, loader, f_loss, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation 

    Returns :

        A tuple with the mean loss and mean accuracy

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        for imgs, labels in loader:
            # We need to copy the data on the GPU if we use one
            inputs, targets = imgs.to(device), labels.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()
        return tot_loss/N, correct/N

# move the model to GPU if possible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#model = models.Classifier()
model = models.ThermalCNN(2)
model.to(device)
checkpoint = models.ModelCheckpoint('model.pt', model)

#defining the loss function and optimizer
#f_loss = nn.NLLLoss()
f_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 30

for t in range(epochs):
    print("Epoch {}".format(t))
    train(model, trainloader, f_loss, optimizer, device)

    val_loss, val_acc = test(model, validloader, f_loss, device)
    print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))
    checkpoint.update(val_loss)

# load the best model
model = models.ThermalCNN(2)
model.load_state_dict(torch.load('model.pt'))

#export model
dummy_input = torch.randn(1, 1, 24, 32)
torch.onnx.export(model, dummy_input, "model.onnx")

#track the test loss
test_loss = 0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

with torch.no_grad():
    model.eval()
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        #forword pass 
        output = model(images)
        #calculate the loss
        loss = f_loss(output, labels)
        #update the test loss
        test_loss += loss.item()*images.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        #compare predictions to the true labes
        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
        #calculate test accuracy for each object class
        for i in range(len(labels)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] +=1

#calcaulate and prınt test loss
test_loss = test_loss/len(testloader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

classes = [0, 1]

for i in range(2):
  if class_total[i] > 0:
    print('Test Accuracy of %5s: %2d%% (%2d/%2d)'%
          (str(i), 100 * class_correct[i]/class_total[i],
           np.sum(class_correct[i]), np.sum(class_total[i])))
  else:
    print('Test Accuracy of %5s: N/A(no training examples)' % classes[i])

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

def plot_result():
    # obtain one batch of test images
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds = torch.max(output, 1)
    # prep images for display
    images = images.numpy()

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        if idx < len(images):
            ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(images[idx]), cmap='gray')
            ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                        color=("green" if preds[idx]==labels[idx] else "red"))
    fig.tight_layout()
    plt.waitforbuttonpress()

plot_result()