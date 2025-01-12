import torch
import torch.nn as nn

class Model(nn.Module):

  def __init__(self, model, loss_function, optimizer, regression = False):
    super(Model, self).__init__()
    self.model = model
    self.loss_function = loss_function
    self.optimizer = optimizer
    self.regression = regression
    self.history = {
      'train_accuracy': [],
      'train_loss' : [],
      'val_accuracy' : [],
      'val_loss' : [],
      'epochs' : 0
    }
    self.output = []


  def train(self, train_loader, val_loader = None, epochs = 10, threashold = 0.5, n_labels = 2):
    self.threashold = threashold
    self.n_labels = n_labels
    for epoch in range(epochs): 
      self.model.train()
      train_loss = 0
      train_accuracy = 0
    

      for X_labels, y_labels in train_loader:

        #Output after prediction
        outputs = self.model(X_labels)

        #Calculating Train Accuracy and Loss for each Batch.
        if self.n_labels == 2:
          acc, loss = self.__calc_train_val_loss_acc(output=outputs, y=y_labels)


        train_accuracy += acc
        train_loss += loss




        self.optimizer.zero_grad()
        self.optimizer.backward()
        self.optimizer.step()
        # self.epoch_loss_train[epoch].append(loss.item())
        # print(f"{epoch + 1}/{epochs} epoch : /nLoss {loss.step():.2f} .... Average Loss : {total_loss/(epoch+1):.2f}")
        # print("-" * 40)
    
    accuracy = (train_accuracy/len(train_loader))
    loss = (train_loss/len(train_loader))
    self.history['train_accuracy'].append(accuracy)
    self.history['train_loss'].append(loss)



    if val_loader:
      self.model.eval()
      with torch.no_grad():
        
        val_loss = 0
        val_accuracy = 0
        for X_labels, y_labels in val_loader:

          outputs = self.model(X_labels, y_labels)
          if self.n_labels == 2:
            acc, loss = self.__calc_train_val_loss_acc(output=outputs, y=y_labels)
          val_accuracy += acc
          val_loss += loss
        

      accuracy = (val_accuracy/len(val_loader))
      loss = (val_loss/len(val_loader))
      self.history['val_accuracy'].append(accuracy)
      self.history['val_loss'].append(loss)

  def __calc_train_val_loss_acc(self, output, y):
    predicted = torch.tensor((output>self.threashold).int().type(torch.FloatTensor))
    acc = torch.mean((predicted == y).type(torch.FLoatTensor))
    loss = self.loss_function(y, output)
    return acc.item(), loss.item()

  def predict(self, test_loader):
    test_accuracy = 0
    test_loss = 0
    t_acc = 0
    t_loss = 0
    self.model.eval()
    with torch.no_grad():
      for X_labels, y_labels in test_loader:

        outputs = self.model(X_labels)

        if self.n_labels == 2:
          acc, loss = self.__calc_train_val_loss_acc(output=outputs, y=y_labels)
        t_acc += acc
        t_loss += loss
    test_accuracy = t_acc/len(test_loader)
    test_loss = t_loss/len(test_loader)




