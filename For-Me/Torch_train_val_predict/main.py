import torch
import torch.nn as nn

class Model(nn.Module):

  def __init__(self, model, loss_function, optimizer):
    super(Model, self).__init__()
    self.model = model
    self.loss_fucntion = loss_function
    self.optimizer = optimizer
    self.epoch_loss = dict()


  def train(self, train_loader, val_loader = None, epochs = 10):
    for epoch in range(epochs): 
      self.model.train()
      total_loss = 0
      self.epoch_loss[epoch] = []

      for X_labels, y_labels in train_loader:
        outputs = self.model(X_labels)
        loss = self.loss_function(y_labels, outputs)
        self.optimizer.zero_grad()
        self.optimizer.backward()
        self.optimizer.step()
        total_loss += loss.step()
        self.epoch_loss[epoch].append(loss.item())
        print(f"{epoch + 1}/{epochs} epoch : /nLoss {loss.step()}\tAverage Loss : {total_loss/(epoch+1):.2f}")
        print("-" * 40)
      if val_loader:
        self.model.eval()
        val_total_loss = 0
        

      print(f"END OF EPOCH {epoch + 1} : \nAverage Loss : {total_loss/epochs:.2f}")





