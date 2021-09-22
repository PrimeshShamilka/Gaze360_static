import model
import numpy as np
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
from Traintest.early_stopping_pytorch.pytorchtools import EarlyStopping

if __name__ == "__main__":
  config = yaml.load(open(f"{sys.argv[1]}"), Loader=yaml.FullLoader)
  readername = config["reader"]
  dataloader = importlib.import_module("reader." + readername)

  config = config["train"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["save"]["model_name"]

  # i represents the i-th folder used as the test set.
  savepath = os.path.join(config["save"]["save_path"], f"checkpoint")
  if not os.path.exists(savepath):
    os.makedirs(savepath)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  print("Read data")
  dataset = dataloader.txtload(labelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=4, header=False)
  val_dataset = dataloader.txtload(val_labelpath, imagepath, 4, shuffle=True, num_workers=4, header=False)

  print("Model building")
  net = model.GazeStatic()
  net.train()
  net.to(device)

  print("optimizer building")
  loss_op = model.PinBallLoss().cuda()
  base_lr = config["params"]["lr"]

  decaysteps = config["params"]["decay_step"]
  decayratio = config["params"]["decay"]

  optimizer = optim.Adam(net.parameters(),lr=base_lr, betas=(0.9,0.95))
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

  print("Traning")
  length = len(dataset)
  total = length * config["params"]["epoch"]
  cur = 0
  timebegin = time.time()

  early_stopping = EarlyStopping(patience=10, verbose=True)
  num_epochs = config["params"]["epoch"]+1

  with open(os.path.join(savepath, "train_log"), 'w') as outfile:
    for epoch in range(1, config["params"]["epoch"]+1):

      optimizer.zero_grad()
      model.train()  # Set model to training mode

      valid_losses = []
      avg_valid_losses = []
      running_loss = []

      # Iterate over data
      n_total_steps = len(dataset)
      for i, (data, label) in enumerate(dataset):
        # Acquire data
        data["face"] = data["face"].to(device)
        label = label.to(device)
 
        # forward
        gaze, gaze_bias = net(data)

        # loss calculation
        loss = loss_op(gaze, label, gaze_bias)
        optimizer.zero_grad()

        # backward
        loss.backward()
        optimizer.step()
        scheduler.step()
        cur += 1

        running_loss.append(loss.item())

        # print logs
        if i % 20 == 0:
          timeend = time.time()
          resttime = (timeend - timebegin)/cur * (total-cur)/3600
          log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
          print(log)
          outfile.write(log + "\n")
          sys.stdout.flush()
          outfile.flush()
          # writer.add_scalar('training_loss', np.mean(running_loss), epoch * n_total_steps + i)
          running_loss = []

      # validate the model
      model.eval()
      n_total_steps_val = len(val_dataset)
      for i, (data, label) in enumerate(val_dataset):
        # Acquire data
        data["face"] = data["face"].to(device)
        label = label.to(device)

        # forward
        gaze, gaze_bias = net(data)

        # loss calculation
        loss = loss_op(gaze, label, gaze_bias)
        optimizer.zero_grad()

        valid_losses.append(loss.item())

      valid_loss = np.average(valid_losses)
      avg_valid_losses.append(valid_loss)

      epoch_len = len(str(num_epochs))

      print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' + f'valid_loss: {valid_loss:.5f}')

      print(print_msg)

      valid_losses = []

      # writer.add_scalar('validation_loss', valid_loss, epoch * n_total_steps_val)

      # early stopping detector
      early_stopping(valid_loss, model, optimizer, epoch)
      if early_stopping.early_stop:
        print("Early stopping")
        break

      # if epoch % config["save"]["step"] == 0:
      #   torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

