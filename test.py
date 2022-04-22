import torch
import torch.nn as nn
from preprocessing.physics_dataloader import PhysicsDataset  # Dataloader for your physical data 
from torch.utils.data import DataLoader
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time 
import json
from models.mlp import MLP # Importing your model 
from utils.utils import seed_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_all(42)

# Testing Random Data (will be replaced tomorrow by your data)

x_train = np.random.randn(100,886)
y_train = np.random.randn(100,3)
x_test = np.random.randn(100,886)
y_test = np.random.randn(100,3)

#DataLoader 

traindata =PhysicsDataset(x_train, y_train,'train')
train_loader = DataLoader(
        traindata,
        batch_size = 16,
        shuffle = True
)


validdata = PhysicsDataset(x_train, y_train, 'valid')
valid_loader = DataLoader(
        validdata,
        batch_size = 16,
        shuffle = True
)


testdata = PhysicsDataset(x_test, y_test, 'test')
test_loader = DataLoader(
        testdata,
        batch_size = 16,
        shuffle = False
)

# Building the mlp model
mlp = MLP().to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.MSELoss()


model_time = time.strftime("%d%b%Y-%Hh%Mm%Ss")
model_folder_name = f"MLP_{model_time}"
checkpoint_filepath = "checkpoints/{}.pt".format(model_folder_name)
writer = SummaryWriter('torch_logs/{}'.format(model_folder_name))

print("Start Training Loop \n")
epochs = 200

#Training Function 
def train(train_loader, valid_loader, verbose):
    
    best_valid_loss = np.inf
    improvement_ratio = 0.005
    num_steps_wo_improvement = 0
    
    for epoch in range(epochs):
        nb_batches_train = len(train_loader)
        mlp.train()
        losses = 0.0
        total_x = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            out= mlp(x)
            loss = criterion(out, y) 
            
            mlp.zero_grad() 

            loss.backward() 
            losses += loss.item()

            optimizer.step()        
            total_x += x.size(0)
            
            if verbose > 0:
                if (i+1) % verbose == 0:
                    print(f"Update Step: | Train_Loss {(losses / total_x):.3f} ")
        
            
        writer.add_scalar('training loss',
            losses / nb_batches_train,
            epoch + 1)
        
            
        print(f"Epoch {epoch}: | Train_Loss {(losses / nb_batches_train):.3f}  ")
        
        valid_loss=evaluate(valid_loader)
        writer.add_scalar('validation loss',
                          valid_loss,
                          epoch + 1)
        
        
        if (best_valid_loss - valid_loss) > np.abs(best_valid_loss * improvement_ratio):
            num_steps_wo_improvement = 0
        else:
            num_steps_wo_improvement += 1
            
        if num_steps_wo_improvement == 7:
            print("Early stopping on epoch:{}".format(str(epoch)))
            break;
            
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss  
            torch.save({
            'epoch': epoch,
            'model_state_dict': mlp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'LOSS': losses / nb_batches_train,
            }, checkpoint_filepath)

   
 
#Evaluation Function
def evaluate(data_loader):
    nb_batches = len(data_loader)
    val_losses = 0.0
    with torch.no_grad():
        mlp.eval()
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            out = mlp(x)
            val_loss = criterion(out, y) 
            val_losses += val_loss.item()
            
            

    print(f"Validation_Loss {val_losses / nb_batches} \n")
    return val_losses / nb_batches

#Testing Function 

def test(data_loader, model):
    with torch.no_grad():
        model.eval()
        step = 0
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            out = mlp(x)
            
            if(step == 0):
                pred = out
                labels = y

            else:
                pred = torch.cat((pred,out), 0)
                labels = torch.cat((labels, y), 0)
            step +=1

    return pred, labels

#Training
train(train_loader, valid_loader, 2000)

#Testing
model_params=dict()
test_model = MLP().to(device)
checkpoint = torch.load(checkpoint_filepath)
test_model.load_state_dict(checkpoint['model_state_dict'])
pred, lab = test(train_loader, test_model)
pred = pred.cpu()
lab = lab.cpu()
train_mse=criterion(pred,lab)
model_params['train_mse'] = train_mse.item()

pred, lab = test(valid_loader, test_model)
pred = pred.cpu()
lab = lab.cpu()
valid_mse=criterion(pred,lab)
model_params['valid_mse'] = valid_mse.item()

pred, lab = test(test_loader, test_model)
pred = pred.cpu()
lab = lab.cpu()
test_mse=criterion(pred,lab)
model_params['test_mse'] = test_mse.item()


model_params['model_name'] = model_folder_name
config = json.dumps(model_params)

f = open("checkpoints/{}.json".format(model_folder_name),"w")
f.write(config)
f.close()


