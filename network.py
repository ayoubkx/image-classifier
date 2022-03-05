import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        # hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:]) #input/output sizes for hidden layers
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
      
    def forward (self,n):
        for hl in self.hidden_layers:
            n = F.relu(hl(n)) # use relu function
            n = self.dropout(n) # dropouts
        n = self.output(n) # output weights
        return F.log_softmax(n,dim = 1) # apply activation log softmax
    
#accuracy and  loss checking 
def check_accuracy_loss(model, loader, criterion, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"
    
    model.eval()
    accuracy = 0
    loss = 0
    
    with torch.no_grad():
        for images,labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            prob = torch.exp(outputs)
            results = (labels.data == prob.max(1)[1]) # labels that give prediction (the highest probability gives prediction)
            
            accuracy += results.type_as(torch.FloatTensor()).mean()
            loss += criterion(outputs,labels)
            
    # we need to get the avg over all the input images
   
    return accuracy/len(loader),loss/len(loader)


# TRAINING THE NETWORK
def train_network(model, trainloader, validloader, epochs, print_every, criterion, optimizer, scheduler, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"
    
    steps = 0
    model.to(device)
    model.train()
    
    for e in range(epochs):
        scheduler.step()
        running_loss = 0
        for x , (inputs,labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            #from udacity exercice notebooks
            if steps % print_every == 0:
                accuracy,valid_loss = check_accuracy_loss(model,validloader,criterion,gpu)
                           
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(valid_loss),
                      "Validation Accuracy: {:.4f}".format(accuracy))
                running_loss = 0
            model.train()
            
            
          
            
    
   

    
    

    