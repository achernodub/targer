import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

def crf_train_loop(model, rolls, targets, n_epochs, learning_rate=0.01):

    optimizer = Adam(model.parameters(), lr=learning_rate,
                     weight_decay=1e-4)

    for epoch in range(n_epochs):
        batch_loss = []
        N = rolls.shape[0]
        model.zero_grad()
        for index, (roll, labels) in enumerate(zip(rolls, targets)):
            # Forward Pass
            neg_log_likelihood = model.neg_log_likelihood(roll, labels)
            batch_loss.append(neg_log_likelihood)
            
            if index % 50 == 0:
                ll = torch.cat(batch_loss).mean()
                ll.backward()
                optimizer.step()
                print("Epoch {}: Batch {}/{} loss is {:.4f}".format(epoch, index//50,N//50,ll.data.numpy()[0]))
                batch_loss = []
    
    return model