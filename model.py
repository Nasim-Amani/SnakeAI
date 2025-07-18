import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size , ouput_size):
        super().__init__()
        self.linear1= nn.Linear(input_size,hidden_size)
        self.linear2= nn.Linear(hidden_size,ouput_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x  

    def save(self, file_name='model.pth'):
      model_folder='./model'
      if not os.path.exists(model_folder):
          os.makedirs(model_folder)
      file_name = os.path.join(model_folder, file_name)    
      torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr=lr
        self.gamma= gamma
        self.model=model
        self.optimizer=optim.Adam(model.parameters(), lr=self.lr)
        self.criterion= nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        #(n,x)

        if len(state.shape) == 1:
            #(1,x)
            state = torch.unsqueeze(state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            next_state = torch.unsqueeze(next_state,0)
            game_over=(game_over,)


        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()    


       