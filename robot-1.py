##
import numpy as np
import torch
import constants
import configuration

class Network(torch.nn.Module):


    def __init__(self):

        super(Network, self).__init__()
   
        self.layer_1 = torch.nn.Linear(in_features=4, out_features=10, dtype=torch.float32)
        self.layer_2 = torch.nn.Linear(in_features=10, out_features=10, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(in_features=10, out_features=2, dtype=torch.float32)

    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

class Robot:
    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.model = Network()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.transitions = []  # Store transitions for training
        self.train_after_transitions = 100  # Tr
        self.paths_to_draw = []
        self.f = False
        self.new = False 

    def process_transition(self, state, action, next_state, money_remaining):
        # Store transition
        self.transitions.append((state, action, next_state))
        
        if np.linalg.norm(state - next_state) < 10:
            self.f = True
        
        # Periodically train the dynamics model using stored transitions
        if len(self.transitions) >= self.train_after_transitions:
            self.train_dynamics_model()
            self.transitions = []
            # Reset transitions after training
            self.new = True 

    def train_dynamics_model(self):

        loss_function = torch.nn.MSELoss()
        for epoch in range(100):
            for state, action, next_state in self.transitions:
                state_action = torch.tensor(np.concatenate([state, action]), dtype=torch.float32).unsqueeze(0)
                next_state_target = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                self.optimizer.zero_grad()
                predicted_next_state = self.model(state_action)
                loss = loss_function(predicted_next_state, next_state_target)
                loss.backward()
                self.optimizer.step()

    def dynamics_model(self, state, action):
       
        with torch.no_grad():
            input_tensor = torch.tensor(np.concatenate([state, action]), dtype=torch.float32).unsqueeze(0)
            predicted_state = self.model(input_tensor).squeeze(0).numpy()
        return predicted_state

    def get_next_action_type(self, state, money_remaining):
        if False:
            return 'demo'
        if self.new:
            #print("reset")
            self.new = False
            self.transitions = [] 
            #print("hi")
            #print(self.planned_actions)
            return 'reset'
        if True:
            #print("hi")
            return 'step'



    def get_next_action_training(self, state, money_remaining):

        if self.f:
            action = state - self.goal_state
            noise = np.random.normal(0, 1, action.shape) 
            action = action + noise
          
            action = np.clip(action, -constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)
            self.f = False  
        else:
            action = self.plan_action_with_cem(state)
   
        return action


    def get_next_action_testing(self, state):
        # Use CEM for planning in testing, could be without exploration
   
        action = self.plan_action_with_cem(state, explore=False)
        next_state = self.dynamics_model(state, action)
        dis = np.linalg.norm(next_state - state)
        if dis < 15:
            action = state - self.goal_state
            noise = np.random.normal(0, 1, action.shape) 
            action = action + noise
            action = np.clip(action, -constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION)
        else:
            return action
       
            
        
        return action

    def plan_action_with_cem(self, state, explore=True):
        state = np.array(state)
        best_action = np.zeros(2)  
        for iteration in range(constants.DEMOS_CEM_NUM_ITERATIONS):
            actions = np.random.uniform(-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION, (constants.DEMOS_CEM_NUM_PATHS, 2))
            if iteration > 0 and explore:  
                actions += np.random.normal(0, 1, actions.shape)
            rewards = np.zeros(constants.DEMOS_CEM_NUM_PATHS)
            for i, action in enumerate(actions):
                predicted_state = self.dynamics_model(state, action)
                rewards[i] = -np.linalg.norm(predicted_state - self.goal_state)  
            elite_idxs = rewards.argsort()[-int(constants.DEMOS_CEM_NUM_ELITES):]
            elite_actions = actions[elite_idxs]
            best_action = elite_actions.mean(axis=0)  # Use mean of elites as next action
        return best_action


