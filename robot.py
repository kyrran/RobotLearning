##
import numpy as np
import torch
import constants
import constants
import matplotlib.pyplot as plt

from graphics import PathToDraw

# class Network(torch.nn.Module):

#     # The class initialisation function.
#     def __init__(self):
#         # Call the initialisation function of the parent class.
#         super(Network, self).__init__()
#         # Define the network layers. This example network has two hidden layers, each with 10 units.
#         self.layer_1 = torch.nn.Linear(in_features=4, out_features=64, dtype=torch.float32)
#         self.layer_2 = torch.nn.Linear(in_features=64, out_features=10, dtype=torch.float32)
#         self.output_layer = torch.nn.Linear(in_features=10, out_features=2, dtype=torch.float32)

#     # Function which sends some input data through the network and returns the network's output.
#     def forward(self, input):
#         layer_1_output = torch.nn.functional.relu(self.layer_1(input))
#         layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
#         output = self.output_layer(layer_2_output)
#         return output

import torch

class Network(torch.nn.Module):
    # The class initialisation function.
    def __init__(self):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        
        # Define the network layers. This network has four hidden layers before the output layer.
        self.layer_1 = torch.nn.Linear(in_features=4, out_features=64, dtype=torch.float32)
        self.layer_2 = torch.nn.Linear(in_features=64, out_features=128, dtype=torch.float32)
        self.layer_3 = torch.nn.Linear(in_features=128, out_features=64, dtype=torch.float32)
        self.layer_4 = torch.nn.Linear(in_features=64, out_features=32, dtype=torch.float32)
        self.layer_5 = torch.nn.Linear(in_features=32, out_features=16, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(in_features=16, out_features=2, dtype=torch.float32)

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))
        layer_5_output = torch.nn.functional.relu(self.layer_5(layer_4_output))
        output = self.output_layer(layer_5_output)
        return output


class Robot:
    def __init__(self, goal_state):
        self.goal_state = goal_state
        
        self.model = Network()
   
        self.paths_to_draw = []

        # planned_actions is the best sequence of actions calculated by the planning, which the robot will then execute
        self.planned_actions = np.zeros([constants.DEMOS_CEM_PATH_LENGTH,2], dtype=np.float32)
        self.plan_index = 0
        # Initialise the robot's model of the dynamics to a random neural network
        self.dynamics_model_network = Network()
        # Create a buffer of data
        self.network_input_data = torch.zeros([0, 4], dtype=torch.float32)
        self.network_label_data = torch.zeros([0, 2], dtype=torch.float32)
        # Keeping track of the number of episodes execute
        self.num_episodes = 0
        # Training data
        self.training_losses = []
        self.num_training_epochs = 0
        # Training optimiser
        self.optimiser = torch.optim.Adam(self.dynamics_model_network.parameters(), lr=0.01)
        # Graph data
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlabel='Training Epochs', ylabel='Training Loss', title='Loss Curve for Model Training')
        plt.yscale('log')
        
        self.budget = constants.STARTING_MONEY
        self.stuck = False
        self.test_planned_actions= np.zeros([constants.DEMOS_CEM_PATH_LENGTH,2], dtype=np.float32)
        self.test_planned_index = 0
        self.train_counter = 0
        self.game_end = False
        self.previous_state = np.array([0.0, 0.0], dtype=np.float32)
        self.previous_action = np.array([0.0, 0.0], dtype=np.float32)
        # Assuming state_data will always be of shape [2] and dtype=torch.float32
        self.path_data = torch.empty((0, 2), dtype=torch.float32)

    def process_transition(self, state, action, next_state, money_remaining):
    
        if np.linalg.norm(state - next_state) <= 4:
            self.stuck = True
        #print(f"caled distance:{np.linalg.norm(state - self.goal_state)}")
        #print(f"stuck:{np.linalg.norm(state - next_state)}")
        state_data = torch.tensor([state[0], state[1]], dtype=torch.float32)
        self.path_data = torch.vstack((self.path_data, state_data.unsqueeze(0))) 
        self.paths_to_draw.append(PathToDraw(path=self.path_data,colour=(255,0,0),width=2))

        state_action_data = torch.tensor([state[0], state[1], action[0], action[1]], dtype=torch.float32)
        next_state_data = torch.tensor([next_state[0], next_state[1]], dtype=torch.float32)

        self.network_input_data = torch.vstack((self.network_input_data, state_action_data))
        self.network_label_data = torch.vstack((self.network_label_data, next_state_data))

        if self.plan_index == len(self.planned_actions) - 1 or np.linalg.norm(next_state - self.goal_state) < 4.0:
    
            self.num_episodes += 1
            if float(money_remaining) > 3.0:
                self.train_dynamic_model()

            self.current_path = []

            self.plan_index=0
            self.planned_actions=np.zeros((constants.DEMOS_CEM_PATH_LENGTH, 2))
            self.train_counter += 1
            self.paths_to_draw=[]

                  
    def train_dynamic_model(self):
        
        # Loop over epochs
        num_training_data = len(self.network_input_data)
        minibatch_size = 5
        num_minibatches = int(num_training_data / minibatch_size)
        num_epochs = 100
        loss_function = torch.nn.MSELoss()
        for epoch in range(num_epochs):
            
            # Create a random permutation of the training indices
            permutation = torch.randperm(num_training_data)
            # Loop over minibatches
            training_epoch_losses = []
            for minibatch in range(num_minibatches):
                # Set all the gradients stored in the optimiser to zero.
                self.optimiser.zero_grad()
                # Get the indices for the training data based on the permutation
                training_indices = permutation[minibatch * minibatch_size: (minibatch + 1) * minibatch_size]
                minibatch_inputs = self.network_input_data[training_indices]
                minibatch_labels = self.network_label_data[training_indices]
                # Do a forward pass of the network using the inputs batch
                training_prediction = self.dynamics_model_network.forward(minibatch_inputs)
                # Compute the loss based on the label's batch
                training_loss = loss_function(training_prediction, minibatch_labels)
                # Compute the gradients based on this loss
                training_loss.backward()
                # Take one gradient step to update the network
                self.optimiser.step()
                # Get the loss as a scalar value
                training_loss_value = training_loss.item()
                training_epoch_losses.append(training_loss_value)
            # Calculate the epoch loss
            training_epoch_loss = np.average(training_epoch_losses)
            # Store this loss in the list
            self.training_losses.append(training_epoch_loss)
            # Update the list of epochs
            self.num_training_epochs += 1
            training_epochs_list = range(self.num_training_epochs)
            # Plot and save the loss vs iterations graph
        #     self.ax.plot(training_epochs_list, self.training_losses, color='blue')
        #     plt.yscale('log')
        #     plt.show()
        # #print("train finished")
        
        # plt.figure()
        # plt.plot(training_epochs_list, self.training_losses, color='blue')
        # plt.yscale('log')
        # plt.xlabel('Training Epochs')
        # plt.ylabel('Training Loss')
        # plt.title('Loss Curve for Model Training')
        # plt.show()
        
            
    def dynamics_model(self, state, action):
        # state_tensor = torch.tensor(state, dtype=torch.float32)
        # action_tensor = torch.tensor(action, dtype=torch.float32)
        # state_tensor = state_tensor.view(-1)  # Reshape to ensure it's flat
        # action_tensor = action_tensor.view(-1)  # Reshape to ensure it's flat
        # network_input = torch.cat((state_tensor, action_tensor), dim=0)
        # network_input = torch.unsqueeze(network_input, 0)
        # #network_input.reshape((1,4))
        # predicted_next_state = self.dynamics_model_network.forward(network_input)[0].detach().numpy()
        
        # predicted_next_state = np.array([[predicted_next_state[0]], [predicted_next_state[1]]])

        # return predicted_next_state
        with torch.no_grad():
            # Convert the numpy arrays to torch tensors
            # The [:, 0] is because states and actions are defined as 2-by-1 arrays elsewhere
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.float32)
            state_tensor = state_tensor.reshape(-1)
            action_tensor = action_tensor.reshape(-1)
            
            # Make a prediction with the neural network
            network_input = torch.cat((state_tensor, action_tensor))
            network_input = torch.unsqueeze(network_input, 0)
            predicted_next_state = self.dynamics_model_network.forward(network_input).squeeze(0) 
            # Convert from shape (2) to shape (2, 1)
            #print(predicted_next_state.numpy().shape)
            #predicted_next_state = np.array([[predicted_next_state[0]], [predicted_next_state[1]]])
            return predicted_next_state.numpy()
        
        
    def compute_reward(self, path):
        reward = -np.linalg.norm(path[-1] - self.goal_state)
        return reward

    def get_next_action_type(self, state, money_remaining):
        
        #print(np.linalg.norm(self.goal_state - state))
        # TODO: This informs robot-learning.py what type of operation to perform
        # It should return either 'demo', 'reset', or 'step'
        if False:
            return 'demo'
        if np.all(self.planned_actions == 0) or np.linalg.norm(state-self.goal_state) < 4.0:
            if float(money_remaining) > float(constants.COST_PER_RESET):
                #print("reset")
                self.planned_actions = np.zeros((constants.DEMOS_CEM_PATH_LENGTH, 2))
                self.planned_path = np.zeros((constants.DEMOS_CEM_PATH_LENGTH, 2))
                
                self.plan_index = 0
                self.planned_actions = self.plan_action(state)
                #print("hi")
                self.game_end = False
                #print(self.planned_actions)
                return 'reset'
            else:
                #print("h2")
                return "reset"
        else:
                #print("h3")
                return 'step'

    def get_next_action_training(self, state, money_remaining):
        
        # action= self.planned_actions[self.plan_index]
        # direction = (state - self.goal_state)
        # norm = direction / np.linalg.norm(state - self.goal_state)
        # diff = np.linalg.norm(norm - action/np.linalg.norm(action))
        # #print(diff)
        # if diff > 1.0: 
        #     print("1")
        #     action = state - self.goal_state
        #     action += np.random.normal(0, 1,action.shape)
        # else:
        #     print("2")
        #print(self.goal_state)
        if self.stuck:
            # state = state.reshape((2,1))
            # # goal = self.goal_state.reshape((2,1))
            # action = state -  self.goal_state
            # # action = action.reshape((2,1))
            #print("1")
            # action += np.random.normal(0, 1,action.shape)
            
            self.planned_actions = self.plan_action(state)
            action = self.planned_actions[0]
            self.plan_index = 0
            self.stuck = False
            
        else:
            #print("2")
            action = self.planned_actions[self.plan_index]
            
        if self.plan_index < len(self.planned_actions) - 1:
            self.plan_index += 1
        #print(self.plan_index)
        #print(action)
        return action

    def get_next_action_testing(self, state):
        
        if self.test_planned_index == len(self.test_planned_actions) - 1:
            self.test_planned_actions = np.zeros((constants.DEMOS_CEM_PATH_LENGTH, 2))
            self.test_planned_index = 0
        
        
        #print(f"index:{self.test_planned_index}")
        test_distance = 5
        
        if (np.all(self.test_planned_actions == 0)):
            #print(f"---------------best path is empty--------")
            self.test_planned_index = 0
            self.test_planned_actions =  self.plan_action(state)
            
            #print(f"planed action generated is empty: {np.any(np.isnan(self.planned_actions))}")
            #print(self.test_planned_actions)    
        action= self.test_planned_actions[self.test_planned_index]
        direction = (state - self.goal_state)
        norm = direction / np.linalg.norm(state - self.goal_state)
        diff = np.linalg.norm(norm - action/np.linalg.norm(action))
        #print(diff)
        if diff > 1.0:
            
            # action = state - self.goal_state
            # action += np.random.normal(0, 1,action.shape)
            
            self.test_planned_actions = self.plan_action(state)
            action = self.planned_actions[0]
            self.test_planned_index = 0
            # #print("2")
        
            
        if self.test_planned_index < len(self.test_planned_actions) - 1:
            self.test_planned_index += 1
        #print(action)
        return action
     
    def plan_action(self, state):
        self.planning_paths = np.zeros([constants.DEMOS_CEM_PATH_LENGTH,2], dtype=np.float32)
        planning_state = np.copy(state)
        # Plan the next 4 steps
        for iteration_num in range(constants.DEMOS_CEM_PATH_LENGTH):
            
            baseline = planning_state -  self.goal_state
            #actions = np.random.uniform(-(baseline), baseline, [constants.DEMOS_CEM_NUM_PATHS, 2])
            actions = np.random.normal(baseline, np.abs(baseline) / 10, size=(constants.DEMOS_CEM_NUM_PATHS, 2))
            # print("------------")
            # print(np.abs(baseline)/10)
            # print(baseline)
            # print(actions)
            min_distance = np.inf
            closest_action = np.array([0.0, 0.0], dtype=np.float32)
            for i in actions:
                distance = np.linalg.norm(i - baseline)
                if distance < min_distance:
                    min_distance = distance
                    closest_action = i   
            action = closest_action + np.random.normal(0,1,[2])
            #print(action)
            next_state = self.dynamics_model(planning_state, action)
            self.planning_paths[iteration_num] = next_state
            self.planned_actions[iteration_num] = action
            planning_state = next_state
        return self.planned_actions
        
                    
