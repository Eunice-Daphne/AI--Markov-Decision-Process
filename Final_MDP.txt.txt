class mdp_node:
    row_val = -1
    col_val = -1
    reward = 0
    walls = False
    is_terminal_state = False
    
    
    def __init__(self, i, j, walls, is_terminal_state, reward):
        self.row_val = j
        self.col_val = i
        self.reward = reward
        self.walls = walls
        self.is_terminal_state = is_terminal_state
        
        
    def calculate_transition_model(self, movement, mov_prob, mdp_obj, utility_list):
        transition_val = 0
        for mov in mov_prob:
            if mov_prob[mov] > 0:
                i, j = getNextLocation(self.row_val, self.col_val, movement, mov, mdp_obj)
                next_indx = (i * mdp_obj.column_size) + j
                transition_val += (mov_prob[mov] * utility_list[next_indx])
        return transition_val
    
class mdp:
    row_size = None
    column_size = None
    mdp_nodes = []
    movement_prob_mapping = {}#{'U' : 0.8, 'L' : 0.1, 'D' : 0, 'R' : 0.1}
    state_unreachable_val = []#[False, False, False, False, False, True, False, False, False, False, False, False]
    state_terminal_val = []#[False, False, False, False, False, False, False, True, False, False, False, True]
    reward = []
    gamma_discount_val = None
    
    def __init__(self, row_size, column_size, walls_location, location_of_terminal_states, reward, transition_probabilities, gamma):
        self.row_size = row_size
        self.column_size = column_size
        self.state_unreachable_val = [False for i in range(self.row_size * self.column_size)]
        self.state_terminal_val = [False for i in range(self.row_size * self.column_size)]
        self.reward = [reward for i in range(self.row_size * self.column_size)]
        for movement in transition_probabilities:
            self.movement_prob_mapping[movement] = transition_probabilities[movement]
        self.gamma_discount_val = gamma
        for entry in walls_location:
            indx = (self.column_size*(entry[1]-1)) + entry[0]-1
            self.reward[indx] = None
            self.state_unreachable_val[indx] = True
        for entry in location_of_terminal_states:
            indx = (self.column_size*(entry[1]-1)) + entry[0]-1
            self.state_terminal_val[indx] = True
            self.reward[indx] = entry[2]
        for j in range(self.row_size):
            for i in range(self.column_size):
                indx = (self.column_size*j) + i
                self.mdp_nodes.append(mdp_node(i, j, self.state_unreachable_val[indx], self.state_terminal_val[indx], self.reward[indx]))
                
#function that calculates the Bellman function                
def value_iteration(mdp_obj, epsilon):
    U = [0 for i in range(mdp_obj.row_size*mdp_obj.column_size)]
    U_prime = [0 for i in range(mdp_obj.row_size*mdp_obj.column_size)]
    
    iter_count = 0
    for state in mdp_obj.mdp_nodes:
            indx = (state.row_val * mdp_obj.column_size) + state.col_val
            if state.walls:
                U[indx] = state.reward
    print("Iteration","0", ":")
    printUtilVal(mdp_obj, U)
    while True:
        U = U_prime.copy()
        
        delta = 0
        iter_count += 1
        
        for state in mdp_obj.mdp_nodes:
            indx = (state.row_val * mdp_obj.column_size) + state.col_val
            if state.is_terminal_state or state.walls:
                U_prime[indx] = state.reward
                continue
            #U_prime[indx] = round(state.reward + mdp_obj.gamma_discount_val * getMaxValue(state, mdp_obj, U_prime), 3)
            
            U_prime[indx] = state.reward + mdp_obj.gamma_discount_val * getMaxValue(state, mdp_obj, U_prime)
            abs_diff = abs(U_prime[indx] - U[indx])
            if (abs_diff) > delta:
                delta = abs_diff
        print("Iteration", iter_count, ":")
        printUtilVal(mdp_obj, U_prime)
        margin = epsilon * (1 - mdp_obj.gamma_discount_val) / mdp_obj.gamma_discount_val
        if delta > margin:
            continue
        break
    return U

#to get the next location for the corresponding action
def getNextLocation(row_val, col_val, movement, mov, mdp_obj):
    actual_movement = movement #mov = 'U'
    old_row_val = row_val
    old_col_val = col_val
    if mov == 'L':
        if movement == 'U':
            actual_movement = 'L'
        elif movement == 'L':
            actual_movement = 'D'
        elif movement == 'R':
            actual_movement = 'U'
        elif movement == 'D':
            actual_movement = 'R'
    elif mov == 'R':
        if movement == 'U':
            actual_movement = 'R'
        elif movement == 'L':
            actual_movement = 'U'
        elif movement == 'R':
            actual_movement = 'D'
        elif movement == 'D':
            actual_movement = 'L'
    elif mov == 'D':
        if movement == 'U':
            actual_movement = 'D'
        elif movement == 'L':
            actual_movement = 'R'
        elif movement == 'R':
            actual_movement = 'L'
        else:
            actual_movement = 'U'
            
    if actual_movement == 'U':
        row_val = row_val+1
    elif actual_movement == 'R':
        col_val = col_val+1
    elif actual_movement == 'L':
        col_val = col_val-1
    else:
        row_val = row_val-1
    
    if row_val < 0 or col_val < 0 or row_val >= mdp_obj.row_size or col_val >= mdp_obj.column_size:
        return old_row_val, old_col_val
    new_indx = (row_val * mdp_obj.column_size) + col_val
    if mdp_obj.mdp_nodes[new_indx].walls:
        return old_row_val, old_col_val
    return row_val, col_val

#print the utility values for each iteration
def printUtilVal(mdp_obj, utility_list):
    output_str_final = []
    output_str = ""
    i = 0
    for state in mdp_obj.mdp_nodes:
            indx = (state.row_val * mdp_obj.column_size) + state.col_val
            output_str += str(utility_list[indx]) + "\t"
            i = i+1
            if(i % mdp_obj.column_size == 0):
                output_str_final.append(output_str)
                output_str = ""
    output_str_final = list(reversed(output_str_final))
    for entry in output_str_final:
        print(entry)
        
#to get the maximum of the transition probability values           
def getMaxValue(state, mdp_obj, U):
    transition_model_vals = []
    for movement in mdp_obj.movement_prob_mapping:
        transition_model_vals.append(state.calculate_transition_model(movement, mdp_obj.movement_prob_mapping, mdp_obj, U))
    return max(transition_model_vals)
    
def read_input_file(input_file):
    file = open(input_file, "r")
    column_size = -1
    row_size = -1
    walls_location = [] #list containing location of walls
    location_of_terminal_states = [] #list containing location of terminal states with their rewards
    reward = 0 #reward value for non-terminal states
    transition_probabilities = {'U': 0.0, 'L': 0.0, 'R': 0.0, 'D': 0.0}
    gamma = 1
    epsilon = 0
    for line in file:#parsing the input text file
        if "#" not in line:
            if "size" in line:
                sizeval = line.split(":")[1].strip()
                column_size = int(sizeval.split(" ")[0].strip())
                row_size = int(sizeval.split(" ")[1].strip())
            elif "wall" in line:
                location_val = line.split(":")[1].strip()
                locations = location_val.split(",")
                for location in locations:
                    location = location.strip()
                    loc = []
                    loc.append(int(location.split(" ")[0].strip()))
                    loc.append(int(location.split(" ")[1].strip()))
                    walls_location.append(loc)
            elif "terminal_states" in line:
                terminal_val = line.split(":")[1].strip()
                terminals = terminal_val.split(",")
                for terminal in terminals:
                    terminal = terminal.strip()
                    ter = []
                    ter.append(int(terminal.split(" ")[0].strip()))
                    ter.append(int(terminal.split(" ")[1].strip()))
                    ter.append(int(terminal.split(" ")[2].strip()))
                    location_of_terminal_states.append(ter)
            elif "reward" in line:
                reward = float(line.split(":")[1].strip())
            elif "transition_probabilities" in line:
                transition_val = line.split(":")[1].strip()
                transitions = transition_val.split(" ")
                transition_probabilities['U'] = float(transitions[0].strip())
                transition_probabilities['L'] = float(transitions[1].strip())
                transition_probabilities['R'] = float(transitions[2].strip())
                transition_probabilities['D'] = float(transitions[3].strip())
            elif "gamma" in line:
                gamma = float(line.split(":")[1].strip())
            elif "epsilon" in line:
                epsilon = float(line.split(":")[1].strip())
    return row_size, column_size, walls_location, location_of_terminal_states, reward, transition_probabilities, gamma, epsilon
          
def main():
    epsilon = 0.001
    try:
        row_size, column_size, walls_location, location_of_terminal_states, reward, transition_probabilities, gamma, epsilon = read_input_file(r"F:\Daffy\UIC\Fall 2018\Artificial Intelligence I\AI Assignment 12\input.txt")
        mdp_obj = mdp(row_size, column_size, walls_location, location_of_terminal_states, reward, transition_probabilities, gamma)
    except:
        print("File not found")
        return
    print("Value iteration")
    utility_vals = value_iteration(mdp_obj, epsilon)
    
    
if __name__ == '__main__':
    main()