import numpy as np
from abc import ABC, abstractmethod

#-------------------------------------------------------
class MAB(ABC):
    '''
       This is the parent class of Multi-armed bandit games. It defines the basic interface (APIs) that each game class should provide. 
    '''
    # ----------------------------------------------
    @abstractmethod
    def get_valid_actions(self):
        '''
            Get number of valid actions in the game. 
            Outputs:
                A: an integer scaler, the number of valid actions in the game. For example, in the 3-armed bandit, the number is 3.
                   In a 5-armed bandit, the number is 5 '''
        pass

    # ----------------------------------------------
    @abstractmethod
    def play_a_game(self, a):
        '''
           Play one game step: After the player choosing an action (the id of the arm being pulled), 
           return a reward to the player. 
            Input:
                a: the action chosen by the player. a is an integer scalar between 0 and n-1. 
                    n is the number of valid actions in the game.
            Output:
                r: the reward returned in the game.
        '''
        pass


    # ----------------------------------------------
    def run_games(self,player, N=1000):
        '''
            let the player play the game for N rounds. For each round, the player first chooses an action, then the game returns a reward to the player.
            Input:
                player: a player or agent that plays the multi-armed bandit game. 
                N: the number of rounds that the player plays the game.
            Outputs:
                e: the average reward per game =  total sum of rewards collected / N, a float scalar. For example, if the player plays 5 rounds, and the total rewards in these 5 rounds is 4. Then the average reward per game is 4/5 = 0.8 
        '''
        Total_reward = 0. # initialize the total rewards
        c = self.get_valid_actions() # get the number of valid actions in the game
        player.initialize_game_statistics(c) # let player to initialize the statistics about the game
        # run N games
        for _ in range(N):
            # run one game
            a = player.choose_action() # player choose an action
            r = self.play_a_game(a) # the game returns a reward
            Total_reward+=r # add to the total rewards
            player.update_memory(a,r) # let player to update the statistics with the chosen action and received rewards.
        e = Total_reward / N  # compute the average reward per game
        return e


#-------------------------------------------------------
class SlotMachine(MAB):
    '''SlotMachine is a game engine for Slot Machine. Slot machine can be considered as an one-armed bandit, where this is only one possible action. Each time the player plays the game, a random reward will be generated.'''
    # ----------------------------------------------
    def __init__(self, p=[0.3,0.2,0.5],r=[1,2,3]):
        ''' Initialize the game setting. 
            Inputs:
                p: the probabilities of each possible rewards.
                r: the possible reward values, a numpy vector. 
            For example, if p =[0.3,0.2,0.5] and r=[1,2,3], the machine has 0.3 (30%) of the chance to generate 1 as the reward,
            0.2 (20%) of the chance to generate 2 as the reward; 0.5 (50%) of the chance to generate 3 as the reward.
        '''
        self._p = p 
        self._r = r

    # ----------------------------------------------
    def get_valid_actions(self):
        '''
            Get the number of valid actions in the game. 
            Outputs:
                A: the number of valid actions in the game. In a slot machine, we only have one valid action, so the number is 1.
    
        '''
        return 1

    # ----------------------------------------------
    def play_a_game(self, a):
        '''
           Play one game step: After the player choosing an action (the id of the arm being pulled), 
           return a reward to the player. 
            Input:
                a: the action chosen by the player. In this case, we only have one possible action a=0 
            Output:
                r: the reward returned in the game.
        '''
        return np.random.choice(self._r, 1, p=self._p)



#-------------------------------------------------------
class BinaryMAB(MAB):
    '''This is a game engine for Multi-Armed Bandit machine where the reward can only be 0 or 1. '''
    # ----------------------------------------------
    def __init__(self, p=[0.7,0.2,0.5]):
        ''' Initialize the game setting. 
            Inputs:
                p: the winning probabilities of each possible action.
            For example, suppose p =[0.3,0.2,0.5], 
            if the player chooses the first action (a=0), the machine has 0.7 (70%) of the chance to generate 1 as the reward, 30% of the chance to generate 0 as the reward;
            if the player chooses the second action (a=1), the machine has 0.2 (20%) of the chance to generate 1 as the reward, 80% of the chance to generate 0 as the reward;
            if the player chooses the third action (a=2), the machine has 0.5 (50%) of the chance to generate 1 as the reward, 50% of the chance to generate 0 as the reward.
        '''
        self._p = p 

    # ----------------------------------------------
    def get_valid_actions(self):
        '''
            Get the number of valid actions in the game. 
            Outputs:
                A: the number of valid actions in the game. 
    
        '''
        return len(self_p) 

    # ----------------------------------------------
    def play_a_game(self, a):
        '''
           Play one game step: After the player choosing an action (the id of the arm being pulled), 
           return a reward to the player. 
            Input:
                a: the action chosen by the player. a is an integer scalar between 0 and n-1. 
                    n is the number of valid actions in the game.
            Output:
                r: the reward returned in the game.
        '''
        p = self._p[a]
        return np.random.choice([0.,1.], 1, p=[1.-p,p])




#-------------------------------------------------------
class CategoricalMAB(MAB):
    '''This is a game engine for Multi-Armed Bandit machine where the rewards can have some categorical values.
       This game can be considered as a collection of slot machines (SlotMachine class) '''
    # ----------------------------------------------
    def __init__(self, s):
        ''' Initialize the game setting. 
            Inputs:
                s: a list of slot machines, each slot machine can have a different probability distribution for reward values.
                 For example, we may have 3 slot machines (3-armed bandit), when taking each action (choosing one slot machine), we play that slot machine to collect the reward.
        '''
        self._s = s 

    # ----------------------------------------------
    def get_valid_actions(self):
        '''
            Get the number of valid actions in the game. 
            Outputs:
                A: the number of valid actions in the game. 
        '''
        return len(self._s) 

    # ----------------------------------------------
    def play_a_game(self, a):
        '''
           Play one game step: After the player choosing an action (the id of the arm being pulled), 
           return a reward to the player. 
            Input:
                a: the action chosen by the player. a is an integer scalar between 0 and n-1. 
                    n is the number of valid actions in the game.
            Output:
                r: the reward returned in the game.
        '''
        return self._s[a].play_a_game(0) 



#-------------------------------------------------------
class FrozenLake:
    '''A python implementation of the frozen lake game. 
        The list of actions for the player:
        ------------------------------
        0 : "LEFT"
        1 : "DOWN"
        2 : "RIGHT"
        3 : "UP"
        ------------------------------
        The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. 
        Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. 
        The agent is rewarded for finding a walkable path to a goal tile.
        The surface is described using a grid like the following:
        SFFF       (S: starting point, safe)
        FHFH       (F: frozen surface, safe)
        FFFH       (H: hole, fall to your doom)
        HFFG       (G: goal, where the frisbee is located)
        The game ends when you reach the goal or fall in a hole. You receive a reward of 1 if you reach the goal, and zero otherwise.
        For more details, please read https://gym.openai.com/envs/FrozenLake-v0/
'''
    # ----------------------------------------------
    def __init__(self, p=0.1, vector_state=False, image_state=False):
        ''' Initialize the game. 
            Inputs:
                p: probability of slippery move
            Outputs:
                self.n_s: the number of states of the machine, an integer scalar.
                self.row: the current row on the map, an integer scalar, initialized as 0.
                self.col: the current column on the map, an integer scalar, initialized as 0.
                self.map: the map of the game 
            Note: agent/player cannot access the above variables in the game. They are supposed to be hidden from the player.
        '''
        self.p = p 
        self.n_s = 16
        self.row = 0
        self.col = 0
        self.s= 0
        self.map=["SFFF",
                  "FHFH",
                  "FFFH",
                  "HFFG"]
        self.done = False # whether the game has ended yet
        self.vector_state = vector_state # whether or not to use a vector to represent a game state  (True: vector)
        self.image_state = image_state # whether returning the game state as an image (True) or a scalar (False)
        if image_state:
            self.map_image = np.zeros((1,9,9),dtype=np.float32)

    # ----------------------------------------------
    def game_state(self): # get game state
        if self.vector_state: # vector representation
            s=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
            s[self.s] = 1.
            return s
        elif self.image_state: # image representation
            x=self.map_image.copy()
            r,c = self.row,self.col # the current position of the agent
            x[0,2*r:2*r+2,2*c:2*c+2] = [[1,1],
                                        [1,1]]
            s = np.array(x,dtype=np.float32)
            return s
        else: # scalar representation
            return self.s

    # ----------------------------------------------
    def step(self, a):
        '''
           Given an action , return the reward and next state. 
           Input:
                a: the index of the lever being pulled by the agent. a is an integer scalar between 0 and n-1. 
                    n is the number of arms in the bandit.
           Output:
                s: the new state of the machine, an integer scalar. 
                r: the reward of the previous action, a float scalar. The "win" return 1., if "lose", return 0. as the reward.
                done: whether the game has ended, a boolean scalar. If the game has ended, return True. Otherwise, return False. 
        '''
        assert a in [0,1,2,3] # check if the action chosen by the player is valid
        assert not self.done # check if the game has already ended. if the game has ended, player can no longer move

        # if slippery, stay at the same location 
        slippery = np.random.choice([False,True], 1, p=[1.-self.p,self.p])

        if slippery:
            return self.game_state(), 0., False # slippery move (stay at the same location)

        if a==0:
            self.col=max(self.col-1,0)
        if a==1:
            self.row=min(self.row+1,3)
        if a==2:
            self.col=min(self.col+1,3)
        if a==3:
            self.row=max(self.row-1,0)
        self.s=self.row*4+self.col
        c = self.map[self.row][self.col]
        if c == 'H':
            self.done = True
            r = 0.
        elif c== 'G':
            self.done = True
            r = 1.
        else:
            r =0.
        return self.game_state(), r, self.done

    # ----------------------------------------------
    def run_games(self,player, N=1000):
        '''
            let the player play the game for N episodes. 
            Input:
                player: a player or agent that plays the frozen lake game. 
                N: the number of episodes that the player plays the game.
            Outputs:
                e: the average reward per game episode =  total sum of rewards collected / N, a float scalar. 
        '''
        Total_reward = 0. # initialize the total rewards
        # run N game episodes
        for _ in range(N):
            done=False
            s = self.game_state() # initial state
            # run 1 episode 
            while not done:
                # one game step
                a = player.choose_action(s) # player choose an action
                s_new,r,done = self.step(a) # play one game step 
                Total_reward+=r # add to the total rewards
                player.update_memory(s,a,s_new,r,done) # let player to update the statistics with the chosen action and received rewards.
                s = s_new
            # reset the game
            self.s=0
            self.row = 0
            self.col = 0
            self.done=False
        e = Total_reward / N  # compute the average reward per game
        return e




