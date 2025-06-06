from environment import *
from typing import List, Tuple
import time

class MyAgent(BlocksWorldAgent):

    def __init__(self, name: str, target_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)

        self.target_state = target_state

        """
        The agent's belief about the world state. Initially, the agent has no belief about the world state.
        """
        self.belief: BlocksWorld = None

        """
        The agent's current desire. It is expressed as a list of blocks for which the agent wants to make a plan to bring to their corresponding
        configuration in the target state. 
        The list can contain a single block or a sequence of blocks that represent: (i) a stack of blocks, (ii) a row of blocks (e.g. going level by level).
        """
        self.current_desire : List[Block] = None

        """
        The current intention is the agent plan (sequence of actions) that the agent is executing to achieve the current desire.
        """
        self.current_intention: List[BlocksWorldAction] = []


    def response(self, perception: BlocksWorldPerception) -> BlocksWorldAction:
        ## if the perceived state contains the target state, the agent has achieved its goal
        if perception.current_world.contains_world(self.target_state):
            return AgentCompleted()
        
        ## revise the agents beliefs based on the perceived state
        self.revise_beliefs(perception.current_world, perception.previous_action_succeeded)

        ## Single minded agent intention execution: if the agent still has actions left in the current intention, and the intention
        ## is still applicable to the perceived state, the agent continues executing the intention
        if len(self.current_intention) > 0 and self._can_apply_action(self.current_intention[0], perception.current_world, perception.holding_block):
            return self.current_intention.pop(0)
        else:
            ## the agent has to set a new current desire and plan to achieve it
            self.current_desire, self.current_intention = self.plan()

        ## If there is an action in the current intention, pop it and return it
        if len(self.current_intention) > 0:
            return self.current_intention.pop(0)
        else:
            ## If there is no action in the current intention, return a NoAction
            return NoAction()        


    def _can_apply_action(self, act: BlocksWorldAction, world: BlocksWorld, holding_block: str) -> bool:
        """
        Check if the action can be applied to the current world state.
        """
        ## create a clone of the world
        sim_world = world.clone()

        ## apply the action to the clone, surrpressing any exceptions
        try:
            ## locking can be performed at any time, so check if the action is a lock actio
            if act.get_type() == "lock":
                ## try to lock the block
                sim_world.lock(act.get_argument())
            else:
                if holding_block is None:
                    if act.get_type() == "putdown" or act.get_type() == "stack":
                        ## If we are not holding anything, we cannot putdown or stack a block
                        return False
                    
                    if act.get_type() == "pickup":
                        ## try to pickup the block
                        sim_world.pickup(act.get_argument())
                    elif act.get_type() == "unstack":
                        ## try to unstack the block
                        sim_world.unstack(act.get_first_arg(), act.get_second_arg())
                else:
                    ## we are holding a block, so we can only putdown or stack
                    if act.get_type() == "pickup" or act.get_type() == "unstack":
                        ## If we are holding a block, we cannot pickup or unstack
                        return False

                    if act.get_type() == "putdown":
                        ## If we want to putdown the block we have to check if it's the same block we are holding
                        if act.get_argument() != holding_block:
                            return False

                    if act.get_type() == "stack":
                        ## If we want to stack the block we have to check if it's the same block we are holding
                        if act.get_first_arg() != holding_block:
                            return False
                        ## try to stack the block
                        sim_world.stack(act.get_first_arg(), act.get_second_arg())
        except Exception as e:
            return False
        
        return True


    def revise_beliefs(self, perceived_world_state: BlocksWorld, previous_action_succeeded: bool):
        """
        TODO: revise internal agent structured depending on whether what the agent *expects* to be true 
        corresponds to what the agent perceives from the environment.
        :param perceived_world_state: the world state perceived by the agent
        :param previous_action_succeeded: whether the previous action succeeded or not
        """
        #raise NotImplementedError("not implemented yet; todo by student")
        pass


    def plan(self) -> Tuple[List[Block], List[BlocksWorldAction]]:
        # TODO: return the current desire from the set of possible / still required ones, on which the agent wants to focus next,
        # and the partial plan, as a sequence of `BlocksWorldAction' instances, that the agent wants to execute to achieve the current desire.
        return [], [NoAction()]


    def status_string(self):
        # TODO: return information about the agent's current state and current plan.
        return str(self) + " : PLAN MISSING"



class Tester(object):
    STEP_DELAY = 0.5
    TEST_SUITE = "tests/0e-large/"

    EXT = ".txt"
    SI  = "si"
    SF  = "sf"

    DYNAMICS_PROB = .5

    AGENT_NAME = "*A"

    def __init__(self):
        self._environment = None
        self._agents = []

        self._initialize_environment(Tester.TEST_SUITE)
        self._initialize_agents(Tester.TEST_SUITE)



    def _initialize_environment(self, test_suite: str) -> None:
        filename = test_suite + Tester.SI + Tester.EXT

        with open(filename) as input_stream:
            self._environment = DynamicEnvironment(BlocksWorld(input_stream=input_stream))


    def _initialize_agents(self, test_suite: str) -> None:
        filename = test_suite + Tester.SF + Tester.EXT

        agent_states = {}

        with open(filename) as input_stream:
            desires = BlocksWorld(input_stream=input_stream)
            agent = MyAgent(Tester.AGENT_NAME, desires)

            agent_states[agent] = desires
            self._agents.append(agent)

            self._environment.add_agent(agent, desires, None)

            print("Agent %s desires:" % str(agent))
            print(str(desires))


    def make_steps(self):
        print("\n\n================================================= INITIAL STATE:")
        print(str(self._environment))
        print("\n\n=================================================")

        completed = False
        nr_steps = 0

        while not completed:
            completed = self._environment.step()

            time.sleep(Tester.STEP_DELAY)
            print(str(self._environment))

            for ag in self._agents:
                print(ag.status_string())

            nr_steps += 1

            print("\n\n================================================= STEP %i completed." % nr_steps)

        print("\n\n================================================= ALL STEPS COMPLETED")





if __name__ == "__main__":
    tester = Tester()
    tester.make_steps()