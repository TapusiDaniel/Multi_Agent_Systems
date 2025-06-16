from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction


class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.round_history = []
        self.fair_share = 0
        self.target_share = 0
        self.previous_resource = 0
        self.round_counter = 0

    def specify_share(self, perception: CommonsPerception) -> float:
        self.round_counter += 1
        self.fair_share = 1.0 / perception.num_agents
        
        current_resource = perception.resource_quantity
        
        if not self.round_history:
            self.target_share = 0.8 / perception.num_agents
            self.previous_resource = current_resource
            return self.target_share
        
        regeneration_occurred = current_resource > self.previous_resource
        
        resource_ratio = perception.resource_quantity / self.round_history[0].get('initial_resource', 2000)
        
        if resource_ratio < 0.01:
            self.target_share = 0.1 / perception.num_agents
        elif resource_ratio < 0.1:
            self.target_share = 0.4 / perception.num_agents 
        elif resource_ratio < 0.3:
            self.target_share = 0.6 / perception.num_agents
        else:
            self.target_share = 0.8 / perception.num_agents
            
        if regeneration_occurred:
            self.target_share *= 1.1
            
        self.target_share = min(self.target_share, 0.9 / perception.num_agents)
        
        self.previous_resource = current_resource
        
        return self.target_share

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        if not perception.resource_shares:
            return AgentAction(self.id, resource_share=self.target_share, consumption_adjustment={}, no_action=True)
            
        current_total_share = sum(perception.resource_shares.values())
        current_share = perception.resource_shares[self.id]
        
        if current_total_share > 0.9:
            adjustments = {}
            target_total = 0.7
            reduction_factor = target_total / current_total_share
            
            for agent_id, share in perception.resource_shares.items():
                if agent_id != self.id:
                    adjustments[agent_id] = share * reduction_factor - share
                    
            new_share = current_share * reduction_factor
            
            return AgentAction(self.id, resource_share=new_share, consumption_adjustment=adjustments, no_action=False)
        
        if negotiation_round > 0 and perception.aggregate_adjustment:
            if self.id in perception.aggregate_adjustment and perception.aggregate_adjustment[self.id] < 0:
                proposed_adjustment = perception.aggregate_adjustment[self.id]
                min_acceptable = 0.3 / perception.num_agents
                if current_share + proposed_adjustment >= min_acceptable:
                    return AgentAction(self.id, resource_share=current_share + proposed_adjustment, 
                                      consumption_adjustment={}, no_action=False)
                else:
                    return AgentAction(self.id, resource_share=min_acceptable, 
                                      consumption_adjustment={}, no_action=False)
            
        return AgentAction(self.id, resource_share=current_share, consumption_adjustment={}, no_action=True)

    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        if perception.resource_shares and perception.agent_utilities:
            round_data = {
                'shares': perception.resource_shares.copy(),
                'utilities': perception.agent_utilities.copy(),
                'remaining': perception.resource_remaining,
                'initial_resource': perception.resource_quantity + (
                    perception.resource_quantity * sum(perception.resource_shares.values()) / 
                    (1 - sum(perception.resource_shares.values())) if sum(perception.resource_shares.values()) < 1 else 0
                )
            }
            self.round_history.append(round_data)