import logging
from typing import List, Dict, Any
from . import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage
logger = logging.getLogger(__name__)

class MyACMEAgent(HouseOwnerAgent):
    """
    Represents ACME, the company seeking to build headquarters.
    Implements ACME's strategy for auctions and negotiations.
    """
    # Strategy constants for ACME
    # For auction: proposed price = item_budget * factor. Max 3 rounds (0, 1, 2).
    AUCTION_PRICE_FACTORS = [0.7, 0.85, 1.0]  # Starts low, increases towards full budget.
    # For negotiation: ACME's offer = auction_settled_price * factor. Max 3 rounds (0, 1, 2).
    NEGOTIATION_OFFER_FACTORS = [0.80, 0.90, 1.01] # Starts below auction price, concedes up to it.

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self).__init__(role, budget_list)
        # Stores the offer ACME made for an item in the current auction round
        self.auction_item_current_offers: Dict[str, float] = {}
        # Stores the price at which an item was settled in the auction phase (ACME's offer that got bids)
        self.auction_settled_prices: Dict[str, float] = {}
        # Stores the last offer received from a partner in negotiation: (item_name, partner_agent_name) -> offer
        self.negotiation_partner_last_offers: Dict[tuple[str, str], float] = {}
        
        self.total_spent_on_contracts: float = 0.0
        logger.info(f"MyACMEAgent '{self.name}' initialized. Budget: {self.budget_dict}")

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        """
        ACME proposes a price for the auction item.
        Starts low and raises the price in subsequent rounds (max 3 times).
        """
        item_max_budget = self.budget_dict[auction_item]
        
        current_round_factor = self.AUCTION_PRICE_FACTORS[-1] # Default to last factor (full budget)
        if auction_round < len(self.AUCTION_PRICE_FACTORS):
            current_round_factor = self.AUCTION_PRICE_FACTORS[auction_round]
        else:
            logger.warning(f"ACME: Auction round {auction_round} for '{auction_item}' is out of defined factor range. Using last factor.")

        proposed_price = item_max_budget * current_round_factor
        self.auction_item_current_offers[auction_item] = proposed_price
        
        logger.debug(f"ACME '{self.name}': Proposing for auction item '{auction_item}' (Round {auction_round}): {proposed_price:.2f} (Budget: {item_max_budget:.2f})")
        return proposed_price

    def notify_auction_round_result(self, auction_item: str, auction_round: int, responding_agents: List[str]):
        """
        ACME is notified of the auction round result. If companies responded,
        the auction for this item ends, and ACME stores the settled price.
        """
        if responding_agents:
            settled_price = self.auction_item_current_offers.get(auction_item)
            if settled_price is not None:
                self.auction_settled_prices[auction_item] = settled_price
                logger.info(f"ACME '{self.name}': Auction for '{auction_item}' (Round {auction_round}) settled at {settled_price:.2f}. Responding agents: {responding_agents}")
            else:
                # This case should ideally not happen if propose_item_budget was called before.
                logger.error(f"ACME '{self.name}': CRITICAL - Auction for '{auction_item}' has respondents, but no current offer was stored by ACME.")
        else:
            current_offer_val = self.auction_item_current_offers.get(auction_item, 'N/A')
            offer_display = f"{current_offer_val:.2f}" if isinstance(current_offer_val, float) else current_offer_val
            logger.debug(f"ACME '{self.name}': Auction for '{auction_item}' (Round {auction_round}): No respondents to price {offer_display}.")

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str, negotiation_round: int) -> float:
        """
        ACME provides an offer during negotiation.
        It starts with a price lower than the auction-settled price and concedes (increases offer)
        upwards, potentially up to the auction-settled price, over max 3 rounds.
        """
        auction_price = self.auction_settled_prices.get(negotiation_item)
        if auction_price is None:
            # Fallback: this is an error state. Use item budget if auction price unknown.
            logger.error(f"ACME '{self.name}': CRITICAL - Negotiating for '{negotiation_item}' but no auction settled price found. Using budget as fallback.")
            auction_price = self.budget_dict[negotiation_item]

        current_round_factor = self.NEGOTIATION_OFFER_FACTORS[-1] # Default to last factor
        if negotiation_round < len(self.NEGOTIATION_OFFER_FACTORS):
            current_round_factor = self.NEGOTIATION_OFFER_FACTORS[negotiation_round]
        else:
            logger.warning(f"ACME '{self.name}': Negotiation round {negotiation_round} for '{negotiation_item}' is out of defined factor range. Using last factor.")
            
        my_offer = auction_price * current_round_factor
        
        logger.debug(f"ACME '{self.name}': Negotiating for '{negotiation_item}' with {partner_agent} (Round {negotiation_round}). Proposing {my_offer:.2f} (Auction price was {auction_price:.2f})")
        return my_offer

    def notify_partner_response(self, response_msg: NegotiationMessage) -> None:
        """
        ACME is notified of a partner's response (counter-offer) in negotiation.
        """
        item = response_msg.negotiation_item
        partner = response_msg.sender
        offer = response_msg.offer
        self.negotiation_partner_last_offers[(item, partner)] = offer
        logger.debug(f"ACME '{self.name}': Received negotiation response for '{item}' from {partner}. Partner offers: {offer:.2f}.")

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str, winning_offer: float) -> None:
        """
        ACME is notified of the outcome of a negotiation for an item.
        """
        self.total_spent_on_contracts += winning_offer
        original_budget = self.budget_dict[negotiation_item]
        savings = original_budget - winning_offer
        logger.info(f"ACME '{self.name}': Contract for '{negotiation_item}' awarded to {winning_agent} at {winning_offer:.2f}. (Budget: {original_budget:.2f}, Savings: {savings:.2f})")
        logger.info(f"ACME '{self.name}': Total spent so far: {self.total_spent_on_contracts:.2f}")


class MyCompanyAgent(CompanyAgent):
    """
    Represents a contractor company.
    Implements the company's strategy for auctions and negotiations.
    """
    # Strategy constants for CompanyAgent
    BASE_PROFIT_MARGIN = 0.20  # Base desired profit (e.g., 20% over cost)
    
    # Adjust profit margin based on number of competitors in negotiation phase
    PROFIT_ADJUST_SOLE_BIDDER = 0.05  # Add to margin if sole competitor (e.g., total 25%)
    PROFIT_ADJUST_MANY_COMPETITORS = -0.05 # Subtract from margin if 3+ competitors (e.g., total 15%)

    # Concession steps for profit margin reduction during negotiation rounds (0, 1, 2)
    PROFIT_CONCESSION_STEPS = [0.0, 0.05, 0.10] # Reduction from (adjusted) base profit margin
    
    DESPERATION_PROFIT_MARGIN = 0.01 # Aim for 1% profit if no contracts won and in last negotiation round
    MAX_NEGOTIATION_ROUNDS = 3 # Used to identify the last negotiation round (0, 1, 2)

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self).__init__(role, specialties)
        self.contracts_won_count: int = 0
        # Store auction-related data: item_name -> {"num_selected": int}
        self.auction_item_details: Dict[str, Dict[str, Any]] = {}
        # Store own last response in negotiation to ensure concession: item_name -> float (price)
        self.negotiation_my_last_responses: Dict[str, float] = {}
        logger.info(f"MyCompanyAgent '{self.name}' initialized. Specialties: {self.specialties}")

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        """
        Company decides whether to bid for an item at ACME's current offered price (item_budget).
        Bids if item_budget >= own cost, to maximize chances of entering negotiation.
        """
        # This check is usually done by the environment, but being defensive.
        if not self.has_specialty(auction_item):
            logger.warning(f"Company {self.name}: Asked to bid on '{auction_item}' but lacks specialty.")
            return False

        cost = self.specialties[auction_item]
        will_bid = item_budget >= cost

        if will_bid:
            logger.debug(f"Company {self.name}: WILL BID for '{auction_item}' (Cost: {cost:.2f}, ACME's Offer: {item_budget:.2f}, Round: {auction_round}).")
        else:
            logger.debug(f"Company {self.name}: WILL NOT BID for '{auction_item}' (Cost: {cost:.2f}, ACME's Offer: {item_budget:.2f}, Round: {auction_round}). Offer too low.")
        return will_bid

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int):
        """
        Company is notified it was selected in the auction for an item.
        Stores num_selected to inform negotiation strategy.
        """
        self.auction_item_details[auction_item] = {"num_selected": num_selected}
        logger.info(f"Company {self.name}: Qualified in auction for '{auction_item}' (Round {auction_round}). Total selected companies: {num_selected}.")

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        """
        Company responds to ACME's negotiation offer.
        Aims for a profit margin, concedes over rounds, considers competitors and desperation.
        """
        item = initiator_msg.negotiation_item
        acme_offer = initiator_msg.offer # ACME's current offer to the company
        negotiation_round = initiator_msg.round
        
        cost = self.specialties[item]
        # Default num_selected to 1 if details are somehow missing (should not happen)
        num_competitors = self.auction_item_details.get(item, {}).get("num_selected", 1)

        # 1. Determine base target profit for this item, adjusted by competition
        target_profit_margin = self.BASE_PROFIT_MARGIN
        if num_competitors == 1: # Sole qualified bidder
            target_profit_margin += self.PROFIT_ADJUST_SOLE_BIDDER
        elif num_competitors >= 3: # Multiple competitors
            target_profit_margin += self.PROFIT_ADJUST_MANY_COMPETITORS
        
        # 2. Apply concession based on negotiation round
        concession_reduction = self.PROFIT_CONCESSION_STEPS[-1] # Default to max concession
        if negotiation_round < len(self.PROFIT_CONCESSION_STEPS):
            concession_reduction = self.PROFIT_CONCESSION_STEPS[negotiation_round]
        else:
            logger.warning(f"Company {self.name}: Negotiation round {negotiation_round} for '{item}' exceeds planned concession steps. Using max concession reduction.")

        effective_profit_margin = target_profit_margin - concession_reduction
        
        # 3. Handle desperation for the first contract if in the last negotiation round
        is_last_round = (negotiation_round == self.MAX_NEGOTIATION_ROUNDS - 1)
        if self.contracts_won_count == 0 and is_last_round:
            effective_profit_margin = self.DESPERATION_PROFIT_MARGIN
            logger.info(f"Company {self.name}: Desperate for first contract on '{item}' (last round), using minimal profit margin: {effective_profit_margin*100:.1f}%.")

        # Calculate company's desired offer for this round based on strategic profit margin
        my_strategic_offer = cost * (1 + effective_profit_margin)
        # Ensure company's offer is never below its actual cost
        my_strategic_offer = max(my_strategic_offer, cost)

        # 4. Ensure monotonic concession: current offer must be <= previous offer from this company for this item
        my_final_offer = my_strategic_offer
        last_response_price = self.negotiation_my_last_responses.get(item)
        if last_response_price is not None:
            my_final_offer = min(my_strategic_offer, last_response_price) # Must concede or hold
        
        # Final check: ensure offer is not below cost after monotonic adjustment
        my_final_offer = max(my_final_offer, cost)

        # Store this offer as the last one made for this item to ensure future concessions
        self.negotiation_my_last_responses[item] = my_final_offer
        
        logger.debug(f"Company {self.name}: Responding for '{item}' (Cost: {cost:.2f}, ACME Offer: {acme_offer:.2f}, Round: {negotiation_round}). Num competitors: {num_competitors}. Effective margin: {effective_profit_margin*100:.1f}%. Proposing: {my_final_offer:.2f}")
        return my_final_offer

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        """
        Company is notified it won a contract.
        """
        self.contracts_won_count += 1
        profit = price - self.specialties[construction_item]
        logger.info(f"Company {self.name}: ASSIGNED contract for '{construction_item}' at {price:.2f}. (Cost: {self.specialties[construction_item]:.2f}, Profit: {profit:.2f}). Total contracts won: {self.contracts_won_count}")
        
        # Clear negotiation state for this item as it's concluded
        if construction_item in self.negotiation_my_last_responses:
            del self.negotiation_my_last_responses[construction_item]

    def notify_negotiation_lost(self, construction_item: str) -> None:
        """
        Company is notified its negotiation for an item failed (another company won).
        """
        logger.info(f"Company {self.name}: LOST negotiation for '{construction_item}'.")
        
        # Clear negotiation state for this item
        if construction_item in self.negotiation_my_last_responses:
            del self.negotiation_my_last_responses[construction_item]