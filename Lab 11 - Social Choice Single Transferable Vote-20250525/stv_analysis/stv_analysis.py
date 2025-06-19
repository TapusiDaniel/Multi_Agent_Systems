from preflibtools import io
from preflibtools.generate_profiles import gen_mallows, gen_cand_map, gen_impartial_culture_strict
from typing import List, Dict, Tuple
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

PHIS = [0.7, 0.8, 0.9, 1.0]
NUM_VOTERS = [100, 500, 1000]
NUM_CANDIDATES = [3, 6, 10, 15]

def generate_random_mixture(nvoters: int = 100, ncandidates: int = 6, num_refs: int = 3, phi: float = 1.0) \
    -> Tuple[Dict[int, str], List[Dict[int, int]], List[int]]:
    """
    Function that will generate a `voting profile` where there are num_refs mixtures of a
    Mallows model, each with the same phi hyperparameter
    """
    candidate_map = gen_cand_map(ncandidates)

    mix = []
    phis = []
    refs = []

    for i in range(num_refs):
        refm, refc = gen_impartial_culture_strict(1, candidate_map)
        refs.append(io.rankmap_to_order(refm[0]))
        phis.append(phi)
        mix.append(random.randint(1,100))

    smix = sum(mix)
    mix = [float(m)/float(smix) for m in mix]

    rmaps, rmapscounts = gen_mallows(nvoters, candidate_map, mix, phis, refs)

    return candidate_map, rmaps, rmapscounts

def stv(nvoters: int,
        candidate_map: Dict[int, str],
        rankings: List[Dict[int, int]],
        ranking_counts: List[int],
        top_k: int,
        required_elected: int = 1) -> List[int]:
    """
    Single Transferable Vote implementation (Instant Runoff Voting for single winner)
    
    :param nvoters: number of voters
    :param candidate_map: the mapping of candidate IDs to candidate names
    :param rankings: the expressed full rankings of voters, specified as a list of mapping from candidate_id -> rank
    :param ranking_counts: count of how many voters have each ranking
    :param top_k: the number of preferences taken into account [min: 2, max: (num_candidates), aka full STV]
    :param required_elected: number of candidates to elect (1 for single winner)
    :return: The list of elected candidate id-s
    """
    
    # Convert rankings to preference lists considering top_k
    votes = []
    for i, ranking in enumerate(rankings):
        count = ranking_counts[i]
        # Convert ranking dict to ordered list of candidates by preference
        # ranking maps candidate_id -> rank, so we sort by rank to get preference order
        sorted_candidates = sorted(ranking.items(), key=lambda x: x[1])
        
        # Take only top_k preferences
        if top_k < len(candidate_map):
            top_candidates = sorted_candidates[:top_k]
        else:
            top_candidates = sorted_candidates
        
        # Extract just the candidate IDs in preference order
        preference_order = [cand_id for cand_id, rank in top_candidates]
        
        # Add this vote 'count' times to our vote list
        for _ in range(count):
            votes.append(preference_order.copy())
    
    # For single winner IRV, we need majority (more than 50%)
    total_active_votes = len(votes)
    majority_threshold = total_active_votes // 2 + 1
    eliminated = set()
    
    while True:
        # Count first preferences for non-eliminated candidates
        first_pref_counts = defaultdict(int)
        exhausted_count = 0
        
        for vote in votes:
            # Find first non-eliminated candidate in this vote
            found_candidate = None
            for candidate in vote:
                if candidate not in eliminated:
                    found_candidate = candidate
                    break
            
            if found_candidate is not None:
                first_pref_counts[found_candidate] += 1
            else:
                # This vote is exhausted (all preferred candidates eliminated)
                exhausted_count += 1
        
        # Update majority threshold based on active votes
        active_votes = total_active_votes - exhausted_count
        if active_votes == 0:
            return []  # No winner possible
        
        majority_threshold = active_votes // 2 + 1
        
        # Check if any candidate has majority
        for candidate, count in first_pref_counts.items():
            if count >= majority_threshold:
                return [candidate]
        
        # No majority, eliminate candidate with fewest votes
        if len(first_pref_counts) <= 1:
            # Only one candidate left
            if first_pref_counts:
                return [max(first_pref_counts.keys(), key=lambda x: first_pref_counts[x])]
            else:
                return []  # No winner possible
        
        # Find candidate with minimum votes
        loser = min(first_pref_counts.keys(), key=lambda x: first_pref_counts[x])
        eliminated.add(loser)

def run_stv_analysis():
    """Run the complete STV analysis"""
    results = defaultdict(lambda: defaultdict(list))
    
    total_experiments = len(NUM_VOTERS) * len(NUM_CANDIDATES) * len(PHIS)
    current_exp = 0
    
    for nvoters in NUM_VOTERS:
        for ncandidates in NUM_CANDIDATES:
            for phi in PHIS:
                print(f"Running experiment {current_exp + 1}/{total_experiments}: "
                      f"voters={nvoters}, candidates={ncandidates}, phi={phi}")
                
                # Generate 1000 voting profiles for this experiment
                exact_matches = defaultdict(int)  # k -> number of exact matches
                total_profiles = 1000
                
                for profile_num in range(total_profiles):
                    if profile_num % 100 == 0:
                        print(f"  Profile {profile_num + 1}/{total_profiles}")
                    
                    # Generate voting profile
                    cmap, rankings, ranking_counts = generate_random_mixture(
                        nvoters=nvoters, 
                        ncandidates=ncandidates,
                        phi=phi
                    )
                    
                    # Compute full STV winner (using all preferences)
                    stv_full_winner = stv(nvoters, cmap, rankings, ranking_counts, ncandidates, 1)
                    
                    # Compute STV-k winners for different k values
                    for k in range(2, ncandidates):  # k from 2 to ncandidates-1
                        try:
                            stv_k_winner = stv(nvoters, cmap, rankings, ranking_counts, k, 1)
                            
                            # Check if winners match exactly
                            if stv_full_winner == stv_k_winner:
                                exact_matches[k] += 1
                        except Exception as e:
                            print(f"Error with k={k}: {e}")
                            continue
                
                # Calculate percentage of exact matches for each k
                for k in range(2, ncandidates):
                    if k in exact_matches:
                        percentage = (exact_matches[k] / total_profiles) * 100
                        results[(nvoters, ncandidates, phi)][k] = percentage
                
                current_exp += 1
    
    return results

def create_charts(results):
    """Create analysis charts"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('STV vs STV-k Overlap Analysis', fontsize=16)
    
    # Chart 1: Effect of k value for different numbers of candidates
    ax = axes[0, 0]
    for ncandidates in NUM_CANDIDATES:
        k_values = []
        avg_percentages = []
        
        for k in range(2, ncandidates):
            percentages = []
            for (nvoters, nc, phi), k_results in results.items():
                if nc == ncandidates and k in k_results:
                    percentages.append(k_results[k])
            
            if percentages:
                k_values.append(k)
                avg_percentages.append(np.mean(percentages))
        
        if k_values:
            ax.plot(k_values, avg_percentages, marker='o', linewidth=2, 
                   markersize=6, label=f'{ncandidates} candidates')
    
    ax.set_xlabel('k (top-k preferences)')
    ax.set_ylabel('Average % Exact Match')
    ax.set_title('Effect of k value on STV-k exact match rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Chart 2: Effect of number of voters
    ax = axes[0, 1]
    voter_percentages = defaultdict(list)
    
    for (nvoters, ncandidates, phi), k_results in results.items():
        for k, percentage in k_results.items():
            voter_percentages[nvoters].append(percentage)
    
    for nvoters in NUM_VOTERS:
        if nvoters in voter_percentages:
            percentages = voter_percentages[nvoters]
            ax.boxplot(percentages, positions=[nvoters], widths=50)
    
    ax.set_xlabel('Number of Voters')
    ax.set_ylabel('% Exact Match')
    ax.set_title('Effect of number of voters')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Chart 3: Effect of phi parameter
    ax = axes[1, 0]
    phi_percentages = defaultdict(list)
    
    for (nvoters, ncandidates, phi), k_results in results.items():
        for k, percentage in k_results.items():
            phi_percentages[phi].append(percentage)
    
    phi_means = []
    phi_stds = []
    for phi in PHIS:
        if phi in phi_percentages:
            percentages = phi_percentages[phi]
            phi_means.append(np.mean(percentages))
            phi_stds.append(np.std(percentages))
        else:
            phi_means.append(0)
            phi_stds.append(0)
    
    ax.bar(PHIS, phi_means, yerr=phi_stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Phi (φ)')
    ax.set_ylabel('Average % Exact Match')
    ax.set_title('Effect of φ parameter')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Chart 4: Effect of number of candidates
    ax = axes[1, 1]
    candidate_percentages = defaultdict(list)
    
    for (nvoters, ncandidates, phi), k_results in results.items():
        for k, percentage in k_results.items():
            candidate_percentages[ncandidates].append(percentage)
    
    cand_means = []
    cand_stds = []
    for ncandidates in NUM_CANDIDATES:
        if ncandidates in candidate_percentages:
            percentages = candidate_percentages[ncandidates]
            cand_means.append(np.mean(percentages))
            cand_stds.append(np.std(percentages))
        else:
            cand_means.append(0)
            cand_stds.append(0)
    
    ax.bar(NUM_CANDIDATES, cand_means, yerr=cand_stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Number of Candidates')
    ax.set_ylabel('Average % Exact Match')
    ax.set_title('Effect of number of candidates')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

def print_summary_statistics(results):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    all_percentages = []
    for (nvoters, ncandidates, phi), k_results in results.items():
        for k, percentage in k_results.items():
            all_percentages.append(percentage)
    
    if all_percentages:
        print(f"Overall Statistics:")
        print(f"  Mean exact match rate: {np.mean(all_percentages):.2f}%")
        print(f"  Median exact match rate: {np.median(all_percentages):.2f}%")
        print(f"  Standard deviation: {np.std(all_percentages):.2f}%")
        print(f"  Min exact match rate: {np.min(all_percentages):.2f}%")
        print(f"  Max exact match rate: {np.max(all_percentages):.2f}%")
    
    print(f"\nTotal experiments: {len(results)}")
    print(f"Total k-value tests: {sum(len(k_results) for k_results in results.values())}")

if __name__ == "__main__":
    print("Starting STV-k Analysis...")
    print("This will analyze how STV outcomes change when voters provide incomplete rankings")
    print(f"Parameters: Voters={NUM_VOTERS}, Candidates={NUM_CANDIDATES}, Phi={PHIS}")
    print(f"Generating 1000 profiles for each parameter combination...\n")
    
    # Run the analysis
    results = run_stv_analysis()
    
    # Print summary statistics
    print_summary_statistics(results)
    
    # Create visualizations
    print("\nCreating charts...")
    create_charts(results)
    
    print("\nAnalysis complete!")
    print("The charts show the percentage of experiments where STV and STV-k produce identical winners.")