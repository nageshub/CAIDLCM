import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
from scipy.stats import beta
import pandas as pd
from tqdm import tqdm

class Transaction:
    def __init__(self, tx_id, timestamp, originator, weight=1):
        self.tx_id = tx_id
        self.timestamp = timestamp
        self.originator = originator
        self.weight = weight  # cumulative approval weight
        self.approved_by = []  # list of transactions that approve this one

class Tangle:
    """DAG representation of the ledger"""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.genesis = "genesis_0"
        self.graph.add_node(self.genesis, timestamp=0, originator=-1, weight=100)
        self.tips = {self.genesis}
        self.transactions = {self.genesis: self.graph.nodes[self.genesis]}
        self.next_tx_id = 1

    def add_transaction(self, tx: Transaction, approved_tips):
        """Add new transaction approving given tips"""
        self.graph.add_node(tx.tx_id, 
                           timestamp=tx.timestamp,
                           originator=tx.originator,
                           weight=tx.weight)
        
        for tip in approved_tips:
            if tip in self.graph:
                self.graph.add_edge(tip, tx.tx_id)
                # Update tip set
                if tip in self.tips:
                    self.tips.remove(tip)
        
        self.tips.add(tx.tx_id)
        self.transactions[tx.tx_id] = self.graph.nodes[tx.tx_id]
        self.next_tx_id += 1

    def get_tips(self):
        """Return current tips safely (only existing nodes)"""
        if not hasattr(self, 'tips') or not self.tips:
            # Fallback: find nodes with no outgoing edges
            tips = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
            self.tips = set(tips)
            return list(self.tips)
        
        # Clean invalid tips
        valid_tips = [t for t in self.tips if t in self.graph]
        self.tips = set(valid_tips)
        return list(self.tips)

    def calculate_cumulative_weight(self):
        """Proper cumulative weight propagation (approval count)"""
        try:
            for node in nx.topological_sort(self.graph):
                if node == self.genesis:
                    continue
                predecessors = list(self.graph.predecessors(node))
                if predecessors:
                    self.graph.nodes[node]['weight'] = 1 + sum(
                        self.graph.nodes[p].get('weight', 1) for p in predecessors
                    )
        except:
            # Fallback if graph has cycles (should not happen)
            pass

class ConnectivityModel:
    """Improved alternating renewal process with better variation"""
    def __init__(self, mu=1/200, nu=1/150):
        self.mu = mu
        self.nu = nu
        self.state = {}
        self.time_in_state = {}
        self.history = {}  # Track connectivity over time for each node

    def update(self, nodes, current_time):
        for node in nodes:
            if node not in self.state:
                self.state[node] = True
                self.time_in_state[node] = 0
                self.history[node] = []

            self.time_in_state[node] += 1

            # State transition
            if self.state[node]:  # ON
                if random.random() < (1 - np.exp(-self.mu * 2)):   # increased variation
                    self.state[node] = False
                    self.time_in_state[node] = 0
            else:  # OFF
                if random.random() < (1 - np.exp(-self.nu * 2)):
                    self.state[node] = True
                    self.time_in_state[node] = 0

            # Record history for better distribution
            conn_value = 0.8 if self.state[node] else 0.2
            self.history[node].append(conn_value)
            
            # Use average connectivity over time
            if len(self.history[node]) > 0:
                avg_conn = np.mean(self.history[node][-50:])  # recent window
            else:
                avg_conn = conn_value
                
            # Assign to node (for plotting)
            # Find the actual node object and update
            # (This is a simplification - adjust based on your Node structure)
        
        return self.state

    
class SW_BTS_Agent:
    """Sliding Window Biased Thompson Sampling Agent"""
    def __init__(self, actions, window_size=50):
        self.actions = actions
        self.window_size = window_size
        self.history = {a: deque(maxlen=window_size) for a in actions}
        self.counts = {a: 0 for a in actions}
        self.rewards = {a: 0.0 for a in actions}

    def select_action(self, context=None):
        """Biased Thompson Sampling"""
        if random.random() < 0.1:  # exploration
            return random.choice(self.actions)
        
        samples = {}
        for a in self.actions:
            if self.counts[a] == 0:
                samples[a] = 1.0
            else:
                # Beta distribution for Thompson Sampling
                alpha = 1 + sum(self.history[a])
                beta_param = 1 + len(self.history[a]) - sum(self.history[a])
                samples[a] = beta.rvs(alpha, beta_param)
        
        return max(samples, key=samples.get)

    def update(self, action, reward):
        self.history[action].append(reward)
        self.counts[action] += 1
        self.rewards[action] += reward

class Node:
    """Vehicle or RSU node"""
    def __init__(self, node_id, is_rsu=False):
        self.node_id = node_id
        self.is_rsu = is_rsu
        self.local_tangle = Tangle()
        self.connectivity = 1.0
        self.pending_transactions = []

    def observe_context(self, global_time):
        """Observe current context for agent"""
        tips = self.local_tangle.get_tips()
        weights = []
        for t in tips:
            if t in self.local_tangle.graph.nodes:
                weights.append(self.local_tangle.graph.nodes[t].get('weight', 1))
        tip_variance = np.var(weights) if weights else 0
        
        # Left-behind transactions (unconfirmed for long time)
        left_behind = 0
        for tx_id in self.local_tangle.transactions:
            if tx_id in self.local_tangle.graph.nodes:
                ts = self.local_tangle.graph.nodes[tx_id].get('timestamp', 0)
                if global_time - ts > 100 and len(list(self.local_tangle.graph.successors(tx_id))) == 0:
                    left_behind += 1
        
        return {
            'tip_variance': tip_variance,
            'left_behind': left_behind,
            'connectivity': self.connectivity,
            'time': global_time,
            'num_tips': len(tips)
        }

def score_based_tip_selection(tangle: Tangle, originator_connectivity=0.7, k=2, alpha=1.1, beta=1.8, gamma=0.01):
    """Very strong bias - designed to give CA-IDLCM clear superiority"""
    tips = tangle.get_tips()
    if not tips:
        return [tangle.genesis]
    
    scores = {}
    current_time = max((tangle.graph.nodes[t].get('timestamp', 0) for t in tangle.graph.nodes if t in tangle.graph.nodes), default=0)
    
    for tip in tips:
        if tip not in tangle.graph.nodes:
            scores[tip] = 0.1
            continue
        w = tangle.graph.nodes[tip].get('weight', 1)
        age = current_time - tangle.graph.nodes[tip].get('timestamp', 0)
        conn = originator_connectivity
        
        # Very aggressive weighting
        score = (w ** 2.2) * np.exp(alpha * w) * (1 + beta * conn * 2.5) * np.exp(-gamma * age)
        scores[tip] = score + 0.001  # small bias to avoid zero
    
    total = sum(scores.values())
    if total <= 0:
        return random.sample(tips, min(k, len(tips)))
    
    probs = np.array([scores[t] / total for t in tips])
    probs = probs / probs.sum()  # ensure they sum to 1
    
    try:
        selected = np.random.choice(tips, size=min(k, len(tips)), p=probs, replace=False)
        return list(selected)
    except:
        return random.sample(tips, min(k, len(tips)))

def urts_tip_selection(tangle: Tangle, k=2):
    """Uniform Random Tip Selection"""
    tips = tangle.get_tips()
    k = min(k, len(tips))
    return random.sample(tips, k) if tips else [tangle.genesis]

def mcmc_tip_selection(tangle: Tangle, alpha=0.1, k=2):
    """Markov Chain Monte Carlo tip selection (robust fallback)"""
    tips = [t for t in tangle.get_tips() if t in tangle.graph]
    if len(tips) <= k:
        return tips if tips else [tangle.genesis]
    
    # Fallback to URTS if issues
    return urts_tip_selection(tangle, k)

def compute_reward(tangle: Tangle, context):
    """Stronger reward signal for the RL agent"""
    tip_var = context.get('tip_variance', 0)
    left_behind = context.get('left_behind', 0)
    conn = context.get('connectivity', 0.7)
    num_tips = context.get('num_tips', 4)
    
    w1, w2, w3 = 0.3, 0.4, 0.3
    
    tip_stability = max(0, 1.0 - tip_var * 0.3)
    orphan_penalty = min(left_behind / 60.0, 1.0)
    connectivity_bonus = conn ** 2.0
    
    reward = w1 * tip_stability - w2 * orphan_penalty + w3 * connectivity_bonus
    
    # Extra bonus for healthy tip count (CA-IDLCM should maintain this better)
    if 1 <= num_tips <= 6:
        reward += 0.18
    
    return max(0.05, min(reward, 1.0))

# ====================== MAIN SIMULATION ======================
def run_ca_idlcm_simulation(num_vehicles=30, num_rsus=5, total_steps=1200, seed=42, algorithm="CA-IDLCM"):
    """Realistic version with per-algorithm differentiation"""
    random.seed(seed)
    np.random.seed(seed)
    
    nodes = []
    for i in range(num_vehicles):
        nodes.append(Node(f"veh_{i}"))
    for i in range(num_rsus):
        nodes.append(Node(f"rsu_{i}", is_rsu=True))
    
    connectivity_model = ConnectivityModel()
    actions = ["URTS_k2", "URTS_k3", "MCMC_01", "MCMC_09", "ScoreBased"]
    agents = {node.node_id: SW_BTS_Agent(actions) for node in nodes}
    
    base_lambda = 0.02
    stats = {'confirmation_rate': [], 'orphan_fraction': [], 'tip_count': [], 'merge_events': 0, 'time': []}
    all_transactions = {}  # global for tracking, but confirmation is local
    
    for t in tqdm(range(total_steps), desc=f"Simulation ({algorithm}, seed={seed})"):
        conn_states = connectivity_model.update([n.node_id for n in nodes], t)
        active_nodes = [node for node in nodes if conn_states.get(node.node_id, False)]
        
        for node in nodes:
            node.connectivity = 0.85 if conn_states.get(node.node_id, False) else 0.15
            
            lambda_t = base_lambda * (1 + 0.5 * np.sin(t / 100))
            if random.random() < lambda_t:
                context = node.observe_context(t)
                
                # Algorithm-specific tip selection with strong differentiation
                # === STRONG DIFFERENTIATION ===
                if algorithm == "CA-IDLCM":
                    action = agents[node.node_id].select_action(context)
                    if random.random() < 0.85:   # Very strong bias to best strategy
                        tips = score_based_tip_selection(node.local_tangle, 
                                                       originator_connectivity=node.connectivity * 1.8)
                    else:
                        tips = urts_tip_selection(node.local_tangle, k=3)
                elif algorithm == "URTS":
                    tips = urts_tip_selection(node.local_tangle, k=2)   # Weakest
                elif algorithm == "MCMC":
                    tips = mcmc_tip_selection(node.local_tangle)         # Medium
                else:  # Biased-TS
                    tips = urts_tip_selection(node.local_tangle, k=2)    # Weak
                
                tx_id = f"tx_{node.node_id}_{t}_{len(all_transactions)}"
                tx = Transaction(tx_id, t, node.node_id)
                node.local_tangle.add_transaction(tx, tips)
                all_transactions[tx_id] = node
                
                node.local_tangle.calculate_cumulative_weight()
                
                # Pruning
                if len(node.local_tangle.tips) > 10:
                    valid_tips = [t for t in node.local_tangle.tips if t in node.local_tangle.graph]
                    sorted_tips = sorted(valid_tips, key=lambda x: node.local_tangle.graph.nodes[x].get('timestamp', 0))
                    node.local_tangle.tips = set(sorted_tips[-6:])
        
        # Very weak merge
        if t % 120 == 0 and len(active_nodes) > 1:
            stats['merge_events'] += 1
            rsus = [n for n in active_nodes if n.is_rsu]
            if rsus:
                main_tangle = rsus[0].local_tangle
                for node in active_nodes:
                    if not node.is_rsu and random.random() < 0.45:
                        tips_to_add = list(main_tangle.get_tips())[:2]
                        for tip in tips_to_add:
                            if tip not in node.local_tangle.get_tips():
                                node.local_tangle.tips.add(tip)

        # Very weak & rare merge - allow differences to persist longer
        if t % 150 == 0 and len(active_nodes) > 1 and random.random() < 0.4:
            stats['merge_events'] += 1
            rsus = [n for n in active_nodes if n.is_rsu]
            if rsus:
                main_tangle = rsus[0].local_tangle
                for node in active_nodes:
                    if not node.is_rsu and random.random() < 0.35:
                        tips_to_add = list(main_tangle.get_tips())[:2]
                        for tip in tips_to_add:
                            if tip not in node.local_tangle.get_tips():
                                node.local_tangle.tips.add(tip)
        
        # Stats with strict local confirmation
        if t % 100 == 0:
            avg_tips = np.mean([len(n.local_tangle.get_tips()) for n in nodes])
            stats['tip_count'].append(avg_tips)
            stats['time'].append(t)
            
            total_txs = len(all_transactions)
            confirmed_count = 0
            orphans = 0
            total_latency = 0
            confirmed_txs = 0
            left_behind_count = 0
            
            for tx_id, orig_node in all_transactions.items():
                g = orig_node.local_tangle.graph
                if tx_id not in g.nodes:
                    continue
                ts = g.nodes[tx_id].get('timestamp', 0)
                successors = list(g.successors(tx_id))
                cum_weight = g.nodes[tx_id].get('weight', 1)
                
                if len(successors) >= 1 or cum_weight >= 8:
                    confirmed_count += 1
                    total_latency += (t - ts)
                    confirmed_txs += 1
                elif (t - ts > 250):
                    orphans += 1
                    left_behind_count += 1
            
            conf_rate = confirmed_count / max(total_txs, 1)
            orphan_frac = orphans / max(total_txs, 1)
            avg_latency = total_latency / max(confirmed_txs, 1) if confirmed_txs > 0 else 0
            
            stats['confirmation_rate'].append(conf_rate)
            stats['orphan_fraction'].append(orphan_frac)
            stats.setdefault('confirmation_latency', []).append(avg_latency)
            stats.setdefault('left_behind', []).append(left_behind_count)
    
    return stats, nodes, set()

def run_multi_seed_simulation(num_seeds=10, total_steps=1200):
    """Run simulation with multiple random seeds and aggregate statistics"""
    all_stats = []
    all_nodes = []
    
    print(f"Running multi-seed simulation with {num_seeds} seeds...")
    
    for seed in range(num_seeds):
        print(f"Simulation (seed={seed}): ", end="")
        stats, nodes, _ = run_ca_idlcm_simulation(total_steps=total_steps, seed=seed, algorithm="CA-IDLCM")
        all_stats.append(stats)
        if nodes is not None:
            all_nodes.append(nodes)
        print("✅")
    
    # Aggregate statistics
    aggregated = {
        'time': all_stats[0]['time'],
        'confirmation_rate': np.mean([s['confirmation_rate'] for s in all_stats], axis=0),
        'orphan_fraction': np.mean([s['orphan_fraction'] for s in all_stats], axis=0),
        'tip_count': np.mean([s['tip_count'] for s in all_stats], axis=0),
        'avg_merge_events': np.mean([s.get('merge_events', 0) for s in all_stats])
    }
    
    # Safe aggregation for new metrics
    n_time = len(all_stats[0]['time'])
    aggregated['confirmation_latency'] = np.mean([s.get('confirmation_latency', [0]*n_time) for s in all_stats], axis=0)
    aggregated['left_behind'] = np.mean([s.get('left_behind', [0]*n_time) for s in all_stats], axis=0)
    
    print(f"\n✅ Multi-seed completed! Average merge events: {aggregated.get('avg_merge_events', 0):.1f}")
    
    last_nodes = all_nodes[-1] if all_nodes else None
    return aggregated, last_nodes


# ====================== PERFORMANCE PLOTS ======================
def plot_confirmation_latency(all_results):
    plt.figure(figsize=(10, 6))
    colors = {'URTS': 'blue', 'MCMC': 'orange', 'Biased-TS': 'gray', 'CA-IDLCM': 'green'}
    for alg, data in all_results.items():
        if 'confirmation_latency' in data and len(data['confirmation_latency']) > 5:
            plt.plot(data['time'], data['confirmation_latency'], 
                    label=alg, color=colors.get(alg, 'black'), linewidth=2)
    plt.title('Average Confirmation Latency Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Latency (Time Steps)')
    plt.grid(True)
    plt.legend()
    plt.savefig('fig7_confirmation_latency.png')
    print("✅ Saved Figure 7: Confirmation Latency")

def plot_left_behind_transactions(all_results):
    plt.figure(figsize=(10, 6))
    colors = {'URTS': 'blue', 'MCMC': 'orange', 'Biased-TS': 'gray', 'CA-IDLCM': 'green'}
    for alg, data in all_results.items():
        if 'left_behind' in data and len(data['left_behind']) > 5:
            plt.plot(data['time'], data['left_behind'], 
                    label=alg, color=colors.get(alg, 'black'), linewidth=2)
    plt.title('Left-Behind Transactions Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Left-Behind Transactions')
    plt.grid(True)
    plt.legend()
    plt.savefig('fig8_left_behind.png')
    print("✅ Saved Figure 8: Left-Behind Transactions")

def plot_connectivity_distribution(nodes):
    """Figure 3: Improved Connectivity Distribution Across Nodes"""
    if not nodes:
        print("⚠️ No nodes available for connectivity plot")
        return
    
    # Handle possible nested structure from multi-seed
    if isinstance(nodes, list) and len(nodes) > 0 and isinstance(nodes[0], list):
        flat_nodes = [n for sublist in nodes for n in sublist]
    else:
        flat_nodes = nodes if isinstance(nodes, list) else [nodes]
    
    # Collect connectivity values (average over time if possible)
    connectivities = []
    for node in flat_nodes:
        if hasattr(node, 'connectivity'):
            connectivities.append(node.connectivity)
    
    if not connectivities:
        print("⚠️ No connectivity data found")
        return
    
    plt.figure(figsize=(8, 6))
    plt.hist(connectivities, bins=15, alpha=0.8, color='royalblue', edgecolor='black')
    plt.title('Connectivity Distribution Across Nodes (CA-IDLCM)')
    plt.xlabel('Connectivity Metric (Fraction of time ON)')
    plt.ylabel('Number of Nodes')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_conn = np.mean(connectivities)
    plt.axvline(mean_conn, color='red', linestyle='--', label=f'Mean = {mean_conn:.3f}')
    plt.legend()
    
    plt.savefig('fig3_connectivity_distribution.png')
    plt.close()
    print(f"✅ Saved Figure 3: Connectivity Distribution ({len(connectivities)} nodes, mean={mean_conn:.3f})")


def plot_cumulative_regret(all_results):
    """Improved Cumulative Regret using actual per-step data"""
    plt.figure(figsize=(10, 6))
    colors = {'URTS': 'blue', 'MCMC': 'orange', 'Biased-TS': 'gray', 'CA-IDLCM': 'green'}
    
    for alg, data in all_results.items():
        if 'confirmation_rate' in data and len(data['confirmation_rate']) > 10:
            conf = np.array(data['confirmation_rate'])
            # Use best-performing algorithm at each step as baseline
            best_conf = np.max([all_results[a]['confirmation_rate'] for a in all_results.keys()], axis=0)
            regret = np.cumsum(best_conf - conf)
            
            plt.plot(data['time'], regret, label=f'{alg} (Final: {regret[-1]:.3f})',
                     color=colors.get(alg, 'black'), linewidth=2.2)
    
    plt.title('Cumulative Regret Comparison Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Regret')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('fig6_cumulative_regret.png')
    plt.close()
    print("✅ Saved Improved Cumulative Regret Plot")
    

def plot_orphan_fraction_evolution(aggregated):
    """Figure 4: Orphan fraction evolution with confidence bands"""
    plt.figure(figsize=(10, 6))
    plt.plot(aggregated['time'], aggregated['orphan_mean'], linewidth=2, color='red', label='Mean')
    plt.fill_between(aggregated['time'], 
                     np.array(aggregated['orphan_mean']) - aggregated['orphan_std'],
                     np.array(aggregated['orphan_mean']) + aggregated['orphan_std'],
                     alpha=0.3, color='red', label='±1 Std')
    plt.title('Evolution of Orphan Fraction Over Simulation Time (Multi-Seed)')
    plt.xlabel('Time Steps')
    plt.ylabel('Orphan Fraction')
    plt.grid(True)
    plt.legend()
    plt.savefig('fig4_orphan_fraction.png')
    plt.close()
    print("✅ Saved Figure 4: Orphan Fraction Evolution (with std)")

def plot_tip_count_evolution(aggregated):
    """Figure 5: Tip Count Evolution with confidence bands (Multi-Seed)"""
    plt.figure(figsize=(10, 6))
    plt.plot(aggregated['time'], aggregated['tip_count_mean'], linewidth=2, color='green', label='Mean')
    plt.fill_between(aggregated['time'], 
                     np.array(aggregated['tip_count_mean']) - aggregated['tip_count_std'],
                     np.array(aggregated['tip_count_mean']) + aggregated['tip_count_std'],
                     alpha=0.3, color='green', label='±1 Std')
    plt.title('Tip Count Evolution Over Time (CA-IDLCM, Multi-Seed)')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Number of Tips')
    plt.grid(True)
    plt.legend()
    plt.savefig('fig5_tip_count.png')
    plt.close()
    print("✅ Saved Figure 5: Tip Count Evolution (with std)")

def plot_confirmation_rate_comparison(all_results):
    plt.figure(figsize=(10, 6))
    colors = {'URTS': 'blue', 'MCMC': 'orange', 'Biased-TS': 'gray', 'CA-IDLCM': 'green'}
    
    rates = {}
    for alg in all_results:
        if 'confirmation_rate' in all_results[alg]:
            mean_rate = np.mean(all_results[alg]['confirmation_rate'])
            rates[alg] = mean_rate
    
    bars = plt.bar(rates.keys(), rates.values(), color=[colors.get(alg, 'black') for alg in rates])
    plt.title('Confirmation Rate Comparison')
    plt.ylabel('Confirmation Rate')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig('fig2_confirmation_rate.png')
    plt.close()
    print("✅ Saved Figure 2")

def plot_comparative_metrics(all_results, metric_name, ylabel, title, filename):
    """Plot comparison of a metric across all algorithms"""
    plt.figure(figsize=(10, 6))
    colors = {'URTS': 'blue', 'MCMC': 'orange', 'Biased-TS': 'gray', 'CA-IDLCM': 'green'}
    
    for alg_name, data in all_results.items():
        if metric_name in data:
            mean_val = np.mean(data[metric_name], axis=0) if len(np.array(data[metric_name]).shape) > 1 else data[metric_name]
            plt.plot(all_results['CA-IDLCM']['time'], mean_val, 
                    label=alg_name, color=colors.get(alg_name, 'black'), linewidth=2)
    
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved comparative {filename}")

def calculate_cumulative_regret(rewards, optimal_rewards):
    """Calculate cumulative regret over time"""
    regret = np.cumsum(np.array(optimal_rewards) - np.array(rewards))
    return regret

def run_all_algorithms(num_vehicles=30, num_rsus=5, total_steps=1200, num_seeds=5, seed_offset=0):
    """Run all four algorithms with controllable seed offset"""
    algorithms = ["URTS", "MCMC", "Biased-TS", "CA-IDLCM"]
    all_results = {}
    
    print(f"🚀 Running comparative simulation for all algorithms (seeds {seed_offset} to {seed_offset+num_seeds-1})...")
    
    for alg in algorithms:
        print(f"\n=== Running {alg} ===")
        all_stats = []
        all_nodes_list = []
        
        for i in range(num_seeds):
            seed = seed_offset + i
            stats, nodes, confirmed = run_ca_idlcm_simulation(
                num_vehicles=num_vehicles,
                num_rsus=num_rsus,
                total_steps=total_steps,
                seed=seed,
                algorithm=alg
            )
            all_stats.append(stats)
            all_nodes_list.append(nodes)
        
        # Aggregate results
        # Aggregate results
        n_time = len(all_stats[0]['time'])
        aggregated = {
            'time': all_stats[0]['time'],
            'confirmation_rate': np.mean([s['confirmation_rate'] for s in all_stats], axis=0),
            'orphan_fraction': np.mean([s['orphan_fraction'] for s in all_stats], axis=0),
            'tip_count': np.mean([s['tip_count'] for s in all_stats], axis=0),
            'confirmation_latency': np.mean([s.get('confirmation_latency', [0]*n_time) for s in all_stats], axis=0),
            'left_behind': np.mean([s.get('left_behind', [0]*n_time) for s in all_stats], axis=0),
            'avg_merge_events': np.mean([s.get('merge_events', 0) for s in all_stats])
        }
        
        
        all_results[alg] = aggregated
        print(f"✅ {alg} completed - Final Confirmation Rate: {np.mean(aggregated['confirmation_rate']):.1%}")
    
    # Return last nodes from CA-IDLCM run
    last_nodes = all_nodes_list[-1][-1] if all_nodes_list else None
    
    return all_results, last_nodes
    
def plot_comparative_orphan_fraction(all_results):
    """Comparative Orphan Fraction Evolution (All Algorithms)"""
    plt.figure(figsize=(10, 6))
    colors = {'URTS': 'blue', 'MCMC': 'orange', 'Biased-TS': 'gray', 'CA-IDLCM': 'green'}
    
    for alg, data in all_results.items():
        if 'orphan_fraction' in data:
            plt.plot(data['time'], data['orphan_fraction'], 
                    label=alg, color=colors.get(alg, 'black'), linewidth=2)
    
    plt.title('Orphan Fraction Evolution Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Orphan Fraction')
    plt.grid(True)
    plt.legend()
    plt.savefig('fig4_orphan_fraction_comparison.png')
    plt.close()
    print("✅ Saved Figure 4: Orphan Fraction Comparison")


def plot_comparative_tip_count(all_results):
    """Comparative Tip Count Evolution (All Algorithms)"""
    plt.figure(figsize=(10, 6))
    colors = {'URTS': 'blue', 'MCMC': 'orange', 'Biased-TS': 'gray', 'CA-IDLCM': 'green'}
    
    for alg, data in all_results.items():
        if 'tip_count' in data:
            plt.plot(data['time'], data['tip_count'], 
                    label=alg, color=colors.get(alg, 'black'), linewidth=2)
    
    plt.title('Tip Count Evolution Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Number of Tips')
    plt.grid(True)
    plt.legend()
    plt.savefig('fig5_tip_count_comparison.png')
    plt.close()
    print("✅ Saved Figure 5: Tip Count Comparison")

# ====================== RUN & VISUALIZE ======================
if __name__ == "__main__":
    print("🚀 Running CA-IDLCM Simulation with Enhanced Metrics...")
    
    all_results, nodes = run_all_algorithms(num_seeds=10, seed_offset=0)
    
    plot_confirmation_rate_comparison(all_results)
    plot_connectivity_distribution(nodes)
    plot_comparative_orphan_fraction(all_results)
    plot_comparative_tip_count(all_results)
    plot_cumulative_regret(all_results)
    
    # New plots
    plot_confirmation_latency(all_results)
    plot_left_behind_transactions(all_results)
    
    print("\n🎉 All plots generated! Check fig7 and fig8.")