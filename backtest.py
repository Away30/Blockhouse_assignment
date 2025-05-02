#!/usr/bin/env python3
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any

# Constants
ORDER_SIZE = 5000  # Target number of shares to buy
STEP_SIZE = 100    # Allocation search step size

class Venue:
    """Class representing a trading venue"""
    def __init__(self, venue_id: str, ask: float, ask_size: int, fee: float = 0.0, rebate: float = 0.0):
        self.venue_id = venue_id
        self.ask = ask
        self.ask_size = ask_size
        self.fee = fee
        self.rebate = rebate

def allocate(order_size: int, venues: List[Venue], lambda_over: float, lambda_under: float, theta_queue: float) -> Tuple[List[int], float]:
    """Allocate order to venues using the Cont-Kukanov algorithm"""
    step = STEP_SIZE
    splits = [[]]  # Start with an empty allocation list
    
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues[v].ask_size)
            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits
    
    best_cost = float('inf')
    best_split = []
    
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
    
    return best_split, best_cost

def compute_cost(split: List[int], venues: List[Venue], order_size: int, lambda_over: float, lambda_under: float, theta_queue: float) -> float:
    """Calculate the cost of an allocation plan"""
    executed = 0
    cash_spent = 0.0
    
    for i in range(len(venues)):
        exe = min(split[i], venues[i].ask_size)
        executed += exe
        cash_spent += exe * (venues[i].ask + venues[i].fee)
        maker_rebate = max(split[i] - exe, 0) * venues[i].rebate
        cash_spent -= maker_rebate
    
    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = theta_queue * (underfill + overfill)
    cost_pen = lambda_under * underfill + lambda_over * overfill
    
    return cash_spent + risk_pen + cost_pen

def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess market data"""
    print(f"Loading data: {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert timestamps to datetime format
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    
    # Keep only necessary columns
    cols = ['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']
    df = df[cols].dropna()
    
    # Ensure correct data types
    df['ask_px_00'] = df['ask_px_00'].astype(float)
    df['ask_sz_00'] = df['ask_sz_00'].astype(int)
    
    # For each unique ts_event, keep only the first message per publisher_id
    df = df.sort_values(['ts_event', 'publisher_id'])
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'], keep='first')
    
    return df

def prepare_snapshots(df: pd.DataFrame) -> List[Dict[str, List[Venue]]]:
    """Create market snapshots from data"""
    # Group data by timestamp
    snapshots = []
    timestamps = df['ts_event'].unique()
    
    for ts in timestamps:
        snapshot_df = df[df['ts_event'] == ts]
        venues = []
        
        for _, row in snapshot_df.iterrows():
            venue = Venue(
                venue_id=str(row['publisher_id']), 
                ask=row['ask_px_00'], 
                ask_size=row['ask_sz_00'],
                fee=0.0,  # Assume no fees/rebates
                rebate=0.0
            )
            venues.append(venue)
            
        snapshots.append({'timestamp': ts, 'venues': venues})
    
    return snapshots

def backtest_strategy(snapshots: List[Dict[str, List[Venue]]], lambda_over: float, lambda_under: float, theta_queue: float) -> Dict[str, Any]:
    """Backtest the Cont-Kukanov strategy with given parameters"""
    remaining_size = ORDER_SIZE
    total_cash_spent = 0.0
    total_shares_executed = 0
    execution_history = []
    
    for snapshot in snapshots:
        if remaining_size <= 0:
            break
            
        venues = snapshot['venues']
        if not venues:
            continue
            
        # Use allocator to decide how many shares to buy at each venue
        allocation, _ = allocate(remaining_size, venues, lambda_over, lambda_under, theta_queue)
        
        # Execute allocated orders
        for i, venue in enumerate(venues):
            if i < len(allocation):  # Ensure allocation and venues lengths match
                allocated_shares = allocation[i]
                executed_shares = min(allocated_shares, venue.ask_size)
                
                if executed_shares > 0:
                    execution_cost = executed_shares * venue.ask
                    total_cash_spent += execution_cost
                    total_shares_executed += executed_shares
                    remaining_size -= executed_shares
                    
                    execution_history.append({
                        'timestamp': snapshot['timestamp'],
                        'venue_id': venue.venue_id,
                        'executed_shares': executed_shares,
                        'price': venue.ask,
                        'cost': execution_cost
                    })
    
    # Calculate average fill price
    avg_fill_price = total_cash_spent / total_shares_executed if total_shares_executed > 0 else 0
    
    return {
        'parameters': {
            'lambda_over': lambda_over,
            'lambda_under': lambda_under,
            'theta_queue': theta_queue
        },
        'total_shares_executed': total_shares_executed,
        'total_cash_spent': total_cash_spent,
        'avg_fill_price': avg_fill_price,
        'execution_history': execution_history
    }

def backtest_best_ask_strategy(snapshots: List[Dict[str, List[Venue]]]) -> Dict[str, Any]:
    """Backtest baseline strategy: Best Ask"""
    remaining_size = ORDER_SIZE
    total_cash_spent = 0.0
    total_shares_executed = 0
    execution_history = []
    
    for snapshot in snapshots:
        if remaining_size <= 0:
            break
            
        venues = snapshot['venues']
        if not venues:
            continue
        
        # Find the lowest ask price
        best_venue = min(venues, key=lambda v: v.ask)
        
        # Execute trade
        executed_shares = min(remaining_size, best_venue.ask_size)
        if executed_shares > 0:
            execution_cost = executed_shares * best_venue.ask
            total_cash_spent += execution_cost
            total_shares_executed += executed_shares
            remaining_size -= executed_shares
            
            execution_history.append({
                'timestamp': snapshot['timestamp'],
                'venue_id': best_venue.venue_id,
                'executed_shares': executed_shares,
                'price': best_venue.ask,
                'cost': execution_cost
            })
    
    # Calculate average fill price
    avg_fill_price = total_cash_spent / total_shares_executed if total_shares_executed > 0 else 0
    
    return {
        'strategy': 'best_ask',
        'total_shares_executed': total_shares_executed,
        'total_cash_spent': total_cash_spent,
        'avg_fill_price': avg_fill_price,
        'execution_history': execution_history
    }

def backtest_twap_strategy(snapshots: List[Dict[str, List[Venue]]]) -> Dict[str, Any]:
    """Backtest baseline strategy: 60-second bucket TWAP"""
    remaining_size = ORDER_SIZE
    total_cash_spent = 0.0
    total_shares_executed = 0
    execution_history = []
    
    # Calculate total seconds in the time window
    start_time = snapshots[0]['timestamp']
    end_time = snapshots[-1]['timestamp']
    total_seconds = (end_time - start_time).total_seconds()
    
    # Calculate shares to execute per 60-second bucket
    bucket_duration = 60  # 60 seconds per bucket
    buckets = int(total_seconds / bucket_duration) + 1
    shares_per_bucket = ORDER_SIZE / buckets
    
    current_bucket = 0
    current_bucket_shares = 0
    
    for snapshot in snapshots:
        if remaining_size <= 0:
            break
            
        venues = snapshot['venues']
        if not venues:
            continue
        
        # Check if we need to enter the next time bucket
        time_since_start = (snapshot['timestamp'] - start_time).total_seconds()
        bucket_index = int(time_since_start / bucket_duration)
        
        if bucket_index > current_bucket:
            current_bucket = bucket_index
            current_bucket_shares = 0
        
        # Calculate shares to execute in this snapshot
        shares_to_execute = min(remaining_size, int(shares_per_bucket - current_bucket_shares))
        
        if shares_to_execute <= 0:
            continue
            
        # Find the venue with the best price
        best_venue = min(venues, key=lambda v: v.ask)
        
        # Execute trade
        executed_shares = min(shares_to_execute, best_venue.ask_size)
        if executed_shares > 0:
            execution_cost = executed_shares * best_venue.ask
            total_cash_spent += execution_cost
            total_shares_executed += executed_shares
            remaining_size -= executed_shares
            current_bucket_shares += executed_shares
            
            execution_history.append({
                'timestamp': snapshot['timestamp'],
                'venue_id': best_venue.venue_id,
                'executed_shares': executed_shares,
                'price': best_venue.ask,
                'cost': execution_cost
            })
    
    # Calculate average fill price
    avg_fill_price = total_cash_spent / total_shares_executed if total_shares_executed > 0 else 0
    
    return {
        'strategy': 'twap',
        'total_shares_executed': total_shares_executed,
        'total_cash_spent': total_cash_spent,
        'avg_fill_price': avg_fill_price,
        'execution_history': execution_history
    }

def backtest_vwap_strategy(snapshots: List[Dict[str, List[Venue]]]) -> Dict[str, Any]:
    """Backtest baseline strategy: VWAP (weighted by displayed ask size)"""
    remaining_size = ORDER_SIZE
    total_cash_spent = 0.0
    total_shares_executed = 0
    execution_history = []
    
    for snapshot in snapshots:
        if remaining_size <= 0:
            break
            
        venues = snapshot['venues']
        if not venues:
            continue
        
        # Calculate total displayed size
        total_size = sum(venue.ask_size for venue in venues)
        if total_size == 0:
            continue
        
        # Allocate orders based on displayed size
        for venue in venues:
            weight = venue.ask_size / total_size
            allocated_shares = int(min(remaining_size * weight, venue.ask_size))
            
            if allocated_shares > 0:
                execution_cost = allocated_shares * venue.ask
                total_cash_spent += execution_cost
                total_shares_executed += allocated_shares
                remaining_size -= allocated_shares
                
                execution_history.append({
                    'timestamp': snapshot['timestamp'],
                    'venue_id': venue.venue_id,
                    'executed_shares': allocated_shares,
                    'price': venue.ask,
                    'cost': execution_cost
                })
    
    # Calculate average fill price
    avg_fill_price = total_cash_spent / total_shares_executed if total_shares_executed > 0 else 0
    
    return {
        'strategy': 'vwap',
        'total_shares_executed': total_shares_executed,
        'total_cash_spent': total_cash_spent,
        'avg_fill_price': avg_fill_price,
        'execution_history': execution_history
    }

def parameter_search(snapshots: List[Dict[str, List[Venue]]]):
    """Search for optimal parameter combination"""
    # Define parameter search grid
    lambda_over_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    lambda_under_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    theta_queue_values = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    
    best_result = None
    best_params = None
    best_avg_price = float('inf')
    
    total_combinations = len(lambda_over_values) * len(lambda_under_values) * len(theta_queue_values)
    print(f"Starting parameter search, {total_combinations} combinations in total...")
    
    results = []
    start_time = time.time()
    
    # Grid search over parameter combinations
    for lambda_over in lambda_over_values:
        for lambda_under in lambda_under_values:
            for theta_queue in theta_queue_values:
                result = backtest_strategy(
                    snapshots,
                    lambda_over=lambda_over,
                    lambda_under=lambda_under,
                    theta_queue=theta_queue
                )
                
                results.append(result)
                
                # Update best result
                if result['avg_fill_price'] < best_avg_price:
                    best_avg_price = result['avg_fill_price']
                    best_result = result
                    best_params = result['parameters']
    
    end_time = time.time()
    print(f"Parameter search completed, time taken: {end_time - start_time:.2f} seconds")
    
    return best_result, best_params, results

def calculate_savings_bps(optimal_price: float, baseline_price: float) -> float:
    """Calculate savings in basis points"""
    return (baseline_price - optimal_price) / baseline_price * 10000  # 10000 basis points = 100%

def plot_cumulative_cost(optimal_result: Dict[str, Any], best_ask_result: Dict[str, Any], 
                        twap_result: Dict[str, Any], vwap_result: Dict[str, Any], 
                        output_file: str = 'results.png'):
    """Generate cumulative cost plot"""
    # Prepare cumulative cost data for each strategy
    strategies = {
        'Optimal (Cont-Kukanov)': optimal_result,
        'Best Ask': best_ask_result,
        'TWAP': twap_result,
        'VWAP': vwap_result
    }
    
    plt.figure(figsize=(12, 6))
    
    for name, result in strategies.items():
        history = result['execution_history']
        if not history:
            continue
            
        # Sort by timestamp
        history = sorted(history, key=lambda x: x['timestamp'])
        
        # Calculate cumulative costs
        timestamps = [h['timestamp'] for h in history]
        cumulative_costs = []
        running_cost = 0
        
        for h in history:
            running_cost += h['cost']
            cumulative_costs.append(running_cost)
        
        # Plot the curve
        plt.plot(timestamps, cumulative_costs, label=f"{name} (${result['avg_fill_price']:.2f}/share)")
    
    plt.title('Cumulative Execution Cost Comparison')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Cost ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    print(f"Saved cumulative cost plot to: {output_file}")

def main():
    start_time = time.time()
    
    # Load market data
    data = load_data('l1_day.csv')
    
    # Prepare market snapshots
    snapshots = prepare_snapshots(data)
    print(f"Data preparation completed, {len(snapshots)} market snapshots in total")
    
    # Parameter search
    optimal_result, best_params, all_results = parameter_search(snapshots)
    
    # Run baseline strategies
    best_ask_result = backtest_best_ask_strategy(snapshots)
    twap_result = backtest_twap_strategy(snapshots)
    vwap_result = backtest_vwap_strategy(snapshots)
    
    # Calculate savings compared to baseline strategies
    savings_vs_best_ask = calculate_savings_bps(optimal_result['avg_fill_price'], best_ask_result['avg_fill_price'])
    savings_vs_twap = calculate_savings_bps(optimal_result['avg_fill_price'], twap_result['avg_fill_price'])
    savings_vs_vwap = calculate_savings_bps(optimal_result['avg_fill_price'], vwap_result['avg_fill_price'])
    
    # Create result JSON
    result = {
        "best_parameters": {
            "lambda_over": best_params['lambda_over'],
            "lambda_under": best_params['lambda_under'],
            "theta_queue": best_params['theta_queue']
        },
        "cont_kukanov_strategy": {
            "total_cash_spent": optimal_result['total_cash_spent'],
            "avg_fill_price": optimal_result['avg_fill_price']
        },
        "best_ask_baseline": {
            "total_cash_spent": best_ask_result['total_cash_spent'],
            "avg_fill_price": best_ask_result['avg_fill_price']
        },
        "twap_baseline": {
            "total_cash_spent": twap_result['total_cash_spent'],
            "avg_fill_price": twap_result['avg_fill_price']
        },
        "vwap_baseline": {
            "total_cash_spent": vwap_result['total_cash_spent'],
            "avg_fill_price": vwap_result['avg_fill_price']
        },
        "savings_bps": {
            "vs_best_ask": savings_vs_best_ask,
            "vs_twap": savings_vs_twap,
            "vs_vwap": savings_vs_vwap
        }
    }
    
    # Output results
    print(json.dumps(result, indent=2))
    
    # Generate cumulative cost plot
    plot_cumulative_cost(optimal_result, best_ask_result, twap_result, vwap_result)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 