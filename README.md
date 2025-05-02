# Cont & Kukanov Order Router Backtesting
## name:Leqi Chen
This project implements a backtesting framework for a Smart Order Router based on the static cost model introduced by Cont & Kukanov. The router determines how to optimally allocate a 5,000-share buy order across multiple trading venues based on best ask prices and quantities to minimize total cost.

## Required Libraries

The following Python libraries are required to run this project:
```bash
numpy
pandas
matplotlib
```

You can install them using pip:
```bash
pip install numpy pandas matplotlib
```

## Code Structure

- `backtest.py`: Main backtesting script containing the following key components:
  - `Venue` class: Represents a trading venue with attributes like venue ID, ask price, available size, etc.
  - `allocate` function: Implements the Cont-Kukanov allocation algorithm
  - `compute_cost` function: Calculates the expected cost of a given allocation plan
  - Backtesting framework: Includes data loading, parameter search, and result generation
  - Baseline strategies: Includes Best Ask, TWAP, and VWAP implementations
  - Result visualization: Generates a cumulative cost comparison chart

## Parameter Search

This implementation uses grid search to find the optimal risk parameter combination:

- `lambda_over`: Cost penalty for over-execution, search range: [0.001, 0.005, 0.01, 0.05, 0.1]
- `lambda_under`: Cost penalty for under-execution, search range: [0.001, 0.005, 0.01, 0.05, 0.1]
- `theta_queue`: Queue risk penalty coefficient, search range: [0.0001, 0.0005, 0.001, 0.005, 0.01]

These parameter ranges were chosen based on the estimated order size and price levels, ensuring that the algorithm balances both price optimization and execution risk.

## Execution

```bash
python backtest.py
```

Upon execution, the script will:
1. Load and process market data
2. Conduct parameter search to find the optimal combination
3. Run three baseline strategies for comparison
4. Output results in JSON format, including best parameters and savings relative to baselines
5. Generate a cumulative cost comparison chart (results.png)

## Idea for Improving Fill Realism

The current implementation assumes trades can be executed fully at the posted ask price and displayed size, but real markets typically involve factors like slippage and queue positioning. An improvement would be:

Introducing a **Queue Position Model**: Currently, I assume orders can be executed immediately and completely, but in reality, they need to wait in a queue. A queue position simulation based on venue liquidity, order size, and historical fill rates could be added, for example:

1. Estimate average queue length and execution rate for each venue
2. Assign a probabilistic queue position to our orders
3. Calculate expected execution delay based on this position
4. If the expected delay exceeds a threshold, reallocate the order or accept a higher price for faster execution

This approach would make the backtesting more closely resemble actual trading conditions, thereby improving the reliability of the strategy in live trading. 
