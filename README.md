# 📈 Dynamic Investment Strategies Through Market Classification and Volatility
**Machine Learning & Bayesian Markov Switching for Real-Time Portfolio Optimization**

---

## 🧠 Overview

This repository implements a dynamic investment strategy that outperforms static portfolio allocation methods by:
- **Clustering market volatility states** using K-Means
- **Predicting future market states** with a Bayesian Markov Switching model (Dirichlet priors + Gibbs Sampling)
- **Dynamically reallocating assets** based on the optimal portfolio method for each state

The methodology is inspired by the paper:

> **Dynamic Investment Strategies Through Market Classification and Volatility: A Machine Learning Approach**  
> *Jinhui Li, Wenjia Xie, Luis Seco – March 2025*

---

## 💡 Key Features

- 📊 **Market Classification**  
  Segment markets into 10 volatility-based states using K-Means on 22-day rolling volatility.

- 🧮 **Bayesian Markov Switching Model**  
  Forecast transitions between volatility states using Dirichlet priors and Gibbs sampling.

- 💼 **Portfolio Strategies Implemented**
  - Equal Weight (1/N)
  - Minimum Variance (MinVar)
  - Equal Risk Contribution (ERC / Risk Parity)
  - Maximum Diversification (MaxDiv)

- 🔁 **Dynamic Portfolio Allocation**  
  Automatically switch to the best-performing strategy for each market state using transition probabilities.

- 📈 **Performance Evaluation**
  - Cumulative return over time
  - Annual return, volatility
  - Sharpe Ratio

---

## 📁 Project Structure

