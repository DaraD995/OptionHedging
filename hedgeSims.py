import numpy as np
from scipy.stats import norm
import pandas as pd
import math as m
import matplotlib.pyplot as plt


def bs_call_price(S, E, r, vol, T, t):
    tau = T - t
    if tau <= 0:
        return max(S - E, 0.0)

    d1 = (np.log(S / E) + (r + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)

    C = S * norm.cdf(d1) - E * np.exp(-r * tau) * norm.cdf(d2)
    return C

def delta_call(S,E, r, vol, T, t):
    tau  = T-t
    d1 = (np.log(S / E) + (r + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    return norm.cdf(d1)

def vega_call(S,E, r, vol, T, t):

    ans=bs_call_price(S, E, r, vol+0.01, T, t)-bs_call_price(S, E, r, vol, T, t)

    return ans

def expS(S, mu, T, t):
    return S*m.exp(mu*(T-t))

seed = 100
np.random.seed(seed)

"""
Monte Carlo simulation of delta-hedging a European call option.

For each of `nsims` simulated stock paths (generated via geometric Brownian
motion using the lognormal exact solution), the option is priced with
Black-Scholes and delta-hedged at discrete intervals (dt = weekly). At each
rebalancing step, the stock position is adjusted to -delta, with the cash
account updated to reflect interest earned and the cost of trading stock to
maintain the hedge. This produces a self-financing replicating portfolio
whose value (and cumulative P&L, 'cpl') is tracked over the life of the option.

The terminal cumulative P&L across all simulations is plotted as a histogram,
showing the distribution of hedging error that arises from discrete (rather
than continuous) rebalancing. Terminal spot price and cumulative P&L are also
reported with their sample means and standard errors across simulations.
"""

nsims=1000
results=np.zeros((nsims,3))

for this_sim_index in range(0,nsims):

    data = pd.DataFrame(columns=['time', 'tau', 'stock_price', 'option_price',
                                 'option_delta', 'position_stock', 'position_option', 'position_cash',
                                 'value', 'pl', 'cpl'])
    if this_sim_index%100==0:
        print(f"sim {this_sim_index}")

    S = 100.0
    E = 100.0
    r = 0.05
    implied_vol = 0.2

    realized_vol=0.2

    T = 1
    t = 0

    priceC = bs_call_price(S, E, r, implied_vol, T, t)
    deltaC = delta_call(S, E, r, implied_vol, T, t)
    #vegaC = vega_call(S, E, r, implied_vol, T, t)
    posStock = -deltaC
    posOption = 1
    posCash = -priceC - posStock * S
    value = priceC * posOption + posStock * S + posCash

    data.loc[len(data)] = [t, T - t, S, priceC, deltaC, posStock, posOption, posCash, value, 0, 0]

    dt = 1 / 52
    mu = r

    for i in range(1, int(T/dt)):

        #dX = np.random.standard_normal() * m.sqrt(dt) #dX updated before dS
        #dS = S * vol * dX + S * mu * dt -- only works if dt is very small.
        #S+= dS

        S = S * m.exp( (mu - realized_vol*realized_vol/2)*dt + realized_vol * np.random.standard_normal() * m.sqrt(dt)) #works for any dt step

        t+=dt

        #reprice option
        priceC = bs_call_price(S, E, r, implied_vol, T, t)
        deltaC = delta_call(S, E, r, implied_vol, T, t)

        #update positions, recalculate value
        posStockOld = posStock
        posStock = -deltaC
        posOption = 1
        posCash = posCash*m.exp(r*dt) + (posStockOld - posStock)*S
        value = priceC * posOption + posStock * S + posCash
        pl = value - data['value'].iloc[i-1]
        cpl = pl + data['cpl'].iloc[i-1]

        #add to file
        data.loc[len(data)] = [t, T - t, S, priceC, deltaC, posStock, posOption,
                               posCash, value, pl ,
                               cpl]

    final_spot = S
    final_cpl = data["cpl"].iloc[-1]
    results[this_sim_index,0]=this_sim_index
    results[this_sim_index,1]=final_spot
    results[this_sim_index,2]=final_cpl

##plot overall data
plt.title('Histogram of cumulative PL with hedge freq =' +str(1/dt)+"per year")
plt.hist(results[:,2])
plt.show()
print(np.mean(results[:,1:],axis=0))
print(np.std(results[:,1:],axis=0))

spot_mean=np.mean(results[:,1])
spot_sd=np.std(results[:,1])
spot_err=spot_sd / m.sqrt(nsims)

cpl_mean=np.mean(results[:,2])
cpl_sd=np.std(results[:,2])
cpl_err=cpl_sd / m.sqrt(nsims)

print("Terminal S = " +str(spot_mean) +"+/-" +str(spot_err))
print("Terminal cpl = " +str(cpl_mean) +"+/-" +str(cpl_err))



