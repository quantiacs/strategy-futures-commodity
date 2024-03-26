# Using IMF Commodity Data

This trading strategy is designed for the [Quantiacs](https://quantiacs.com/contest) platform, which hosts competitions
for trading algorithms. Detailed information about the competitions is available on
the [official Quantiacs website](https://quantiacs.com/contest).

## How to Run the Strategy

### In an Online Environment

The strategy can be executed in an online environment using Jupiter or JupiterLab on
the [Quantiacs personal dashboard](https://quantiacs.com/personalpage/homepage). To do this, clone the template in your
personal account.

### In a Local Environment

To run the strategy locally, you need to install the [Quantiacs Toolbox](https://github.com/quantiacs/toolbox).

## Strategy Overview

The "Using IMF Commodity Data" notebook showcases the integration of International Monetary Fund (IMF) commodity data
into a trading algorithm. It guides through accessing and utilizing this data to inform trading decisions, focusing on a
strategy that trades futures based on the comparison of Gold futures prices and the IMF's Gold commodity prices. Key
steps include loading the necessary data, implementing a strategy that goes long under specific conditions related to
Gold prices, and backtesting the strategy using Quantiacs' toolkit. This example demonstrates the value of incorporating
external economic indicators into algorithmic trading strategies.

```python
import xarray as xr
import numpy as np
import pandas as pd

import qnt.backtester as qnbt
import qnt.data as qndata
# commodity listing
commodity_list = qndata.imf_load_commodity_list()
pd.DataFrame(commodity_list)
def load_data(period):
    # load Futures Gold and Gold data:
    futures   = qndata.futures_load_data(assets=['F_GC'], tail=period, dims=('time','field','asset'))
    commodity = qndata.imf_load_commodity_data(assets=['PGOLD'], tail=period).isel(asset=0)
    return dict(commodity=commodity, futures=futures), futures.time.values


def window(data, max_date: np.datetime64, lookback_period: int):
    # build sliding window for rolling evaluation:
    min_date = max_date - np.timedelta64(lookback_period, 'D')
    return dict(
        futures   = data['futures'].sel(time=slice(min_date, max_date)),
        commodity = data['commodity'].sel(time=slice(min_date, max_date))
    )


def strategy(data):
    # strategy uses both Futures Gold and Gold data:
    close = data['futures'].sel(field='close')
    commodity = data['commodity']
    if commodity.isel(time=-1) > commodity.isel(time=-2) and close.isel(time=-1) > close.isel(time=-20):
        return xr.ones_like(close.isel(time=-1))
    else:
        return xr.zeros_like(close.isel(time=-1))


weights = qnbt.backtest(
    competition_type='futures',
    load_data=load_data,
    window=window,
    lookback_period=365,
    start_date='2006-01-01',
    strategy=strategy,
    analyze=True,
    build_plots=True
)
```

More examples of use data from the International Monetary Fund (IMF) in [documentation](https://quantiacs.com/documentation/en/data/imf.html).
