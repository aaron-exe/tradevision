"""
Stock Universe Data — Multi-market ticker definitions.
Used by InvestmentPlanner for candidate selection.
"""

# Format: (ticker, name, sector, cap_tier, div_tier)
# cap_tier: 'mega'|'large'|'mid'   div_tier: 'high'|'medium'|'low'|'none'

_US = [
    # Technology
    ('AAPL','Apple Inc.','Technology','mega','low'),
    ('MSFT','Microsoft Corp.','Technology','mega','low'),
    ('GOOGL','Alphabet Inc.','Technology','mega','none'),
    ('NVDA','NVIDIA Corp.','Technology','mega','low'),
    ('META','Meta Platforms','Technology','mega','low'),
    ('AVGO','Broadcom Inc.','Technology','mega','medium'),
    ('ADBE','Adobe Inc.','Technology','large','none'),
    ('CRM','Salesforce Inc.','Technology','large','none'),
    ('ORCL','Oracle Corp.','Technology','large','medium'),
    ('CSCO','Cisco Systems','Technology','large','medium'),
    ('AMD','AMD Inc.','Technology','large','none'),
    ('INTC','Intel Corp.','Technology','large','medium'),
    ('QCOM','Qualcomm Inc.','Technology','large','medium'),
    ('TXN','Texas Instruments','Technology','large','medium'),
    ('NOW','ServiceNow Inc.','Technology','large','none'),
    ('INTU','Intuit Inc.','Technology','large','low'),
    ('AMAT','Applied Materials','Technology','large','low'),
    # Consumer Discretionary
    ('AMZN','Amazon.com Inc.','Consumer Cyclical','mega','none'),
    ('TSLA','Tesla Inc.','Consumer Cyclical','mega','none'),
    ('HD','Home Depot Inc.','Consumer Cyclical','large','medium'),
    ('MCD',"McDonald's Corp.",'Consumer Cyclical','large','medium'),
    ('NKE','Nike Inc.','Consumer Cyclical','large','low'),
    ('SBUX','Starbucks Corp.','Consumer Cyclical','large','medium'),
    ('TJX','TJX Companies','Consumer Cyclical','large','low'),
    ('LOW',"Lowe's Cos.",'Consumer Cyclical','large','medium'),
    ('BKNG','Booking Holdings','Consumer Cyclical','large','none'),
    ('GM','General Motors','Consumer Cyclical','mid','low'),
    ('F','Ford Motor Co.','Consumer Cyclical','mid','medium'),
    # Healthcare
    ('UNH','UnitedHealth Group','Healthcare','mega','low'),
    ('JNJ','Johnson & Johnson','Healthcare','mega','medium'),
    ('LLY','Eli Lilly & Co.','Healthcare','mega','low'),
    ('PFE','Pfizer Inc.','Healthcare','large','high'),
    ('ABT','Abbott Labs','Healthcare','large','medium'),
    ('TMO','Thermo Fisher Sci.','Healthcare','large','low'),
    ('ABBV','AbbVie Inc.','Healthcare','large','high'),
    ('MRK','Merck & Co.','Healthcare','large','medium'),
    ('AMGN','Amgen Inc.','Healthcare','large','medium'),
    ('GILD','Gilead Sciences','Healthcare','large','high'),
    ('ISRG','Intuitive Surgical','Healthcare','large','none'),
    ('CVS','CVS Health Corp.','Healthcare','large','medium'),
    # Financial Services
    ('JPM','JPMorgan Chase','Financial Services','mega','medium'),
    ('BAC','Bank of America','Financial Services','large','medium'),
    ('V','Visa Inc.','Financial Services','mega','low'),
    ('MA','Mastercard Inc.','Financial Services','large','low'),
    ('GS','Goldman Sachs','Financial Services','large','medium'),
    ('MS','Morgan Stanley','Financial Services','large','medium'),
    ('BLK','BlackRock Inc.','Financial Services','large','medium'),
    ('AXP','American Express','Financial Services','large','low'),
    ('USB','U.S. Bancorp','Financial Services','mid','high'),
    ('SCHW','Charles Schwab','Financial Services','large','low'),
    # Consumer Defensive
    ('PG','Procter & Gamble','Consumer Defensive','mega','medium'),
    ('KO','Coca-Cola Co.','Consumer Defensive','mega','high'),
    ('PEP','PepsiCo Inc.','Consumer Defensive','large','medium'),
    ('COST','Costco Wholesale','Consumer Defensive','large','low'),
    ('WMT','Walmart Inc.','Consumer Defensive','mega','low'),
    ('CL','Colgate-Palmolive','Consumer Defensive','large','medium'),
    ('MDLZ','Mondelez Intl.','Consumer Defensive','large','medium'),
    ('MO','Altria Group','Consumer Defensive','large','high'),
    ('PM','Philip Morris Intl.','Consumer Defensive','large','high'),
    ('GIS','General Mills','Consumer Defensive','mid','high'),
    # Energy
    ('XOM','Exxon Mobil Corp.','Energy','mega','high'),
    ('CVX','Chevron Corp.','Energy','mega','high'),
    ('COP','ConocoPhillips','Energy','large','medium'),
    ('EOG','EOG Resources','Energy','large','medium'),
    ('SLB','Schlumberger Ltd.','Energy','large','medium'),
    ('MPC','Marathon Petroleum','Energy','large','medium'),
    ('VLO','Valero Energy','Energy','mid','high'),
    ('OXY','Occidental Petrol.','Energy','mid','low'),
    # Communication Services
    ('NFLX','Netflix Inc.','Communication Services','large','none'),
    ('DIS','Walt Disney Co.','Communication Services','large','low'),
    ('CMCSA','Comcast Corp.','Communication Services','large','medium'),
    ('TMUS','T-Mobile US','Communication Services','large','low'),
    ('T','AT&T Inc.','Communication Services','large','high'),
    ('VZ','Verizon Comms.','Communication Services','large','high'),
    # Industrials
    ('CAT','Caterpillar Inc.','Industrials','large','medium'),
    ('GE','GE Aerospace','Industrials','large','low'),
    ('HON','Honeywell Intl.','Industrials','large','medium'),
    ('UNP','Union Pacific','Industrials','large','medium'),
    ('BA','Boeing Co.','Industrials','large','none'),
    ('RTX','RTX Corp.','Industrials','large','medium'),
    ('DE','Deere & Co.','Industrials','large','low'),
    ('LMT','Lockheed Martin','Industrials','large','medium'),
    ('ETN','Eaton Corp.','Industrials','large','low'),
    # Utilities
    ('NEE','NextEra Energy','Utilities','large','medium'),
    ('DUK','Duke Energy','Utilities','large','high'),
    ('SO','Southern Co.','Utilities','large','high'),
    ('D','Dominion Energy','Utilities','large','high'),
    ('AEP','Amer. Electric Pwr','Utilities','large','high'),
    ('SRE','Sempra','Utilities','large','high'),
    ('XEL','Xcel Energy','Utilities','mid','high'),
    # Real Estate
    ('AMT','American Tower','Real Estate','large','medium'),
    ('PLD','Prologis Inc.','Real Estate','large','medium'),
    ('CCI','Crown Castle','Real Estate','large','high'),
    ('PSA','Public Storage','Real Estate','large','high'),
    ('O','Realty Income','Real Estate','large','high'),
    ('SPG','Simon Property','Real Estate','large','high'),
    # Materials
    ('LIN','Linde plc','Materials','large','low'),
    ('APD','Air Products','Materials','large','medium'),
    ('SHW','Sherwin-Williams','Materials','large','low'),
    ('NEM','Newmont Corp.','Materials','large','medium'),
    ('FCX','Freeport-McMoRan','Materials','mid','low'),
    ('NUE','Nucor Corp.','Materials','mid','medium'),
]

_IN = [
    # IT
    ('TCS.NS','Tata Consultancy','Technology','mega','medium'),
    ('INFY.NS','Infosys Ltd.','Technology','mega','medium'),
    ('WIPRO.NS','Wipro Ltd.','Technology','large','medium'),
    ('HCLTECH.NS','HCL Technologies','Technology','large','medium'),
    ('TECHM.NS','Tech Mahindra','Technology','large','medium'),
    ('LTIM.NS','LTIMindtree Ltd.','Technology','large','low'),
    # Banks
    ('HDFCBANK.NS','HDFC Bank','Financial Services','mega','low'),
    ('ICICIBANK.NS','ICICI Bank','Financial Services','mega','low'),
    ('SBIN.NS','State Bank India','Financial Services','mega','low'),
    ('KOTAKBANK.NS','Kotak Mahindra','Financial Services','large','low'),
    ('AXISBANK.NS','Axis Bank','Financial Services','large','low'),
    ('INDUSINDBK.NS','IndusInd Bank','Financial Services','mid','low'),
    ('BANKBARODA.NS','Bank of Baroda','Financial Services','mid','medium'),
    # FMCG
    ('HINDUNILVR.NS','Hindustan Unilever','Consumer Defensive','mega','high'),
    ('ITC.NS','ITC Ltd.','Consumer Defensive','mega','high'),
    ('NESTLEIND.NS','Nestle India','Consumer Defensive','large','medium'),
    ('BRITANNIA.NS','Britannia Ind.','Consumer Defensive','large','medium'),
    ('DABUR.NS','Dabur India','Consumer Defensive','mid','medium'),
    ('MARICO.NS','Marico Ltd.','Consumer Defensive','mid','medium'),
    # Pharma
    ('SUNPHARMA.NS','Sun Pharma Ind.','Healthcare','large','low'),
    ('DRREDDY.NS',"Dr. Reddy's Labs",'Healthcare','large','low'),
    ('CIPLA.NS','Cipla Ltd.','Healthcare','large','low'),
    ('DIVISLAB.NS',"Divi's Labs",'Healthcare','large','low'),
    ('APOLLOHOSP.NS','Apollo Hospitals','Healthcare','large','none'),
    # Auto
    ('MARUTI.NS','Maruti Suzuki','Consumer Cyclical','large','low'),
    ('TATAMOTORS.NS','Tata Motors','Consumer Cyclical','large','low'),
    ('BAJAJ-AUTO.NS','Bajaj Auto','Consumer Cyclical','large','medium'),
    ('HEROMOTOCO.NS','Hero MotoCorp','Consumer Cyclical','large','medium'),
    ('EICHERMOT.NS','Eicher Motors','Consumer Cyclical','large','low'),
    # Energy & Infra
    ('RELIANCE.NS','Reliance Ind.','Energy','mega','low'),
    ('ONGC.NS','ONGC Ltd.','Energy','large','high'),
    ('BPCL.NS','Bharat Petroleum','Energy','mid','high'),
    ('NTPC.NS','NTPC Ltd.','Utilities','large','high'),
    ('POWERGRID.NS','Power Grid Corp.','Utilities','large','high'),
    # Metals
    ('TATASTEEL.NS','Tata Steel','Materials','large','medium'),
    ('HINDALCO.NS','Hindalco Ind.','Materials','large','low'),
    ('JSWSTEEL.NS','JSW Steel','Materials','large','low'),
    ('COALINDIA.NS','Coal India','Materials','large','high'),
    # Financial Services
    ('BAJFINANCE.NS','Bajaj Finance','Financial Services','large','low'),
    ('BAJAJFINSV.NS','Bajaj Finserv','Financial Services','large','low'),
    ('HDFCLIFE.NS','HDFC Life Ins.','Financial Services','large','low'),
    ('SBILIFE.NS','SBI Life Ins.','Financial Services','mid','low'),
    # Conglomerate / Others
    ('LT.NS','Larsen & Toubro','Industrials','large','low'),
    ('TITAN.NS','Titan Company','Consumer Cyclical','large','low'),
    ('ASIANPAINT.NS','Asian Paints','Materials','large','low'),
    ('ULTRACEMCO.NS','UltraTech Cement','Materials','large','low'),
    ('ADANIENT.NS','Adani Enterprises','Industrials','large','none'),
    ('ADANIPORTS.NS','Adani Ports','Industrials','large','low'),
]

_INTL = [
    # Europe
    ('ASML','ASML Holding','Technology','mega','low'),
    ('SAP','SAP SE','Technology','mega','low'),
    ('NVO','Novo Nordisk','Healthcare','mega','low'),
    ('AZN','AstraZeneca','Healthcare','mega','medium'),
    ('SHEL','Shell plc','Energy','mega','high'),
    ('UL','Unilever plc','Consumer Defensive','large','high'),
    ('NVS','Novartis AG','Healthcare','large','high'),
    ('SNY','Sanofi SA','Healthcare','large','medium'),
    ('DEO','Diageo plc','Consumer Defensive','large','high'),
    ('RIO','Rio Tinto','Materials','large','high'),
    ('BP','BP plc','Energy','large','high'),
    ('TTE','TotalEnergies','Energy','large','high'),
    # Asia-Pacific
    ('TM','Toyota Motor','Consumer Cyclical','mega','medium'),
    ('SONY','Sony Group','Technology','large','low'),
    ('HMC','Honda Motor','Consumer Cyclical','large','medium'),
    ('MUFG','Mitsubishi UFJ','Financial Services','large','medium'),
    # ADRs - Emerging
    ('TSM','Taiwan Semi (TSMC)','Technology','mega','medium'),
    ('BABA','Alibaba Group','Consumer Cyclical','large','none'),
    ('PDD','PDD Holdings','Consumer Cyclical','large','none'),
    ('JD','JD.com Inc.','Consumer Cyclical','large','none'),
    ('VALE','Vale SA','Materials','large','high'),
    ('PBR','Petrobras','Energy','large','high'),
    ('NU','Nu Holdings','Financial Services','mid','none'),
    ('MELI','MercadoLibre','Consumer Cyclical','large','none'),
    ('SE','Sea Limited','Technology','mid','none'),
    ('GRAB','Grab Holdings','Technology','mid','none'),
]


def _build(raw):
    return {t[0]: {'name': t[1], 'sector': t[2], 'cap_tier': t[3], 'div_tier': t[4]} for t in raw}

US_STOCKS = _build(_US)
INDIAN_STOCKS = _build(_IN)
INTERNATIONAL_STOCKS = _build(_INTL)

MARKET_UNIVERSES = {
    'US Stocks': US_STOCKS,
    'Indian Stocks': INDIAN_STOCKS,
    'International': INTERNATIONAL_STOCKS,
}

MARKET_CHOICES = list(MARKET_UNIVERSES.keys()) + ['Mixed Global']
STRATEGY_CHOICES = ['AI Optimized', 'Growth', 'Value', 'Dividend', 'Momentum']
GOAL_MODES = ['Target Return %', 'Target Final Value', 'Target Profit']

def get_sectors_for_market(market):
    """Return sorted list of sectors available in a given market."""
    if market == 'Mixed Global':
        all_stocks = {}
        for u in MARKET_UNIVERSES.values():
            all_stocks.update(u)
        return sorted(set(v['sector'] for v in all_stocks.values()))
    universe = MARKET_UNIVERSES.get(market, US_STOCKS)
    return sorted(set(v['sector'] for v in universe.values()))

def get_universe(market):
    """Return the stock dict for a market, combining all for Mixed Global."""
    if market == 'Mixed Global':
        combined = {}
        for u in MARKET_UNIVERSES.values():
            combined.update(u)
        return combined
    return dict(MARKET_UNIVERSES.get(market, US_STOCKS))

def get_universe_size(market):
    return len(get_universe(market))
