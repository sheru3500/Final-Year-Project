
                data = yf.download(tickers=yf_ticker_name, period='10y')
            else:
                data = yf.download(tickers=yf_ticker_name, per