#!/usr/local/bin/python3.10 

import os
import time
import sys
import logging
import math
import time

from datetime import datetime, timedelta
from pandas import DataFrame
from pathlib import Path

from tinkoff.invest import InstrumentStatus, CandleInterval
from tinkoff.invest import Client, RequestError, PortfolioResponse, PositionsResponse, PortfolioPosition, AccessLevel
from tinkoff.invest import OrderDirection, OrderType, Quotation
from tinkoff.invest.services import Services
from tinkoff.invest.utils import now
from tinkoff.invest.caching.cache_settings import MarketDataCacheSettings
from tinkoff.invest.services import MarketDataCache

from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

"""
Для участия в конкурсе по разработке бота для трейдинга на баз tinkoff.invest (python)
https://tinkoff.github.io/investAPI
https://tinkoff.github.io/investAPI/faq_custom_types/
https://azzrael.ru/api-v2-tinkoff-invest-get-candles-python
https://github.com/AzzraelCode/api_v2_tinvest/blob/main/v2_portfolio/__init__.py
https://github.com/AzzraelCode/api_v2_tinvest/blob/main/v2_portfolio/__init__.py#L104
https://technical-analysis-library-in-python.readthedocs.io/en/latest/index.html

"""

def get_list_shares(local_client):
    """
    Получаем список акций
    :param local_client:
    :return: 
    """
    try:
        shares = local_client.instruments.shares().instruments

    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))

    return shares


def get_list_futures(local_client):
    """
    Получаем список фьючерсов
    :param local_client:
    :return:
    """
    try:
        futs = local_client.instruments.futures().instruments

    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))

    return futs


def get_list_bonds(local_client):
    """
    Получаем список бондов
    :param local_client:
    :return:
    """
    try:
        bonds = local_client.instruments.bonds().instruments

    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))

    return bonds


def get_candles(local_client, local_figi, local_day, local_interval):
    """
    Получаем значение свечей для отдельного инструмента (используем figi)
    :param local_client:
    :param local_figi:
    :param local_day:
    :param local_interval:
    :return: 
    """
    try:
        # кэши пока отключаем, ждем ответа на issue:
        #   https://github.com/Tinkoff/invest-python/issues/18#issuecomment-1122386454
        # settings = MarketDataCacheSettings(base_cache_dir=Path(".market_data_cache"))
        # market_data_cache = MarketDataCache(settings=settings, services=local_client)

        candles = local_client.get_all_candles(
            figi=local_figi,
            from_=now() - timedelta(days=local_day),
            interval=local_interval,
        )

    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))

    return candles


def cast_money(client, v, to_rub=True):
    """
    Преоборазование в float (если нужно, то конвертация из USD в RUB)
    :param v: Quotation from API
    :return: Price in standart float type
    """
    r = v.units + v.nano / 1e9 # nano - 9 нулей
    if to_rub and hasattr(v, 'currency') and getattr(v, 'currency') == 'usd':
        r *= get_usdrur(client)

    return r


def create_df(client, candles):
    """
    Создаеv датафрейм для списка свечей
    :param dict candles: List of candles from API
    :return: DataFrame with candles
    """
    df = DataFrame([{
            'time': c.time,
            'volume': c.volume,
            'open': cast_money(client, c.open),
            'close': cast_money(client, c.close),
            'high': cast_money(client, c.high),
            'low': cast_money(client, c.low),
        } for c in candles])

    return df


def get_usdrur(client):
    """
    Получаем курс доллара
    :return:
    """

    usdrur = 0
    try:
        u = client.market_data.get_last_prices(figi=['USD000UTSTOM'])
        usdrur = cast_money(client, u.last_prices[0].price)
    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))


    return usdrur

def get_accounts(client):
    """
    Получаем все аккаунты и буду использовать только те
    котoрые текущий токен может хотя бы читать, остальные аккаунты пропускаем
    :return:
    """
    accounts = [] 

    r = client.users.get_accounts()
    for acc in r.accounts:
        if acc.access_level != AccessLevel.ACCOUNT_ACCESS_LEVEL_NO_ACCESS:
            accounts.append(acc.id)

    return accounts



def get_portfolio_list(client, account_id: str):
    """
    Преобразуем PortfolioResponse в список
    :param account_id:
    :return:
    """
    r = client.operations.get_portfolio(account_id=account_id)
    if len(r.positions) < 1:
        return None

    mas = []
    for p in r.positions:
        mas.append(portfolio_pose_todict(client, p))

    return mas


def portfolio_pose_todict(client, p : PortfolioPosition):
    """
    Преобразуем PortfolioPosition в dict
    :param p:
    :return:
    """
    r = {
            'figi': p.figi,
            'quantity': cast_money(client, p.quantity),
            'expected_yield': cast_money(client, p.expected_yield),
            'instrument_type': p.instrument_type,
            'average_buy_price': cast_money(client, p.average_position_price),
            'current_price': cast_money(client, p.current_price),
            'currency': p.average_position_price.currency,
            'current_nkd': cast_money(client, p.current_nkd),
    }

    if r['currency'] == 'usd':
        r['expected_yield'] *= get_usdrur(client)

    return r


def get_trading_status(local_client, local_figi):
    """
    Проверка доступности инструмента для трейдинга
    см. https://tinkoff.github.io/investAPI/marketdata/#securitytradingstatus
    :return
    """

    stat_ok = 'SecurityTradingStatus.SECURITY_TRADING_STATUS_NORMAL_TRADING'
    stat_not = 'SecurityTradingStatus.SECURITY_TRADING_STATUS_NOT_AVAILABLE_FOR_TRADING'
    try:
        stat = local_client.market_data.get_trading_status(figi=local_figi)
        if not (str(stat.trading_status) == stat_ok):
            logger.info(stat)
            return False

    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))

    return True


def backtest(client, start_depo :float, ticker :str, df :DataFrame, lot :int, usd: bool ):
    """
    Тестирование гипотез на истории значений котировок (бэк-тест)
    :return
    """

    # константы для расчета индикаторов
    wx = 20
    sma_period = 14

    depo = 0
    val = 'rub'
    if usd:
        usdrur = get_usdrur(client)
        if usdrur > 0.0:
            val = 'usd'
            depo = round(start_depo/usdrur, 2)
    else:
        depo = start_depo
    print('\nStart depo: ', depo, val) 
    price = df['close'].iloc[-1]
    max_nums = math.ceil(depo/(price*lot))
    qbuy, qsell = int(max_nums*0.1), int(max_nums*0.1)
    if (qbuy == 0):
        qbuy, qsell = 1, 1
    prc_buy, prc_sell, prc_min, kx = 0.5, 0.5, 0.5, 1.2
    nums = 0
    avg_price = 0.0
    comm = 0.0005

    c_min, c_max = 0, 0
    num_sell_trades, num_buy_trades = 0, 0

    for i in range(wx, df.size-1, 1):
        dfc = df.copy(deep=True)
        dfc = dfc.iloc[0:i]

        last_price = dfc['close'].iloc[-1]
        last_time = dfc['time'].iloc[-1]

        # завершаем бэк-тест перед праздниками (?)
        # if '2022-05-06' in str(last_time):
        #    break

        
        if (round(last_price, 6) <= round(dfc['close'].iloc[-2], 6)):
            c_min += 1
            c_max = 0
        if (round(last_price, 6) > round(dfc['close'].iloc[-2], 6)):
            c_min = 0
            c_max += 1

        # Инициализируем индикатор  'Bollinger Bands'
        indicator_bb = BollingerBands(close=dfc["close"], window=wx, window_dev=2)

        # Записываем значения индикатора 'Bollinger Bands' для оценки
        # df['bb_bbm'] = indicator_bb.bollinger_mavg() # пока среднее не используется
        dfc['bb_bbh'] = indicator_bb.bollinger_hband()
        dfc['bb_bbl'] = indicator_bb.bollinger_lband()

        # Получаем значение индикатора RSI
        dfc['rsi'] = RSIIndicator(close=dfc["close"], window=sma_period, fillna=False).rsi()
        
        rsi = dfc['rsi'].iloc[-1]
        bbl = dfc['bb_bbl'].iloc[-1] 
        bbh = dfc['bb_bbh'].iloc[-1] 


        if (max_nums > 0) and (nums > 0):
            prc_buy, prc_sell = (nums*1.0/max_nums + 0.5)*kx, (2.0 - nums*1.0/max_nums)*kx

        if prc_buy < prc_min:
            prc_buy = prc_min
        if prc_sell < prc_min:
            prc_sell = prc_min

        
        # Условия выполнения гипотезы (испольуются индикаторы RSI, BB и свечи)
        # Если rsi < 30 и значение цены закрытия свечи меньше bbl, то проверяем avg_price: buy
        if (last_price <= bbl) and (rsi <= 30) and (c_min > 5):
            quant = math.ceil(qbuy*(nums/max_nums + 1.2))
            if (nums+quant) > max_nums:
                quant = max_nums - nums
            pr_need_buy = False
            if (quant > 0) and (last_price > 0.0):
                if (avg_price == 0):
                    pr_need_buy = True
                elif (100.0*(avg_price/last_price - 1.0) >= prc_buy):
                    pr_need_buy = True

            if pr_need_buy:
                nums += quant
                avg_price = (quant*last_price + (nums-quant)*avg_price)/nums
                depo = depo - (quant*lot*last_price + quant*lot*last_price*comm)
                #prc_buy *= kx
                #prc_sell /= kx
                #if prc_sell < prc_min:
                #    prc_sell = prc_min
                num_buy_trades += 1

        # Иначе, если rsi > 70 и значение цены закрытия свечи больше bbh, то проверяем avg_price: sell
        elif (last_price >= bbh) and (rsi >= 70) and (c_max > 5):
            quant = math.ceil(qsell*(nums/max_nums + 1.2))
            if (quant > nums):
                quant = nums
            pr_need_sell = False
            if (quant > 0) and (last_price > 0.0) and (avg_price > 0):
                if (100.0*(last_price/avg_price - 1.0) >= prc_sell):
                    pr_need_sell = True

            if pr_need_sell:
                nums -= quant
                depo = depo + quant*lot*last_price - quant*lot*last_price*comm
                # prc_sell *= kx
                # prc_buy /= kx
                # if prc_buy < prc_min:
                #     prc_buy = prc_min
                if (nums == 0):
                    prc_buy, prc_sell = prc_min, prc_min
                    avg_price = 0
                num_sell_trades += 1


    now_price = df['close'].iloc[-1]
    print(ticker + ' result: ', str(depo+nums*lot*now_price), val)
    print('num_trades: %s\n' %(str(num_sell_trades + num_buy_trades)))


def make_buy_order(client, fg, quant, acc_id):
    try:
        # Рыночная, без указания цены (по лучшей доступной для объема)
        r = client.orders.post_order(
            order_id=str(datetime.utcnow().timestamp()),
            figi=fg,
            quantity=int(quant),
            account_id=acc_id,
            direction=OrderDirection.ORDER_DIRECTION_BUY,
            order_type=OrderType.ORDER_TYPE_MARKET
        )
        print(r)
        return(True)

    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))


def make_sell_order(client, fg, quant, acc_id):
    try:
        # Рыночная, без указания цены (по лучшей доступной для объема)
        r = client.orders.post_order(
            order_id=str(datetime.utcnow().timestamp()),
            figi=fg,
            quantity=int(quant),
            account_id=acc_id,
            direction=OrderDirection.ORDER_DIRECTION_SELL,
            order_type=OrderType.ORDER_TYPE_MARKET
        )
        print(r)
        return(True)

    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))



def check_signal(client, figi, acc_id, act :dict, start_depo :float, ticker :str, mx_nums :int, df :DataFrame, lot :int, usd: bool):
    """
    Проверка сигнала по индикаторам для принятия решения о выставлении заявки
    :return
    """
    # константы для расчета индикаторов
    wx = 20
    sma_period = 14

    depo = 0.0
    val = 'rub'
    if usd:
        usdrur = get_usdrur(client)
        if usdrur > 0.0:
            val = 'usd'
            depo = round(start_depo/usdrur, 2)
    else:
        depo = start_depo

    price = df['close'].iloc[-1]
    max_nums = mx_nums # math.ceil(depo/(price*lot))
    qbuy = qsell = int(max_nums*0.1)
    if (qbuy == 0):
        qbuy = qsell = 1
    prc_buy, prc_sell, prc_min, kx = 0.5, 0.5, 0.5, 1.2
    nums = 0
    avg_price = 0.0
    comm = 0.0005

    c_min, c_max = 0, 0
    for i in range(df.size-7, df.size-1, 1):
        dfc = df.copy(deep=True)
        dfc = dfc.iloc[0:i]

        last_price = dfc['close'].iloc[-1]
        # print(last_price)

        if (round(last_price, 6) <= round(dfc['close'].iloc[-2], 6)):
            c_min += 1
            c_max = 0
        if (round(last_price, 6) > round(dfc['close'].iloc[-2], 6)):
            c_min = 0
            c_max += 1

    # Инициализируем индикатор  'Bollinger Bands'
    indicator_bb = BollingerBands(close=df["close"], window=wx, window_dev=2)

    # Записываем значения индикатора 'Bollinger Bands' для оценки
    # df['bb_bbm'] = indicator_bb.bollinger_mavg() # пока среднее не используется
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Получаем значение индикатора RSI
    df['rsi'] = RSIIndicator(close=df["close"], window=sma_period, fillna=False).rsi()

    rsi = round(df['rsi'].iloc[-1], 2)
    bbl = df['bb_bbl'].iloc[-1]
    bbh = df['bb_bbh'].iloc[-1]


    last_price = df['close'].iloc[-1]
    if len(act) == 0:
        avg_price = 0.0
        nums = 0
    else:
        avg_price = act['average_buy_price']
        if usd:
            usdrur = get_usdrur(client)
            avg_price = round(act['average_buy_price']/usdrur, 2)
        nums = int(act['quantity'])
        if nums == 0:
            avg_price = 0.0

    if (max_nums > 0) and (nums > 0):
        prc_buy, prc_sell = (nums*1.0/max_nums + 0.5)*kx, (2.0 - nums*1.0/max_nums)*kx
    if (nums == 0):
        prc_buy, prc_sell = prc_min, prc_min

    if prc_buy < prc_min:
        prc_buy = prc_min
    if prc_sell < prc_min:
        prc_sell = prc_min

    print('max_nums: ', max_nums, ' nums: ', nums, ' avg_price: ', avg_price)
    print('rsi: ', rsi, 'bbl: ', round(bbl, 3), 'bbh:', round(bbh, 3))
    print('c_min: ', c_min, 'c_max: ', c_max)
    print('act: ', act)
    #sys.exit()

    # Если rsi < 30 и значение цены закрытия свечи меньше bbl, то проверяем avg_price: buy
    if (last_price <= bbl) and (rsi <= 30) and (c_min > 5):
        quant = math.ceil(qbuy*(nums/max_nums + 1.2))
        if int(nums+quant) > max_nums:
            quant = max_nums - nums
        pr_need_buy = False
        book = client.market_data.get_order_book(figi=figi, depth=50)
        asks = [cast_money(client, p.price) for p in book.asks] # продавцы
        last_price = asks[0]
        stq = book.asks[0].quantity
        if (stq > quant) and (quant > 0) and (last_price > 0.0):
            if (avg_price == 0):
                pr_need_buy = True
            elif (100.0*(avg_price/last_price - 1.0) >= prc_buy):
                pr_need_buy = True

        if pr_need_buy:
            if make_buy_order(client, figi, quant, acc_id):
                nums += quant
                avg_price = (quant*last_price + (nums-quant)*avg_price)/nums
                depo = depo - (quant*lot*last_price + quant*lot*last_price*comm)

    # Иначе, если rsi > 70 и значение цены закрытия свечи больше bbh, то проверяем avg_price: sell
    elif (last_price >= bbh) and (rsi >= 70) and (c_max > 5):
        quant = math.ceil(qsell*(nums/max_nums + 1.2))
        if (quant > nums):
            quant = nums
        pr_need_sell = False
        book = client.market_data.get_order_book(figi=figi, depth=50)
        bids = [cast_money(client, p.price) for p in book.bids] # покупатели
        last_price = bids[0]
        stq = book.bids[0].quantity

        if (stq > quant) and (quant > 0) and (last_price > 0.0) and (avg_price > 0):
            if (100.0*(last_price/avg_price - 1.0) >= prc_sell):
                pr_need_sell = True

        if pr_need_sell:
            if make_sell_order(client, figi, quant, acc_id):
                nums -= quant
                depo = depo + quant*lot*last_price - quant*lot*last_price*comm



def main():
    """
    Точка входа в программу
    """

    try:
        with Client(TOKEN) as client:

            # Получаем список аккаунтов доступных для токена (хотя бы на READ)
            accounts = get_accounts(client)
            if (len(accounts) == 0):
                # logger.info("Not available acc for trading")                
                return

            port = []
            acc_id = ''
            for account_id in accounts:
                port = get_portfolio_list(client, account_id)
                acc_id = account_id
                if len(port) == 0:
                    continue

            # Обработка активов из TRADELIST
            for tx in get_list_shares(client):
                if (tx.ticker in TLIST) and (TLIST[tx.ticker]['type'] == 'share'):
                    last_day = TDAY
                    lot = TLIST[tx.ticker]['lot']
                    ticker = tx.ticker
                    figi = tx.figi
                    interval = CandleInterval.CANDLE_INTERVAL_HOUR
                    mx_nums = TLIST[tx.ticker]['mx_nums'] # вводим ограничение по кол-ву
                    usd = TLIST[tx.ticker]['usd'] # валюта актива

                    hist_candles = get_candles(client, figi, last_day, interval)

                    df = create_df(client, hist_candles)
                    df = dropna(df)

                    # включаем, когда нужен ли back-test инструмента
                    if RUN_BACKTEST:
                        start_depo = 10**5 # виртуальный начальный депозит в RUB для бэктеста
                        backtest(client, start_depo, ticker, df, lot, usd)

                    # запрос состояния инструмента
                    status = get_trading_status(client, figi)
                    if status:
                        print('\nTrading enable for %s' %(ticker))
                        act = []
                        for p in port:
                            if p['figi'] == figi:
                                act = p
                                break
                        limit_depo = 0.0 # 0.0 is unlimited
                        check_signal(client, figi, acc_id, act, limit_depo, ticker, mx_nums, df, lot, usd)
                    # пауза 2s  после обработка каждого тикера
                    time.sleep(2)


    except RequestError as e:
        print(str(e))


TDAY = 30
RUN_BACKTEST = False
TLIST = {
        # --------------------------------------------------------------------------------------------------
        # RUB
        # --------------------------------------------------------------------------------------------------
        'SBER'  : {'type': 'share', 'mx_nums': 4,   'lot': 10,   'usd': False,   }, # Сбер, ао
        'LKOH'  : {'type': 'share', 'mx_nums': 1,   'lot': 1,    'usd': False,   }, # Лукойл, ао
        'GAZP'  : {'type': 'share', 'mx_nums': 4,   'lot': 10,   'usd': False,   }, # Газпром, ао

        # --------------------------------------------------------------------------------------------------
        # USD
        # --------------------------------------------------------------------------------------------------
        'AAPL'  : {'type': 'share', 'mx_nums': 1,   'lot': 1,    'usd': True,    }, # Apple Inc.
        'INTC'  : {'type': 'share', 'mx_nums': 1,   'lot': 1,    'usd': True,    }, # Intel Corporation
        'NFLX'  : {'type': 'share', 'mx_nums': 1,   'lot': 1,    'usd': True,    }, # Netflix

        'ZY'    : {'type': 'share', 'mx_nums': 10,  'lot': 1,    'usd': True,    }, # Zymergen Inc.
        'VEON'  : {'type': 'share', 'mx_nums': 10,  'lot': 1,    'usd': True,    }, # VEON
        'INO'   : {'type': 'share', 'mx_nums': 12,  'lot': 1,    'usd': True,    }, # Inovio Pharmaceuticals

        # ...
}

#logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    while(True):
        main()

