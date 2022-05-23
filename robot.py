#!/usr/local/bin/python3.10 
import os
import os.path
import time
import sys
import logging
import math
import time
from datetime import datetime, timedelta
from pandas import DataFrame
from pathlib import Path

from tinkoff.invest import (
    Client,
    RequestError,
    PortfolioResponse,
    PositionsResponse,
    PortfolioPosition,
    AccessLevel,
    OrderDirection,
    OrderType,
    Quotation,
    InstrumentStatus,
    CandleInterval,
    CandleInstrument,
    MarketDataRequest,
    SubscribeCandlesRequest,
    SubscriptionAction,
    SubscriptionInterval,
    InfoInstrument,
)
from tinkoff.invest.services import Services
from tinkoff.invest.utils import now
from tinkoff.invest.caching.cache_settings import MarketDataCacheSettings
from tinkoff.invest.services import MarketDataCache
from tinkoff.invest.services import MarketDataStreamManager

from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

# 
sys.path.append('lib')
import bcommon

"""
Для участия в конкурсе по разработке бота для трейдинга на баз tinkoff.invest (python)
https://tinkoff.github.io/investAPI
https://tinkoff.github.io/investAPI/faq_custom_types/
https://github.com/AzzraelCode/api_v2_tinvest/blob/main/v2_portfolio/__init__.py
https://github.com/AzzraelCode/api_v2_tinvest/blob/main/v2_portfolio/__init__.py#L104
https://technical-analysis-library-in-python.readthedocs.io/en/latest/index.html
--

"""

def get_list_futures(client):
    """
    Получаем список фьючерсов
    :param local_client:
    :return:
    """

    try:
        futs = client.instruments.futures().instruments

    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))

    return futs



def get_hist_candles(client, figi, total_day, interval):
    """
    Получаем значение свечей для отдельного инструмента (используем figi)
    :param client:
    :param figi:
    :param day:
    :param interval:
    :return: 
    """

    try:
        settings = MarketDataCacheSettings(base_cache_dir=Path(".market_data_cache"))
        market_data_cache = MarketDataCache(settings=settings, services=client)

        candles = client.get_all_candles(
            figi=figi,
            from_=now() - timedelta(days=total_day),
            interval=interval,
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


def backtest(client, start_depo :float, df :DataFrame, ticker :str, lot :int, usd: bool ):
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
    prc_buy, prc_sell, prc_stop = 2.0, 6.0, 3.0
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

        
        # Условия выполнения гипотезы (испольуются индикаторы RSI, BB и свечи)
        # Если rsi < 30 и значение цены закрытия свечи меньше bbl, то проверяем avg_price: buy
        if (rsi < 30) and (last_price < bbl): # and (c_min > 5)
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
                num_buy_trades += 1

        # Иначе, если rsi > 70 и значение цены закрытия свечи больше bbh, то проверяем avg_price: sell
        elif (rsi > 70) and (last_price > bbh): # and (c_max > 5)
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
                if nums == 0:
                    avg_price = 0
                num_sell_trades += 1

        # stop_loss не используется
        # if (nums == max_nums):
        #    if 100.0*(avg_price/last_price - 1.0) > prc_stop:
        #        # print('stop_loss: ', last_price)
        #        quant = nums
        #        nums = 0
        #        depo = depo + quant*lot*last_price - quant*lot*last_price*comm
        #        avg_price = 0



    now_price = df['close'].iloc[-1]
    print(ticker + ' result: ', str(depo+nums*lot*now_price), val)
    print('num_trades: %s\n' %(str(num_sell_trades + num_buy_trades)))


def make_buy_order(client, fg, quant, acc_id):
    """
    Выставление buy-ордера по рыночной цене
    :return
    """

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
        bcommon.write_log(LOG, r)
        return(True)

    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))


def make_sell_order(client, fg, quant, acc_id):
    """
    Выставление sell-ордера по рыночной цене
    :return
    """

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
        bcommon.write_log(LOG, r)
        return(True)

    except RequestError as err:
        tracking_id = err.metadata.tracking_id if err.metadata else ""
        logger.error("Error tracking_id=%s code=%s", tracking_id, str(err.code))



def check_signal(client, ticker, figi, acc_id, act :dict, start_depo :float, mx_nums :int, df :DataFrame, 
        prc_buy, prc_sell, prc_min, kx, lot :int, usd: bool, rsi_buy_limit, rsi_sell_limit, sma_period, wx):
    """
    Проверка сигнала по индикаторам для принятия решения о выставлении заявки
    :return
    """

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
    nums = 0
    avg_price = 0.0
    comm = 0.0005

    c_min, c_max = 0, 0
    for i in range(df.size-7, df.size-1, 1):
        dfc = df.copy(deep=True)
        dfc = dfc.iloc[0:i]

        # last_price = dfc['close'].iloc[-1]
        # print(last_price)
        prices = client.market_data.get_last_prices(figi=[figi])
        last_price = cast_money(client, prices.last_prices[0].price)


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

    bcommon.write_log(LOG, '\n[{0:s}]\n'.format(ticker))
    bcommon.write_log(LOG, 'max_nums: {0} nums: {1} avg_price: {2}\n'.format(max_nums, nums, avg_price))
    bcommon.write_log(LOG, 'prc_buy: {0} prc_sell: {1} prc_min: {2} kx: {3}\n'.format(prc_buy, prc_sell, prc_min, kx))
    bcommon.write_log(LOG, 'rsi: {0} bbl: {1} bbh: {2} \n'.format(rsi, round(bbl, 2), round(bbh, 2)))
    bcommon.write_log(LOG, 'act: {0}\n'.format(act))

    # Если rsi < rsi_buy_limit и значение цены закрытия свечи меньше bbl, то проверяем avg_price: buy
    if (rsi < rsi_buy_limit): #(last_price < bbl) and (rsi < 30) and (c_min > 5):
        bcommon.write_log(LOG, ' -> [rsi_buy] rsi: {0}\n'.format(rsi))
        quant = 1 # math.ceil(qbuy*(nums/max_nums + 1.2))
        if int(nums+quant) > max_nums:
            quant = max_nums - nums

        pr_need_buy = False
        book = client.market_data.get_order_book(figi=figi, depth=50)
        asks = [cast_money(client, p.price) for p in book.asks] # продавцы
        if (len(asks) > 0):
            last_price = asks[0]
            stq = book.asks[0].quantity
        else:
            stq = 0

        if (stq >= quant) and (quant > 0) and (last_price > 0.0):
            if (avg_price == 0):
                pr_need_buy = True
            elif (100.0*(avg_price/last_price - 1.0) >= prc_buy):
                pr_need_buy = True

        if pr_need_buy:
            bcommon.write_log(LOG, '[buy] num: {0} price: {1}\n'.format(quant, last_price))
            if make_buy_order(client, figi, quant, acc_id):
                nums += quant
                prc_buy *= kx
                prc_sell /= kx
                if prc_sell < prc_min:
                    prc_sell = prc_min
                write_state(ticker, round(prc_buy, 2), round(prc_sell, 2), prc_min, kx)

    # Иначе, если rsi > rsi_sell_limit и значение цены закрытия свечи больше bbh, то проверяем avg_price: sell
    elif (rsi > rsi_sell_limit): #(last_price > bbh) and (rsi > 70) and (c_max > 5):
        bcommon.write_log(LOG, ' -> [rsi_sell] rsi: {0}\n'.format(rsi))
        quant = 1 # math.ceil(qsell*(nums/max_nums + 1.2))
        if (quant > nums):
            quant = nums

        book = client.market_data.get_order_book(figi=figi, depth=50)
        bids = [cast_money(client, p.price) for p in book.bids] # покупатели
        if len(bids) > 0:
            last_price = bids[0]
            stq = book.bids[0].quantity
        else:
            stq = 0

        pr_need_sell = False
        if (stq >= quant) and (quant > 0) and (last_price > 0.0) and (avg_price > 0):
            if (100.0*(last_price/avg_price - 1.0) >= prc_sell):
                pr_need_sell = True

        if pr_need_sell:
            bcommon.write_log(LOG, '[sell] num: {0} price: {1}\n'.format(quant, last_price))
            if make_sell_order(client, figi, quant, acc_id):
                nums -= quant
                prc_sell *= kx
                prc_buy /= kx
                if prc_buy < prc_min:
                    prc_buy = prc_min
                if (nums == 0):
                    prc_buy = prc_min
                    prc_sell = prc_min
                write_state(ticker, round(prc_buy, 2), round(prc_sell, 2), prc_min, kx)
                

def get_request(client, my_figi, my_interval):
    """
    Подписка на свечи по иснтрументу
    :return marketdata
    """

    market_data_stream: MarketDataStreamManager = client.create_market_data_stream()
    market_data_stream.candles.subscribe(
        [
            CandleInstrument(
                figi=my_figi,
                interval=my_interval,
            )
        ]
    )

    marketdata = []
    for marketdata in market_data_stream:
        # print(marketdata)
        market_data_stream.info.subscribe([InfoInstrument(figi=my_figi)])
        if marketdata.subscribe_info_response:
            market_data_stream.stop()

    return marketdata

def write_state(ticker, prc_buy, prc_sell, prc_min, kx):
    """
    Запись state-файла с процентами отклонения от средней
    :return
    """

    file_path = ticker + '_STATE.txt'
    dt_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    new_str = dt_str + ' ' + ticker + ' ' + str(prc_buy) + ' ' + str(prc_sell) + ' ' + str(prc_min) + ' ' + str(kx)
    file1 = open(file_path,"a+")
    file1.write(new_str+'\n')
    file1.close()


def read_state(ticker, prc_buy, prc_sell, prc_min, kx):
    """
    Чтение state-файла
    :return
    """

    file_path = ticker + '_STATE.txt'
    if os.path.exists(file_path):
        file1 = open(file_path,"r")
        cont = file1.readlines()
        file1.close()
        lst_str = ''
        for c in cont:
            if not(c.rstrip() == ''):
                lst_str = c.rstrip()
        if not(lst_str == ''):
            a = lst_str.split(' ')
            prc_buy = float(a[2])
            prc_sell = float(a[3])
            prc_min = float(a[4])
            kx = float(a[5])

    return prc_buy, prc_sell, prc_min, kx


def init_ticker(ticker, figi, fut):
    """
    Инициализация граничных значений процентов отклонений для тикера
    :return
    """

    prc_buy = fut[ticker]['prc_buy']
    prc_sell = fut[ticker]['prc_sell']
    prc_min = fut[ticker]['prc_min']
    kx = fut[ticker]['kx']

    # print(prc_buy, prc_sell, prc_min, kx)

    file_path = ticker + '_STATE.txt'
    if not(os.path.exists(file_path)):
        write_state(ticker, prc_buy, prc_sell, prc_min, kx)
    else:
        prc_buy, prc_sell, prc_min, kx = read_state(ticker, prc_buy, prc_sell, prc_min, kx)

    return prc_buy, prc_sell, prc_min, kx


def main():
    """
    Точка входа в программу
    """

    try:
        with Client(TOKEN) as client:

            # Получаем список аккаунтов доступных для токена (хотя бы на READ)
            accounts = get_accounts(client)
            if (len(accounts) == 0):
                bcommon.write_log(LOG, 'Нет доступных аккаунтов для трейдинга этим токеном: {0:s}\n'.format(TOKEN)) 
                bcommon.write_log(LOG, 'Работа завершена штатно.\n')
                sys.exit(0)

            # Получаем в список состав портфеля port и его acc_id
            port = []
            acc_id = ''
            for account_id in accounts:
                portfolio = get_portfolio_list(client, account_id)
                acc_id = account_id
                for p in portfolio:
                    # Оставляем только фьючерсы из портфеля
                    if p['instrument_type'] == INSTRUMENT_TYPE:
                        port.append(p)

            # Поиск контрактаиз FUT_LIST
            for cont in get_list_futures(client):
                # print(cont)
                if (cont.ticker in FUT_LIST):
                    # Текущий контракт cont
                    last_day = 3 # количество дней при запросе свечей
                    ticker = cont.ticker
                    figi = cont.figi
                    m5 = CandleInterval.CANDLE_INTERVAL_5_MIN
                    m15 = CandleInterval.CANDLE_INTERVAL_15_MIN
                    max_nums = FUT_LIST[ticker]['max_nums'] # вводим ограничение по кол-ву контрактов
                    rsi_buy_limit = FUT_LIST[ticker]['rsi_buy_limit']
                    rsi_sell_limit = FUT_LIST[ticker]['rsi_sell_limit']
                    sma_period = FUT_LIST[ticker]['sma_period']
                    wx = FUT_LIST[ticker]['wx']
                    lot = 1
                    usd = False
                    prc_buy, prc_sell, prc_min, kx = init_ticker(ticker, figi, FUT_LIST)

                    candles = get_hist_candles(client, figi, last_day, m15)

                    df = create_df(client, candles)
                    df = dropna(df)

                    # запрос состояния инструмента
                    status = get_trading_status(client, figi)
                    if (status):
                        act = []
                        for p in port:
                            if p['figi'] == figi:
                                act = p
                                break
                        limit_depo = 0.0 # 0.0 is unlimited
                        check_signal(client, ticker, figi, acc_id, act, limit_depo, max_nums, df, prc_buy, prc_sell, prc_min, kx, 
                                lot, usd, rsi_buy_limit, rsi_sell_limit, sma_period, wx)
                    else:
                        bcommon.write_log(LOG, 'Trading disable for {0}'.format(ticker))
                    # Пауза 2s  после обработка каждого тикера
                    time.sleep(2)


    except RequestError as e:
        print(str(e))


TOKEN = os.environ["INVEST_TOKEN"]

LOG = 'trades.log'
NUM_TEST_DAYS = 30
INSTRUMENT_TYPE = 'futures'
RUN_BACKTEST = False
FUT_LIST = {
    'RMM2'  : { 
        'prc_buy': 1.2, 'prc_sell': 1.2, 'prc_min': 1.2, 'kx': 1.2, # проценты на покупку и продажу
        'max_nums': 4,                                              # максимальное количество контрактов
        'rsi_buy_limit': 30, 'rsi_sell_limit': 70,                  # нижнее и верхнее ограничение для RSI
        'sma_period': 14, 'wx': 20,                                 # константы для расчета значений индикаторов RSI и BB
    },
    'SiM2'  : {
        'prc_buy': 1.2, 'prc_sell': 1.2, 'prc_min': 1.2, 'kx': 1.2,
        'max_nums': 4,
        'rsi_buy_limit': 30, 'rsi_sell_limit': 70,
        'sma_period': 14, 'wx': 20,
    },
    # ...
}

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    while(True):
        main()

