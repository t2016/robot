# robot

# Стратегия RSI+Линии Болинджера 

# Пререквизиты
```
Python версии не ниже 3.10
Модули Python:
grpcio              1.46.0
numpy               1.22.3
pandas              1.4.2
python-dateutil     2.8.2
ta                  0.10.1
tinkoff             0.1.1
tinkoff-investments 0.2.0b26
```

# Описание стратегии
RSI (relative strength index) - индекс относительной силы. Индикатор показывается насколько сильно актив перекуплен (т.е. стоит слишком дорого) или перепродан (т.е. стоит слишком дешево) относительно средних значений. Если RSI высоко, то это сигнал к продаже актива, если низко - к покупке.

Классическими являются отсечки 30 и 70 (значение индикатора выше 70 - продавать, ниже 30 - покупать). Однако, у всех активов разные характеристики, поэтому лучшим решением будет определение этого диапазона для каждого актива индивидуально.

Линии Боллинджера (Bollinger bands, BB) — инструмент технического анализа финансовых рынков, отражающий текущие отклонения цены акции, товара или валюты.

Индикатор рассчитывается на основе стандартного отклонения от простой скользящей средней. Обычно отображается поверх графика цены. Параметрами для расчета служит тип стандартного отклонения (обычно двойное) и период скользящей средней (зависит от предпочтений трейдера).

Индикатор помогает оценить, как расположены цены относительно нормального торгового диапазона. Линии Боллинджера создают рамку, в пределах которой цены считаются нормальными. Линии Боллинджера строятся в виде верхней и нижней границы вокруг скользящей средней, но ширина полосы не статична, а пропорциональна среднеквадратическому отклонению от скользящей средней за анализируемый период времени.

Торговым сигналом считается, когда цена выходит из торгового коридора — либо поднимаясь выше верхней линии, либо пробивая нижнюю линию. Если график цены колеблется между линиями — индикатор не даёт торговых сигналов.

Для торговли выбираются иснтументы с контр-трендом, например RI/Si.

Используются значения граничные значения индикаторов RSI и BB для принятия решения о покупке или продаже контракта.
При контроле позиций применяется ограниченный Мартингейл (для контроля используется максимальное число контрактов max_nums, задаваемое в настройках).

При торговле:

> Если цена больще средней цены на 1% (значение в конфиге prc_sell) и значения индикаторов RSI и BB вышли из граничных значений вверх, то выставляется рыночная заявка на продажу для фиксирования прибыли.

> Если цена упала на 1% (значение в конфиге prc_buy) и значения индикаторов RSI и BB вышли из граничных значений вниз, то выставляется рыночная заявка на покупку.


Пример конфигурации в словаре FUT_LIST:
```python
  'RMM2' - идентификатор инструмента
  'prc_buy': 1.2, 'prc_sell': 1.2, 'prc_min': 1.2, 'kx': 1.2 - проценты отклонений на покупку и продажу, минимальный процент, коэффициент роста процентов
  'max_nums': 4 - максимальное количество контрактов
  'rsi_buy_limit': 30, 'rsi_sell_limit': 70 -  нижнее и верхнее ограничение для RSI
  'sma_period': 14, 'wx': 20 -  константы для расчета индикаторов RSI и BB
```
