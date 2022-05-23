import datetime
import os
from time import mktime, strptime


def localstr2epoch(datestr, fmat="%Y-%m-%d %H:%M:%S"):
    return mktime(strptime(datestr, fmat))

def uts_to_date(uts, fmat="%Y-%m-%d %H:%M:%S"):
    timestamp = datetime.datetime.fromtimestamp(uts)
    return timestamp.strftime(fmat)

def write_log(log_name, s):
    """
    Запись лога действий бота
    :return
    """
    if os.path.isfile(log_name):
        if os.path.getsize(log_name)/1024/1024 > 2000: # max size: 2 GB
            try:
                os.remove(log_name)
            except OSError:
                pass
    fh = open(log_name,'a+')
    fh.write(s)
    fh.close()

def transform_dict_to_arr(avgs):
    kh, vh = [], []
    for key in sorted(avgs.iterkeys()):
        vh.append(avgs[key])
        kh.append(key)
    return kh, vh

