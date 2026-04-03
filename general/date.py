############################################################################## 
# IMPORT DEPENDECIES
##############################################################################

import pandas as pd
import datetime
import sys
import traceback

__all__ = ['BDAY','BDAY_USD','BDAY_EUREX','BDAY_FX','BDAY_JP','BDAY_LSE','today','yesterday','yesterdays','tic','toc','is_trading_day']


##############################################################################
# DEFINE TRADING CALENDAR
##############################################################################

from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import AbstractHolidayCalendar
from pandas_market_calendars import MarketCalendar
from pytz import timezone
from pandas import DateOffset, Timestamp
from functools import partial
from datetime import time, timedelta


from pandas.tseries.holiday import Holiday,USMartinLutherKingJr,USPresidentsDay,GoodFriday,USMemorialDay,USLaborDay,USThanksgivingDay,EasterMonday, USColumbusDay
from pandas.tseries.holiday import nearest_workday,weekend_to_monday,MO,sunday_to_monday, next_monday_or_tuesday, previous_friday
from pandas_market_calendars.holidays.us import USNewYearsDay, Christmas, ChristmasEveInOrAfter1993, USNationalDaysofMourning, USMartinLutherKingJrAfter1998, USIndependenceDay
from pandas_market_calendars.holidays.jpx_equinox import autumnal_equinox, vernal_equinox
from pandas_market_calendars.holidays.cn import sf_mapping, bsd_mapping, maf_mapping

class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """ NYD, MLK, Presidents, Good Friday, Memorial, Juneteenth, July 4, Labor, Thanksgiving, Xmas
    http://markets.on.nytimes.com/research/markets/holidays/holidays.asp?display=market&exchange=NYQ
    Unlike Federal Calendar, no Columbus or Veterans Day
    """
    
    rules = [
        Holiday('New Years Day', month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, observance=nearest_workday, start_date=Timestamp(2022,6,19)),
        Holiday('July 4th', month=7, day=4, observance=nearest_workday),
        USLaborDay,
#        USColumbusDay,
#        Holiday('Veterans Day', month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]
    
BDAY = CustomBusinessDay(calendar=NYSEHolidayCalendar())

class USDSettlementCalendar(AbstractHolidayCalendar):
    """ NYD, MLK, Presidents, Memorial, July 4, Labor, Columbus Day, Veteran's Day, Thanksgiving, Xmas
    """
    
    rules = [
        Holiday('New Years Day', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday('July 4th', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USColumbusDay,
        Holiday('Veterans Day', month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

BDAY_USD = CustomBusinessDay(calendar=USDSettlementCalendar())

class FXTradingCalendar(MarketCalendar):
    """
    Open Time: 5:01 PM, America/New_York
    Close Time: 5:00 PM, America/New_York
    
    Regularly-Observed Trading Holidays:
    - New Year's Day (January 1)
    - Christmas Day (December 25)
    
    Regularly-Observed Early Closes:
    - Last Business Day before Christmas Day
    - Last Business Day of the Year
    
    Regularly-Observed Late Opens:
    - First Day after Christmas Day
    - First Business Day of the Year
    """
    aliases = ['FOREX']
    regular_early_close = datetime.time(14, 0)
    regular_late_open = datetime.time(2, 1)
    
    @property
    def name(self):
        return "FOREX"

    @property
    def tz(self):
        return timezone("America/New_York")

    @property
    def open_time_default(self):
        return datetime.time(17, 1, tzinfo=self.tz)

    @property
    def close_time_default(self):
        return datetime.time(17, 0, tzinfo=self.tz)
    
    @property
    def open_offset(self):
        return -1

    @property
    def regular_holidays(self):
        return AbstractHolidayCalendar(rules=[
            Holiday("New Year's Day", month=1,day=1),
            Holiday("Christmas",month=12,day=25),
        ])
        
    @property
    def special_closes(self):
        return [(
            self.regular_early_close, 
            AbstractHolidayCalendar(rules=[
                Holiday("Christmas Eve", month=12,day=24),
                Holiday("Last Business Day", month=12,day=31),
            ])
        )]

    @property
    def special_opens(self):
        return [(
            self.regular_late_open, 
            AbstractHolidayCalendar(rules=[
                Holiday("First Day After New Year", month=1,day=2),
                Holiday("First Day After Christmas", month=12,day=26),
            ])
        )]

class CMEDExchangeCalendar(MarketCalendar):
    """
    Exchange calendar for CME with only Day Session
    This calendar is used before 07/01/2003

    Open Time: 0:00 AM, America/Chicago
    Close Time: 3:15 PM, America/Chicago

    Regularly-Observed Holidays:
    - New Years Day
    - Good Friday
    - Christmas
    - most other holidays were off for the day
    """
    aliases = ['CMED','CBOTD','COMEXD','NYMEXD']

    @property
    def name(self):
        return "CMED"

    @property
    def tz(self):
        return timezone('America/Chicago')

    @property
    def open_time_default(self):
        return time(0, 1, tzinfo=self.tz)

    @property
    def close_time_default(self):
        return time(15, 15, tzinfo=self.tz)

    @property
    def regular_holidays(self):
        # The CME has different holiday rules depending on the type of
        # instrument. For example, http://www.cmegroup.com/tools-information/holiday-calendar/files/2016-4th-of-july-holiday-schedule.pdf # noqa
        # shows that Equity, Interest Rate, FX, Energy, Metals & DME Products
        # close at 1200 CT on July 4, 2016, while Grain, Oilseed & MGEX
        # Products and Livestock, Dairy & Lumber products are completely
        # closed.

        # For now, we will treat the CME as having a single calendar, and just
        # go with the most conservative hours - and treat July 4 as an early
        # close at noon.
        return AbstractHolidayCalendar(rules=[
            USNewYearsDay,
            USMartinLutherKingJrAfter1998,
            USPresidentsDay,
            GoodFriday,
            USMemorialDay,
            USIndependenceDay,
            USLaborDay,
            USThanksgivingDay,
            Christmas,
        ])

    @property
    def adhoc_holidays(self):
        return USNationalDaysofMourning

    @property
    def special_closes(self):
        return [(
            time(12),
            AbstractHolidayCalendar(rules=[
                #USBlackFridayInOrAfter1993,
                #ChristmasEveBefore1993,
                ChristmasEveInOrAfter1993,
            ])
        )]

class ASXExchangeCalendar(MarketCalendar):
    """
    Open Time: 10:00 AM, Australia/Sydney
    Close Time: 4:10 PM, Australia/Sydney

    Regularly-Observed Holidays:
    - New Year's Day (observed on Monday when Jan 1 is a Saturday or Sunday)
    - Australia Day (observed on Monday when Jan 26 is a Saturday or Sunday)
    - Good Friday (two days before Easter Sunday)
    - Easter Monday (the Monday after Easter Sunday)
    - ANZAC Day (April 25)
    - Queen's Birthday (second Monday in June)
    - Christmas Day (December 25, Saturday/Sunday to Monday)
    - Boxing Day (December 26, Saturday to Monday, Sunday to Tuesday)

    Regularly-Observed Early Closes:
    - Last Business Day before Christmas Day
    - Last Business Day of the Year
    """
    aliases = ['ASX']
    regular_early_close = datetime.time(14, 10)

    @property
    def name(self):
        return "ASX"

    @property
    def tz(self):
        return timezone("Australia/Sydney")

    @property
    def open_time_default(self):
        return datetime.time(10, 1, tzinfo=self.tz)

    @property
    def close_time_default(self):
        return datetime.time(16, 10, tzinfo=self.tz)

    @property
    def regular_holidays(self):
        return AbstractHolidayCalendar(rules=[
            Holiday("New Year's Day",month=1,day=1,observance=weekend_to_monday),
            Holiday("Australia Day",month=1,day=26,observance=weekend_to_monday,),
            Holiday("ANZAC Day",month=4,day=25,),
            Holiday("Queen's Birthday",month=6,day=1,offset=pd.DateOffset(weekday=MO(2)),),
            Holiday("Christmas",month=12,day=25,observance=weekend_to_monday,),
            Holiday("Boxing Day",month=12,day=26,),
            Holiday("Weekend Boxing Day",month=12,day=28,days_of_week=(0,1),),
            GoodFriday,
            EasterMonday,
        ])

    @property
    def special_closes(self):
        return [(
            self.regular_early_close, 
            AbstractHolidayCalendar(rules=[
                Holiday("Christmas Eve", month=12,day=24, days_of_week=(0,1,2,3,4)),
                Holiday("Last Business Day", month=12,day=31, days_of_week=(0,1,2,3,4)),
            ])
        )]

class ASXOExchangeCalendar(MarketCalendar):
    """
    ASX Exchange with Overnight Session
    Open Time: 8:30AM
    Close Time: 7:00AM AEST / 7:30AM AEDT

    Regularly-Observed Holidays:
    - New Year's Day (observed on Monday when Jan 1 is a Saturday or Sunday)
    - Australia Day (observed on Monday when Jan 26 is a Saturday or Sunday)
    - Good Friday (two days before Easter Sunday)
    - Easter Monday (the Monday after Easter Sunday)
    - ANZAC Day (April 25)
    - Queen's Birthday (second Monday in June)
    - Christmas Day (December 25, Saturday/Sunday to Monday)
    - Boxing Day (December 26, Saturday to Monday, Sunday to Tuesday)

    Regularly-Observed Early Closes:
    - Last Business Day before Christmas Day
    - Last Business Day of the Year
    """
    aliases = ['ASXO']
    regular_early_close = datetime.time(12, 30)

    @property
    def name(self):
        return "ASXO"

    @property
    def tz(self):
        return timezone("Australia/Sydney")

    @property
    def open_time_default(self):
        return datetime.time(8, 31, tzinfo=self.tz)

    @property
    def close_time_default(self):
        return datetime.time(7, 0, tzinfo=self.tz)

    @property
    def close_offset(self):
        return 1

    @property
    def regular_holidays(self):
        return AbstractHolidayCalendar(rules=[
            Holiday("New Year's Day", month=1,day=1,observance=weekend_to_monday),
            Holiday("Australia Day",month=1,day=26,observance=weekend_to_monday,),
            Holiday("ANZAC Day",month=4,day=25,),
            Holiday("Queen's Birthday",month=6,day=1,offset=pd.DateOffset(weekday=MO(2)),),
            Holiday("Christmas",month=12,day=25,observance=weekend_to_monday,),
            Holiday("Boxing Day",month=12,day=26,),
            Holiday("Weekend Boxing Day",month=12,day=28,days_of_week=(0,1),),
            GoodFriday,
            EasterMonday,
        ])

    @property
    def special_closes(self):
        return [(
            self.regular_early_close,
            AbstractHolidayCalendar(rules=[
                Holiday("Christmas Eve", month=12,day=24, days_of_week=(0,1,2,3,4)),
                Holiday("Last Business Day", month=12,day=31, days_of_week=(0,1,2,3,4)),
            ])
        )]

class ASXONZExchangeCalendar(MarketCalendar):
    """
    ASX Exchange with Overnight Session (based on NZ time)
    Open Time: 8:30AM
    Close Time: 7:00AM

    Regularly-Observed Holidays:
    - New Year's Day (observed on Monday when Jan 1 is a Saturday or Sunday)
    - Day after New Year's Day (observed on Monday or Tuesday when Jan 1 is on Weekend)
    - Waitangi Day (Feb 6, observed on Monday when it is on Weekend since 2014)
    - Good Friday (two days before Easter Sunday)
    - Easter Monday (the Monday after Easter Sunday)
    - ANZAC Day (April 25, observed on Monday when it is on Weekend since 2014)
    - Queen's Birthday (first Monday in June)
    - Labour Day (fourth Monday in October)
    - Christmas Day (December 25, Saturday/Sunday to Monday)
    - Boxing Day (December 26, Saturday to Monday, Sunday to Tuesday)

    Regularly-Observed Early Closes:
    - Last Business Day before Christmas Day
    - Last Business Day of the Year
    """
    aliases = ['ASXONZ']
    regular_early_close = datetime.time(12, 0)

    @property
    def name(self):
        return "ASXONZ"

    @property
    def tz(self):
        return timezone("Pacific/Auckland")

    @property
    def open_time_default(self):
        return datetime.time(8, 31, tzinfo=self.tz)

    @property
    def close_time_default(self):
        return datetime.time(7, 0, tzinfo=self.tz)

    @property
    def close_offset(self):
        return 1

    @property
    def regular_holidays(self):
        return AbstractHolidayCalendar(rules=[
            Holiday("New Year's Day",month=1,day=1,observance=weekend_to_monday),
            Holiday("Day after New Year's Day",month=1,day=2,observance=next_monday_or_tuesday),
            Holiday("Waitangi Day",month=2,day=6,end_date=Timestamp(2013,12,31)),
            Holiday("Waitangi Day",month=2,day=6,observance=weekend_to_monday,start_date=Timestamp(2014,1,1)),
            Holiday("ANZAC Day", month=4, day=25,end_date=Timestamp(2013,12,31)),
            Holiday("ANZAC Day",month=4,day=25,observance=weekend_to_monday,start_date=Timestamp(2014,1,1)),
            Holiday("Queen's Birthday",month=6,day=1,offset=pd.DateOffset(weekday=MO(1)),),
            Holiday("Labour Day", month=10, day=1, offset=pd.DateOffset(weekday=MO(4)), ),
            Holiday("Christmas",month=12,day=25,observance=weekend_to_monday,),
            Holiday("Boxing Day",month=12,day=26,),
            Holiday("Weekend Boxing Day",month=12,day=28,days_of_week=(0,1),),
            GoodFriday,
            EasterMonday,
        ])

    @property
    def special_closes(self):
        return [(
            self.regular_early_close,
            AbstractHolidayCalendar(rules=[
                Holiday("Christmas Eve", month=12,day=24, days_of_week=(0,1,2,3,4)),
                Holiday("Last Business Day", month=12,day=31, days_of_week=(0,1,2,3,4)),
            ])
        )]
    
class KSEExchangeCalendar(MarketCalendar):
    """
    Open Time: 9:00 AM, Asia/Seoul
    Close Time: 3:45 PM, Asia/Seoul

    Regularly-Observed Holidays:
    - New Year's Day, January 1st
    - Lunar New Year's Day, 3-Day Holiday
    - Independence Movement Day, March 1st
    - Buddha's Birthday
    - Labor Day, May 1st
    - Children's Day, May 5th
    - Memorial Day, June 6th
    - Constitution Day, July 17th, removed since 2008
    - Independence Day, August 15th
    - Chuseok, 3-Day Holiday
    - National Foundation Day, October 3rd
    - Hangeul Day, October 9th
    - Christmas Day, December 25th
    - End of Year Holiday, December 31st

    Adhoc Holidays:
    - Mostly Election Days
    """

    aliases = ['KSE']

    @property
    def name(self):
        return "KSE"

    @property
    def tz(self):
        return timezone("Asia/Seoul")

    @property
    def open_time_default(self):
        return datetime.time(9, 1, tzinfo=self.tz)

    @property
    def close_time_default(self):
        return datetime.time(15, 45, tzinfo=self.tz)

    @property
    def regular_holidays(self):
        return AbstractHolidayCalendar(rules=[
            Holiday("New Year's Day",month=1,day=1),
            Holiday("Day before Lunar New Year's Day",month=1,day=19,observance=partial(lunartosolar,mapping=sf_mapping,func=process_korea_lunar_day_before,mode='LNY'),start_date=Timestamp('1991-01-01')),
            Holiday("Lunar New Year's Day",month=1,day=20,observance=partial(lunartosolar,mapping=sf_mapping,func=process_korea_lunar_day,mode='LNY'),start_date=Timestamp('1991-01-01')),
            Holiday("Day after Lunar New Year's Day",month=1,day=21,observance=partial(lunartosolar,mapping=sf_mapping,func=process_korea_lunar_day_after,mode='LNY'),start_date=Timestamp('1991-01-01')),
            Holiday("Independence Movement Day",month=3,day=1),
            Holiday("Buddha's Birthday",month=4,day=28,observance=partial(lunartosolar,mapping=bsd_mapping),start_date=Timestamp('1991-01-01')),
            Holiday("Labor Day",month=5,day=1),
            Holiday("Children's Day",month=5,day=5,observance=weekend_to_monday,start_date=Timestamp(2015,1,1)),
            Holiday("Children's Day",month=5,day=5,end_date=Timestamp(2014,12,31)),
            Holiday("Memorial Day",month=6,day=6),
            Holiday("Constitution Day",month=7,day=17,end_date=Timestamp(2007,12,31)),
            Holiday("Independence Day",month=8,day=15),
            Holiday("Day before Chuseok",month=9,day=7,observance=partial(lunartosolar,mapping=maf_mapping,func=process_korea_lunar_day_before,mode='MAF'),start_date=Timestamp('1991-01-01')),
            Holiday("Chuseok",month=9,day=8,observance=partial(lunartosolar,mapping=maf_mapping,func=process_korea_lunar_day,mode='MAF'),start_date=Timestamp('1991-01-01')),
            Holiday("Day after Chuseok",month=9,day=9,observance=partial(lunartosolar,mapping=maf_mapping,func=process_korea_lunar_day_after,mode='MAF'),start_date=Timestamp('1991-01-01')),
            Holiday("National Foundation Day",month=10,day=3),
            Holiday("Hangeul Day",month=10,day=9),
            Holiday('Christmas Day',month=12,day=25),
            Holiday("End of Year Holiday",month=12,day=31,observance=previous_friday),
        ])

    @property
    def adhoc_holidays(self):
        return [
            Timestamp('2020-04-15'), # General Election Day
            Timestamp('2020-08-17'), # Temporary Holiday
            Timestamp('2018-06-13'), # Provincial Election Day
            Timestamp('2017-05-09'), # Presidential Election Day
            Timestamp('2017-10-02'), # Temporary Holiday
            Timestamp('2016-04-13'), # General Election Day
            Timestamp('2016-05-06'), # Temporary Holiday
            Timestamp('2015-08-14'), # Temporary Holiday
            Timestamp('2014-06-04'), # Provincial Election Day
            Timestamp('2012-04-11'), # Unknown
            Timestamp('2012-05-28'), # Unknown
            Timestamp('2012-12-19'), # Unknown
            Timestamp('2010-06-02'), # Unknown
        ]
    
class JPOExchangeCalendar(MarketCalendar):
    """
    Exchange calendar for Japan Exchange with Overnight Session

    Open Time: 8:45 AM, Asia/Tokyo
    Close Time: 3:00 AM, Asia/Tokyo
    Trading Break: 15:15-16:30, Asia/Tokyo
    """
    aliases = ['JPO']

    @property
    def name(self):
        return "JPO"

    @property
    def tz(self):
        return timezone('Asia/Tokyo')

    @property
    def open_time_default(self):
        return datetime.time(8, 46, tzinfo=self.tz)

    @property
    def close_time_default(self):
        return datetime.time(3, tzinfo=self.tz)

    @property
    def close_offset(self):
        return 1

    @property
    def regular_holidays(self):
        return AbstractHolidayCalendar(rules=[
            USNewYearsDay,
            Holiday(name="New Year's Day",month=1,day=2,observance=sunday_to_monday,),
            Holiday(name="New Year's Day",month=1,day=3,observance=sunday_to_monday,),
            Holiday(name="Coming of Age Day",month=1,day=1,offset=DateOffset(weekday=MO(2)),),
            Holiday(name="National foundation day",month=2,day=11,observance=sunday_to_monday,),
            Holiday(name="Vernal Equinox",month=3,day=20,observance=vernal_equinox),
            Holiday(name="Showa day",month=4,day=29,observance=sunday_to_monday,),
            Holiday(name="Constitution memorial day",month=5,day=3,observance=sunday_to_monday,),
            Holiday(name="Greenery day",month=5,day=4,observance=sunday_to_monday,),
            Holiday(name="Children's day",month=5,day=5,observance=sunday_to_monday,),
            Holiday(name="Marine day",month=7,day=1,offset=DateOffset(weekday=MO(3)),),
            Holiday(name="Mountain day",month=8,day=11,observance=sunday_to_monday,),
            Holiday(name="Respect for the aged day",month=9,day=1,offset=DateOffset(weekday=MO(3)),),
            Holiday(name="Autumnal equinox",month=9,day=22,observance=autumnal_equinox,),
            Holiday(name="Health and sports day",month=10,day=1,offset=DateOffset(weekday=MO(2)),),
            Holiday(name="Culture day",month=11,day=3,observance=sunday_to_monday,),
            Holiday(name="Labor Thanksgiving Day",month=11,day=23,observance=sunday_to_monday,),
            Holiday(name="Emperor's Birthday",month=12,day=23,observance=sunday_to_monday,),
            Holiday(name="Before New Year's Day",month=12,day=31,observance=sunday_to_monday,),
        ])

    @property
    def adhoc_holidays(self):
        return [
            Timestamp('2021-07-23'), # Olympics 2021 Health/Sports
        ]

    @property
    def special_opens_adhoc(self):
        return [
            (datetime.time(23,46), ['2021-10-11']) # Doesnt work... Olympics 2021 Health/Sports
        ]    
    
BDAY_JP = CustomBusinessDay(calendar=JPOExchangeCalendar())


import pandas_market_calendars as mcal

EUREX      = mcal.get_calendar('EUREX')
FOREX      = mcal.get_calendar('FOREX')
LSE        = mcal.get_calendar('LSE')

BDAY_EUREX = EUREX.holidays()
BDAY_FX    = FOREX.holidays()
BDAY_LSE   = LSE.holidays()


##############################################################################
# DATE FUNCTIONS
##############################################################################

def today():
    """ Return today's date as a datetime.date
    """
    return datetime.date.today()

def yesterday(bday=BDAY):
    """ Return yesterday's date as a datetime.date  
    """
    return (datetime.datetime.now() - bday).date()

def yesterdays(n=0,bday=BDAY):
    """ Return date N-days ago as a datetime.date 
    """
    return (datetime.datetime.now() - n * bday).date()

def tic():
    " Start timer. End with toc()."
    return datetime.datetime.now()

def toc(tic):
    " End timer. Pass in result from tic()."
   
    toc   = datetime.datetime.now()
    delta = (toc - tic).seconds
    
    print("%ss elapsed" %delta)
    
    return delta

def is_trading_day(date=datetime.datetime.now(), market="us" ):
    """ Check if the date is a trading day.
        market: us
        date: YYYYMMDD string or datetime
    """
    if type(date) == str:
        try:
            dt_date = datetime.datetime.strptime(date, '%Y%m%d')
        except:
            print(traceback.format_exc())
            print("Wrong string date format, must be YYYYMMDD.")
            sys.exit(1)
    elif type( date ) == datetime.datetime or type( date ) == datetime.date or type( date ) == pd.tslib.Timestamp:
        dt_date = date
    else:
        print("date must be a YYYYMMDD string or datetime object.")
        sys.exit(1)
    return (dt_date - BDAY + BDAY).date() == dt_date.date()

def lunartosolar(dt, mapping, func=None, delta=None, mode=None):
    new_dt = mapping[dt.year]
    if delta:
        new_dt = new_dt + timedelta(delta)
    if func:
        return func(new_dt, mode)
    else:
        return new_dt
    
def process_korea_lunar_day_before(dt, mode=None):
    lunar_recipe = {'LNY':2015,'MAF':2014}
    assert mode in lunar_recipe, 'mode not defined.'
    if dt.year >= lunar_recipe[mode]:
        dow = dt.weekday()
        if dow == 0:
            return dt
        else:
            return dt + timedelta(-1)
    else:
        return dt + timedelta(-1)

def process_korea_lunar_day(dt, mode=None):
    lunar_recipe = {'LNY': 2015, 'MAF': 2014}
    assert mode in lunar_recipe, 'mode not defined.'
    if dt.year >= lunar_recipe[mode]:
        dow = dt.weekday()
        if dow == 0:
            return dt + timedelta(1)
        else:
            return sunday_to_monday(dt)
    else:
        return dt

def process_korea_lunar_day_after(dt, mode=None):
    lunar_recipe = {'LNY': 2015, 'MAF': 2014}
    assert mode in lunar_recipe, 'mode not defined.'
    if dt.year >= lunar_recipe[mode]:
        dow = dt.weekday()
        if dow in [0, 5, 6]:
            return dt + timedelta(2)
        else:
            return dt + timedelta(1)
    else:
        return dt + timedelta(1)

