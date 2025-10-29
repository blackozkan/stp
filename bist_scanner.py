"""
BIST HİSSE TARAMA - GitHub Actions İçin Optimize Edilmiş
Hızlı + Paralel + Tüm Özellikler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import requests
import io
import time
import os
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ayarlar
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Telegram
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.session = self._create_session()
    
    def _create_session(self):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def send_message(self, text):
        if not self.bot_token or not self.chat_id:
            return False
        try:
            url = f"{self.base_url}/sendMessage"
            data = {'chat_id': self.chat_id, 'text': text, 'parse_mode': 'Markdown'}
            response = self.session.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def send_photo(self, image_buffer, caption=''):
        if not self.bot_token or not self.chat_id:
            return False
        try:
            url = f"{self.base_url}/sendPhoto"
            image_buffer.seek(0)
            files = {'photo': image_buffer}
            data = {'chat_id': self.chat_id, 'caption': caption}
            response = self.session.post(url, files=files, data=data, timeout=30)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram photo error: {e}")
            return False


class AlternativeDataFetcher:
    """Hızlı veri çekme"""
    
    def __init__(self):
        self.session = self._create_session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
    
    def _create_session(self):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def fetch_yahoo_alternative(self, symbol, period_days=90):
        """Yahoo Finance v8 API"""
        try:
            end_time = int(time.time())
            start_time = end_time - (period_days * 24 * 60 * 60)
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': start_time,
                'period2': end_time,
                'interval': '1d',
                'events': 'history'
            }
            
            response = self.session.get(url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if 'chart' not in data or 'result' not in data['chart']:
                return None
            
            result = data['chart']['result'][0]
            
            if 'timestamp' not in result or not result['timestamp']:
                return None
            
            indicators = result['indicators']['quote'][0]
            
            df = pd.DataFrame({
                'Date': pd.to_datetime(result['timestamp'], unit='s'),
                'Open': indicators.get('open', []),
                'High': indicators.get('high', []),
                'Low': indicators.get('low', []),
                'Close': indicators.get('close', []),
                'Volume': indicators.get('volume', [])
            })
            
            df = df.dropna(subset=['Close'])
            df = df.set_index('Date')
            
            if len(df) < 20:
                return None
            
            return df
            
        except:
            return None


class BISTScanner:
    def __init__(self, telegram_notifier=None):
        self.symbols = [
           'JANTs.IS', 'KAPLM.IS', 'KAREL.IS', 'KARSN.IS', 'KARTN.IS', 'KATMR.IS',
    'KAYSE.IS', 'KBORU.IS', 'KCAER.IS', 'KCHOL.IS', 'KENT.IS', 'KERVN.IS',
    'KFEIN.IS', 'KIMMR.IS', 'KLKIM.IS', 'KLMSN.IS', 'KLNMA.IS', 'KLRHO.IS',
    'KLSER.IS', 'KLSYN.IS', 'KLYPv.IS', 'KMPUR.IS', 'KNFRT.IS', 'KOCMT.IS',
    'KONKA.IS', 'KONTR.IS', 'KONYA.IS', 'KOPOL.IS', 'KORDS.IS', 'KOTON.IS',
    'KOZAA.IS', 'KOZAL.IS', 'KRDMA.IS', 'KRONT.IS', 'KRPLS.IS', 'KRSTL.IS',
    'KRTEK.IS', 'KRVGD.IS', 'KSTUR.IS', 'KTLEV.IS', 'KUTPO.IS', 'KUVVA.IS',
    'KZBGY.IS', 'KZGYO.IS', 'LIDER.IS', 'LIDFA.IS', 'LILAK.IS', 'LINK.IS',
    'LKMNH.IS', 'Lmkdc.IS', 'LOGO.IS', 'LRSHO.IS', 'LUKSK.IS', 'LYDHO.IS',
    'LYDYE.IS', 'MAALT.IS', 'MACKO.IS', 'MAGEN.IS', 'MAKIM.IS', 'MAKTK.IS',
    'MANAS.IS', 'MARBL.IS', 'MARKA.IS', 'MARMr.IS', 'MARTI.IS', 'MNDTR.IS',
    'MOBTL.IS', 'MOGAN.IS', 'MOPAS.IS', 'MPARK.IS', 'MRSHL.IS', 'MTRKS.IS',
    'MTRYO.IS', 'MZHLD.IS', 'NATEN.IS', 'NETAS.IS', 'NIBAS.IS', 'NTGAZ.IS',
    'NTHOL.IS', 'NUHCM.IS', 'OBAMS.IS', 'OBASE.IS', 'ODAS.IS', 'ODINE.IS',
    'OFSYM.IS', 'ONCSM.IS', 'ONRYT.IS', 'ORCAY.IS', 'ORGE.IS', 'ORMA.IS',
    'OSMEN.IS', 'OSTIM.IS', 'OTKAR.IS', 'OTTO.IS', 'OYAKC.IS', 'OYAYO.IS',
    'OYLUM.IS', 'OYYAT.IS', 'OZATD.IS', 'OZrdn.IS', 'OZSUB.IS', 'OZGYO.IS',
    'PAGYO.IS', 'PAmEL.IS', 'PAPIL.IS', 'PARSN.IS', 'PASEU.IS', 'PATEK.IS',
    'PCILT.IS', 'PEKGY.IS', 'PENGD.IS', 'PENTA.IS', 'PETKM.IS', 'PETUN.IS',
    'PGSUS.IS', 'PINSU.IS', 'PKART.IS', 'PKENT.IS', 'PLTUR.IS',
    'PNLSN.IS', 'PNSUT.IS', 'POLHO.IS', 'POLTK.IS', 'PRdGS.IS', 'PRKAB.IS',
    'PRKME.IS', 'PRZMA.IS', 'PSDTc.IS', 'QNBtr.IS', 'QNBfk.IS', 'QUAGR.IS', 'RALYH.IS',
    'RAYsG.IS', 'REEDR.IS', 'ROYAl.IS', 'RNPOL.IS', 'RODRg.IS', 'RTALB.IS',
    'RUBNS.IS', 'RUZYE.IS', 'RYSAS.IS', 'SAFKR.IS', 'SAHOL.IS', 'SAMAT.IS',
    'SANEL.IS', 'SANFM.IS', 'SANKO.IS', 'SARKY.IS', 'SASA.IS', 'SAYAS.IS',
    'SDTTR.IS', 'SEGMn.IS', 'SEGYO.IS', 'SEKfk.IS', 'SEKUR.IS', 'SELEC.IS',
    'SELVA.IS', 'SERNT.IS', 'SEYKM.IS', 'SILVR.IS', 'SISE.IS', 'SKBNK.IS', 
    'SKTAS.IS', 'SKYLP.IS', 'SKYMD.IS', 'SMART.IS', 'SMRTG.IS', 'SMRVA.IS',
    'SNICA.IS', 'SNKRN.IS', 'SNPAM.IS', 'SODSN.IS', 'SOKE.IS', 'SOKM.IS',
    'SONME.IS', 'SRVGY.IS', 'SUMAS.IS', 'SUNTK.IS', 'SURGY.IS', 'SUWEN.IS',
    'TABGD.IS', 'TARKM.IS', 'TATEN.IS', 'TATGD.IS', 'TAVHL.IS', 'TBORG.IS',
    'TCELL.IS', 'Tckrc.IS', 'TDGYO.IS', 'TEKTU.IS', 'TERA.IS', 'TEZOL.IS',
    'TGSAS.IS', 'THYAO.IS', 'TKFEN.IS', 'TKNsa.IS', 'TLMAN.IS', 'TMPOL.IS',
    'TMSN.IS', 'TNZTP.IS', 'TOASO.IS', 'TRCAS.IS', 'TRGYO.IS', 'TRILC.IS',
    'TSKB.IS', 'TSPOR.IS', 'TTKOM.IS', 'TTRaK.IS', 'TUCLk.IS', 'TUKAS.IS',
    'TUPRS.IS', 'TUREX.IS', 'TURG.IS', 'TURSG.IS', 'UFUK.IS', 'ULAS.IS',
    'ULKER.IS', 'ULUFA.IS', 'ULUSE.IS', 'ULUun.IS', 'UNLU.IS', 'USAK.IS',
    'VAKBN.IS', 'VAKFN.IS', 'VAKKO.IS', 'VANGD.IS', 'VBTYZ.IS', 'VERUS.IS',
    'VESBE.IS', 'VESTL.IS', 'VKFYO.IS', 'VKING.IS', 'VRGYO.IS', 'VSNMD.IS',
    'YAPRK.IS', 'YATAS.IS', 'YAYLA.IS', 'YBTAS.IS', 'YEOTK.IS', 'YESIL.IS',
    'YGGYO.IS', 'YIGIT.IS', 'YKBNK.IS', 'YKSLN.IS', 'YONGA.IS', 'YUNSA.IS',
    'YYAPI.IS', 'YYLGD.IS', 'ZEDUR.IS', 'ZOREN.IS', 'ZRGYO.IS', 'A1CAP.IS',
    'A1YEN.IS', 'ACSEL.IS', 'ADEL.IS', 'ADESE.IS', 'ADGYO.IS', 'AEFES.IS',
    'AFYON.IS', 'AGESA.IS', 'AGHOL.IS', 'AGROT.IS', 'AHSGY.IS', 'AKBNK.IS',
    'AKCNS.IS', 'AKENR.IS', 'AKFIS.IS', 'AKFYE.IS', 'AKGRT.IS', 'AKSA.IS',
    'AKSEN.IS', 'AKSUE.IS', 'AKYHO.IS', 'ALARK.IS', 'ALBRK.IS', 'ALCAR.IS',
    'ALCTL.IS', 'ALFAS.IS', 'ALKA.IS', 'ALKIM.IS', 'ALKLC.IS',
    'ALTNY.IS', 'ALVES.IS', 'ANELE.IS', 'ANGEN.IS', 'ANHYT.IS', 'ANSGR.IS',
    'ARASE.IS', 'ARCLK.IS', 'ARDYZ.IS', 'ARENA.IS', 'ARMGD.IS', 'ARSAN.IS',
    'ARTMS.IS', 'ARZUM.IS', 'ASELS.IS', 'ASTOR.IS', 'ASUZU.IS', 'ATATP.IS',
    'ATEKS.IS', 'ATLAS.IS', 'ATSYH.IS', 'AVGYO.IS', 'AVHOL.IS', 'AVOD.IS',
    'AVPGY.IS', 'AYCES.IS', 'AYDEM.IS', 'AYEN.IS', 'AYES.IS', 'AYGAZ.IS',
    'AZTEK.IS', 'BAGFS.IS', 'BAHKM.IS', 'BAKAB.IS', 'BALAT.IS', 'BALSU.IS',
    'BANVT.IS', 'BARMA.IS', 'BASCM.IS', 'BASGZ.IS', 'BAYRK.IS', 'BEGYO.IS',
    'BERA.IS', 'BESLR.IS', 'BEYAZ.IS', 'BFREN.IS', 'BIENY.IS', 'BIGCH.IS', 
    'BIOEN.IS', 'BIZIM.IS', 'BJKAS.IS', 'BLCYT.IS', 'BLUME.IS', 'BMSCH.IS',
    'BMSTL.IS', 'BNTAS.IS', 'BObet.IS', 'BORLS.IS', 'BORSK.IS', 'BOSSA.IS',
    'BRISA.IS', 'BRKO.IS', 'BRKSN.IS', 'BRKVY.IS', 'BRLSM.IS', 'BRMEN.IS',
    'BRSAN.IS', 'BRYAT.IS', 'BSOKE.IS', 'BTCIM.IS', 'BULGs.IS', 'BURCE.IS',
    'BURVA.IS', 'BVSAN.IS', 'BYDNR.IS', 'CANTE.IS', 'CASA.IS', 'CATES.IS',
    'CCOLA.IS', 'CELHA.IS', 'CEMAS.IS', 'CEMTS.IS', 'CEMZY.IS', 'CEDEM.IS',
    'Cmbtn.IS', 'CIMSA.IS', 'CLEBI.IS', 'CMBTN.IS', 'CMEnT.IS', 'CONSE.IS',
    'COSMO.IS', 'CRDFA.IS', 'CRFSA.IS', 'CUSAN.IS', 'CVKmD.IS', 'CWENE.IS',
    'DAGI.IS', 'DAPGM.IS', 'DARDL.IS', 'DCTTr.IS', 'DENGE.IS', 'DERHL.IS',
    'DERIM.IS', 'DESA.IS', 'DESPC.IS', 'DEVA.IS', 'DGATE.IS', 'DGNMO.IS',
    'DIRIT.IS', 'DITAS.IS', 'DMRgd.IS', 'DMSAS.IS', 'DNISI.IS', # DİRİT -> DIRIT
    'DOAS.IS', 'DOBUR.IS', 'DOFER.IS', 'DOFRB.IS', 'DOGUB.IS', 'DOHOL.IS',
    'DOKTA.IS', 'DSTKF.IS', 'DUNYH.IS', 'DURDO.IS', 'DURkn.IS', 'DYOBY.IS',
    'DZgYO.IS', 'EBEBK.IS', 'ECILC.IS', 'ECZYT.IS', 'EDATA.IS', 'EDIP.IS',
    'EFORc.IS', 'EGEEN.IS', 'EGEGY.IS', 'EGEPO.IS', 'EGgUb.IS', 'EGPRO.IS',
    'EGSER.IS', 'EKIZ.IS', 'EKOS.IS', 'EKSUN.IS', 'ELITE.IS', 'EMKEL.IS',
    'EMNIS.IS', 'ENDAe.IS', 'ENERY.IS', 'ENJSA.IS', 'ENKAI.IS', 'ENSRI.IS',
    'ENTRA.IS', 'EPLAS.IS', 'ERBOS.IS', 'ERCb.IS', 'EREGL.IS', 'ERSU.IS',
    'ESCAR.IS', 'ESCOM.IS', 'ESEN.IS', 'ETILR.IS', 'ETYAT.IS', 'EUHOL.IS', 
    'EUKYO.IS', 'EUPWR.IS', 'EUREN.IS', 'EUYO.IS', 'FADE.IS', 'FENER.IS',
    'FLAP.IS', 'FMIzp.IS', 'FONET.IS', 'FORMT.IS', 'FORTE.IS', 'FRIGO.IS', 
     'MAALT.IS', 'MACKO.IS','MAGEN.IS','MAKIM.IS', 'MAKTK.IS', 'MANAS.IS', 'MARBL.IS',
    'MARKA.IS', 'MARMR.IS',  'MARTI.IS', 'MAVI.IS', 'MEDTR.IS', 'MEGAP.IS', 'MEGMT.IS',
    'MEKAG.IS', 'MEPET.IS', 'MERCN.IS', 'MERIT.IS', 'MERKO.IS', 'METRO.IS', 'MGROS.IS',
    'MHRGY.IS', 'MIATK.IS', 'MMCAS.IS', 'MNDRS.IS', 'MNDTR.IS', 'MOBTL.IS', 'MOGAN.IS',
    'MOPAS.IS', 'MPARK.IS', 'MRGYO.IS', 'MRSHL.IS', 'MSGYO.IS', 'MTRKS.IS', 'MTRYO.IS',
    'MZHLD.IS','HEKTS.IS', 'HKTM.IS', 'HDFGS.IS', 'HRKET.IS', 'HTTBt.IS', 'HUBVC.IS', 'HUNER.IS',
    'HURGZ.IS', 'ICBCT.IS', 'ICUGS.IS', 'IEYHO.IS', 'IHAAS.IS', 'IHEVA.IS',
    'IHGZT.IS', 'IHLAS.IS', 'IHLGM.IS', 'IHYAY.IS', 'IMASM.IS', 'INDES.IS',
    'INFO.IS', 'INGRM.IS', 'INTEK.IS', 'INTEM.IS', 'INVEO.IS', 'INVES.IS',
    'IPEKE.IS', 'ISBIR.IS', 'ISDMR.IS', 'ISFIN.IS', 'ISKPL.IS', 'ISMEN.IS',
    'ISSEN.IS', 'IZMDC.IS', 'IZenr.IS', 'IZFAS.IS', 'IZINV'
        ]
        
        self.results = []
        self.telegram = telegram_notifier
        self.failed = []
        self.data_fetcher = AlternativeDataFetcher()
        
    def get_stock_data(self, symbol, max_retries=2):
        """Veri çekme - 2 deneme"""
        for attempt in range(max_retries):
            try:
                df = self.data_fetcher.fetch_yahoo_alternative(symbol, period_days=120)
                if df is not None and len(df) >= 30:
                    return df
                time.sleep(0.5)
            except:
                if attempt < max_retries - 1:
                    time.sleep(1)
        return None
    
    def calculate_rsi(self, df, period=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, df):
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, macd - signal
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bandwidth = ((upper_band - lower_band) / (sma + 1e-10)) * 100
        percent_b = (df['Close'] - lower_band) / ((upper_band - lower_band) + 1e-10)
        return upper_band, lower_band, bandwidth, percent_b
    
    def calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_stochastic(self, df, period=14, smooth_k=3, smooth_d=3):
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        k_percent = 100 * ((df['Close'] - low_min) / ((high_max - low_min) + 1e-10))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()
        return k_percent, d_percent
    
    def calculate_obv(self, df):
        return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    def calculate_vwap(self, df):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        return (typical_price * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-10)
    
    def calculate_ichimoku(self, df):
        nine_period_high = df['High'].rolling(window=9).max()
        nine_period_low = df['Low'].rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) / 2
        
        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        return tenkan_sen, kijun_sen
    
    def analyze_stock(self, symbol):
        """Tam analiz"""
        df = self.get_stock_data(symbol)
        
        if df is None or len(df) < 30:
            return None
        
        try:
            current_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            avg_volume_20 = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
            
            rsi = float(self.calculate_rsi(df).iloc[-1])
            macd, signal, histogram = self.calculate_macd(df)
            macd_value = float(macd.iloc[-1])
            signal_value = float(signal.iloc[-1])
            histogram_value = float(histogram.iloc[-1])
            macd_cross = macd_value > signal_value
            macd_bullish = histogram_value > 0 and histogram.iloc[-2] <= 0
            
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma20
            ma200 = df['Close'].rolling(min(len(df), 200)).mean().iloc[-1]
            
            above_ma20 = current_price > ma20
            above_ma50 = current_price > ma50
            above_ma200 = current_price > ma200
            golden_cross = ma20 > ma50 and ma50 > ma200
            
            upper_bb, lower_bb, bb_bandwidth, bb_percent = self.calculate_bollinger_bands(df)
            current_bb_bandwidth = float(bb_bandwidth.iloc[-1])
            avg_bb_bandwidth = bb_bandwidth.rolling(50).mean().iloc[-1]
            bb_squeeze = current_bb_bandwidth < avg_bb_bandwidth * 0.8
            bb_percent_value = float(bb_percent.iloc[-1])
            bb_near_lower = bb_percent_value < 0.2
            bb_near_upper = bb_percent_value > 0.8
            
            atr = self.calculate_atr(df).iloc[-1]
            atr_percent = (atr / current_price) * 100
            
            stoch_k, stoch_d = self.calculate_stochastic(df)
            stoch_k_value = float(stoch_k.iloc[-1])
            stoch_d_value = float(stoch_d.iloc[-1])
            stoch_oversold = stoch_k_value < 20
            stoch_overbought = stoch_k_value > 80
            stoch_bullish_cross = stoch_k_value > stoch_d_value and stoch_k.iloc[-2] <= stoch_d.iloc[-2]
            
            obv = self.calculate_obv(df)
            obv_ma = obv.rolling(20).mean()
            obv_rising = obv.iloc[-1] > obv_ma.iloc[-1]
            
            vwap = float(self.calculate_vwap(df).iloc[-1])
            above_vwap = current_price > vwap
            
            tenkan, kijun = self.calculate_ichimoku(df)
            ichimoku_bullish = tenkan.iloc[-1] > kijun.iloc[-1]
            
            week52_high = df['Close'].rolling(min(len(df), 252)).max().iloc[-1]
            week52_low = df['Close'].rolling(min(len(df), 252)).min().iloc[-1]
            distance_from_low = ((current_price - week52_low) / week52_low) * 100
            distance_from_high = ((week52_high - current_price) / week52_high) * 100
            
            score = 0
            reasons = []
            risk_level = "Orta"
            
            if volume_ratio > 3:
                score += 20
                reasons.append(f"💥 Hacim patlaması: {volume_ratio:.1f}x")
            elif volume_ratio > 2:
                score += 15
                reasons.append(f"📊 Yüksek hacim: {volume_ratio:.1f}x")
            elif volume_ratio > 1.5:
                score += 10
                reasons.append(f"📈 Artmış hacim: {volume_ratio:.1f}x")
            
            if 40 < rsi < 60:
                score += 15
                reasons.append(f"✅ İdeal RSI: {rsi:.0f}")
            elif 30 < rsi < 70:
                score += 10
                reasons.append(f"⚡ İyi RSI: {rsi:.0f}")
            elif rsi < 30:
                reasons.append(f"⚠️ Aşırı satım: RSI {rsi:.0f}")
            elif rsi > 70:
                reasons.append(f"⚠️ Aşırı alım: RSI {rsi:.0f}")
                risk_level = "Yüksek"
            
            if macd_bullish:
                score += 15
                reasons.append("🚀 MACD yeni AL sinyali!")
            elif macd_cross:
                score += 10
                reasons.append("📈 MACD pozitif")
            
            if golden_cross:
                score += 20
                reasons.append("⭐ Golden Cross!")
            elif above_ma20 and above_ma50 and above_ma200:
                score += 18
                reasons.append("📊 Tüm MA'ların üzerinde")
            elif above_ma20 and above_ma50:
                score += 15
                reasons.append("📊 MA20+50 üzerinde")
            elif above_ma20:
                score += 8
                reasons.append("📈 MA20 üzerinde")
            
            if bb_squeeze and bb_near_lower:
                score += 15
                reasons.append("🎯 BB Squeeze + Alt bant (PATLAMA POTANSİYELİ!)")
            elif bb_squeeze:
                score += 10
                reasons.append("🔸 Bollinger daralması")
            elif bb_near_lower and not stoch_overbought:
                score += 8
                reasons.append("📍 Alt banda yakın")
            elif bb_near_upper:
                risk_level = "Yüksek"
                reasons.append("⚠️ Üst banda yakın")
            
            if stoch_bullish_cross and stoch_k_value < 50:
                score += 10
                reasons.append(f"💫 Stoch AL: {stoch_k_value:.0f}")
            elif stoch_oversold:
                score += 5
                reasons.append(f"🔽 Aşırı satım: Stoch {stoch_k_value:.0f}")
            
            if obv_rising and above_vwap:
                score += 10
                reasons.append("💪 OBV yükseliyor + VWAP üzeri")
            elif obv_rising:
                score += 5
                reasons.append("📈 OBV yükseliyor")
            elif above_vwap:
                score += 5
                reasons.append("✅ VWAP üzerinde")
            
            if ichimoku_bullish:
                score += 5
                reasons.append("🎌 Ichimoku bullish")
            
            if distance_from_low > 40 and distance_from_high > 30:
                score += 10
                reasons.append(f"📊 Dengeli pozisyon (52W: +{distance_from_low:.0f}%)")
            elif distance_from_low > 30:
                score += 8
                reasons.append(f"📍 52W düşükten +{distance_from_low:.0f}%")
            elif distance_from_low < 10:
                score += 5
                reasons.append(f"🔽 52W düşüğe yakın (+{distance_from_low:.0f}%)")
            
            if price_change > 3:
                score += 10
                reasons.append(f"💹 Güçlü artış: +{price_change:.1f}%")
            elif price_change > 1:
                score += 5
                reasons.append(f"📈 Artış: +{price_change:.1f}%")
            elif price_change < -3:
                risk_level = "Yüksek"
            
            if score >= 80:
                risk_level = "Düşük"
            elif score >= 60:
                risk_level = "Orta"
            elif rsi > 75 or stoch_overbought or bb_near_upper:
                risk_level = "Yüksek"
            
            stop_loss_price = current_price - (atr * 2)
            stop_loss_percent = ((current_price - stop_loss_price) / current_price) * 100
            
            return {
                'symbol': symbol.replace('.IS', ''),
                'price': current_price,
                'change_%': price_change,
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'macd_signal': 'AL' if macd_cross else 'BEKLE',
                'stoch_k': stoch_k_value,
                'bb_squeeze': bb_squeeze,
                'obv_rising': obv_rising,
                'above_vwap': above_vwap,
                'ichimoku_bullish': ichimoku_bullish,
                'above_ma20': above_ma20,
                'above_ma50': above_ma50,
                'above_ma200': above_ma200,
                'distance_from_low_%': distance_from_low,
                'atr_%': atr_percent,
                'stop_loss': stop_loss_price,
                'stop_loss_%': stop_loss_percent,
                'score': score,
                'risk_level': risk_level,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"{symbol} error: {e}")
            return None
    
    def scan_all(self, max_workers=5):
        """PARALEL TARAMA - Çok daha hızlı!"""
        print("="*70)
        print("🔍 BIST PROFESYONEL TARAMA (Paralel)")
        print("="*70)
        print(f"📊 Taranacak: {len(self.symbols)} | 🚀 {max_workers} paralel işlem\n")
        
        if self.telegram:
            self.telegram.send_message(
                f"🔍 *BIST PRO TARAMA*\n\n"
                f"📊 {len(self.symbols)} hisse\n"
                f"⏰ {datetime.now().strftime('%H:%M')}\n"
                f"🚀 Paralel tarama"
            )
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.analyze_stock, symbol): symbol 
                              for symbol in self.symbols}
            
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                        emoji = "🔥" if result['score'] >= 80 else "⭐" if result['score'] >= 60 else "✨"
                        print(f"[{completed}/{len(self.symbols)}] {symbol:12} {emoji} {result['score']:.0f}")
                    else:
                        self.failed.append(symbol)
                        print(f"[{completed}/{len(self.symbols)}] {symbol:12} ❌")
                except Exception as e:
                    self.failed.append(symbol)
                    print(f"[{completed}/{len(self.symbols)}] {symbol:12} ❌ {e}")
        
        elapsed = time.time() - start_time
        
        if len(self.results) == 0:
            return pd.DataFrame()
        
        self.df = pd.DataFrame(self.results)
        self.df = self.df.sort_values('score', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"✅ BAŞARILI: {len(self.results)} | ❌ BAŞARISIZ: {len(self.failed)}")
        print(f"⏱️  SÜRE: {elapsed:.1f} saniye")
        print(f"{'='*70}\n")
        
        return self.df
    
    def send_report(self, top_n=10, min_score=40):
        """Detaylı rapor - eski format"""
        if not self.telegram or len(self.df) == 0:
            return
        
        top = self.df[self.df['score'] >= min_score].head(top_n)
        
        if len(top) == 0:
            self.telegram.send_message(f"⚠️ {min_score}+ skor yok")
            return
        
        # Özet
        summary = f"✅ *PRO TARAMA TAMAM*\n\n"
        summary += f"📊 Top {len(top)}:\n\n"
        
        for _, row in top.iterrows():
            emoji = "🔥" if row['score'] >= 80 else "⭐" if row['score'] >= 60 else "✨"
            risk_emoji = "🟢" if row['risk_level'] == 'Düşük' else "🟡" if row['risk_level'] == 'Orta' else "🔴"
            
            summary += f"{emoji} *{row['symbol']}* - {row['score']:.0f}/100 {risk_emoji}\n"
            summary += f"   💰 {row['price']:.2f} TL ({row['change_%']:+.1f}%)\n"
            summary += f"   📊 Hacim: {row['volume_ratio']:.1f}x | RSI: {row['rsi']:.0f}\n\n"
        
        self.telegram.send_message(summary)
        
        # Top 3 detay
        for _, row in top.head(3).iterrows():
            risk_emoji = "🟢" if row['risk_level'] == 'Düşük' else "🟡" if row['risk_level'] == 'Orta' else "🔴"
            
            detail = f"📋 *{row['symbol']}* - {row['score']:.0f}/100\n"
            detail += f"━━━━━━━━━━━━━━\n"
            detail += f"{risk_emoji} Risk: *{row['risk_level']}*\n\n"
            detail += f"💰 Fiyat: {row['price']:.2f} TL ({row['change_%']:+.2f}%)\n"
            detail += f"📊 Hacim: {row['volume_ratio']:.1f}x\n"
            detail += f"⚡ RSI: {row['rsi']:.0f}\n"
            detail += f"🎯 MACD: {row['macd_signal']}\n"
            detail += f"📉 Stoch: {row['stoch_k']:.0f}\n"
            detail += f"🎌 Ichimoku: {'✅' if row['ichimoku_bullish'] else '❌'}\n"
            detail += f"💹 VWAP: {'Üzeri ✅' if row['above_vwap'] else 'Altı ❌'}\n"
            detail += f"🔸 BB Squeeze: {'Evet ⚡' if row['bb_squeeze'] else 'Hayır'}\n\n"
            detail += f"🛑 *Stop Loss:* {row['stop_loss']:.2f} TL (-{row['stop_loss_%']:.1f}%)\n"
            detail += f"📍 52W Düşükten: +{row['distance_from_low_%']:.0f}%\n\n"
            detail += f"*Güçlü Yönler:*\n"
            for r in row['reasons'][:7]:
                detail += f"• {r}\n"
            
            self.telegram.send_message(detail)
    
    def create_chart(self):
        """Grafik oluştur"""
        if len(self.df) == 0 or not self.telegram:
            return
        
        try:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            fig.suptitle('BIST Profesyonel Tarama Analizi', fontsize=18, fontweight='bold')
            
            # Top 10 skor
            ax1 = fig.add_subplot(gs[0, :2])
            top = self.df.head(10)
            colors = ['#2ecc71' if s >= 80 else '#f39c12' if s >= 60 else '#e74c3c' for s in top['score']]
            ax1.barh(top['symbol'], top['score'], color=colors)
            ax1.set_title('Top 10 Skor', fontweight='bold')
            ax1.set_xlabel('Skor')
            ax1.invert_yaxis()
            ax1.grid(alpha=0.3)
            
            # Risk dağılımı
            ax2 = fig.add_subplot(gs[0, 2])
            risk_counts = self.df['risk_level'].value_counts()
            colors_risk = ['#2ecc71', '#f39c12', '#e74c3c']
            ax2.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.0f%%',
                   colors=colors_risk, startangle=90)
            ax2.set_title('Risk Dağılımı', fontweight='bold')
            
            # RSI vs Skor
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.scatter(self.df['rsi'], self.df['score'], s=100, alpha=0.6,
                       c=self.df['score'], cmap='RdYlGn')
            ax3.axvline(30, color='r', linestyle='--', alpha=0.5)
            ax3.axvline(70, color='r', linestyle='--', alpha=0.5)
            ax3.set_xlabel('RSI')
            ax3.set_ylabel('Skor')
            ax3.set_title('RSI vs Skor', fontweight='bold')
            ax3.grid(alpha=0.3)
            
            # Hacim vs Skor
            ax4 = fig.add_subplot(gs[1, 1])
            scatter = ax4.scatter(self.df['volume_ratio'], self.df['score'], 
                                 s=100, c=self.df['change_%'], cmap='RdYlGn', alpha=0.6)
            ax4.set_xlabel('Hacim Oranı')
            ax4.set_ylabel('Skor')
            ax4.set_title('Hacim vs Skor', fontweight='bold')
            ax4.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Değişim %')
            
            # Stochastic vs Skor
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.scatter(self.df['stoch_k'], self.df['score'], s=100, alpha=0.6)
            ax5.axvline(20, color='g', linestyle='--', alpha=0.5)
            ax5.axvline(80, color='r', linestyle='--', alpha=0.5)
            ax5.set_xlabel('Stochastic K')
            ax5.set_ylabel('Skor')
            ax5.set_title('Stochastic vs Skor', fontweight='bold')
            ax5.grid(alpha=0.3)
            
            # Skor dağılımı
            ax6 = fig.add_subplot(gs[2, :])
            ax6.hist(self.df['score'], bins=15, color='#3498db', edgecolor='black', alpha=0.7)
            ax6.set_xlabel('Skor')
            ax6.set_ylabel('Hisse Sayısı')
            ax6.set_title('Skor Dağılımı', fontweight='bold')
            ax6.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            
            self.telegram.send_photo(buf, f"📊 {datetime.now().strftime('%d.%m %H:%M')}")
            plt.close()
            buf.close()
        except Exception as e:
            logger.error(f"Chart error: {e}")


def main():
    """Ana fonksiyon"""
    print("\n" + "="*70)
    print("BIST HİSSE TARAMA - HIZLI & TAM ÖZELLİKLİ")
    print("="*70 + "\n")
    
    # Telegram kontrolü
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("⚠️  Telegram yok - devam ediliyor")
        telegram = None
    else:
        telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)
        logger.info("✅ Telegram bağlandı")
    
    # Tarama başlat - PARALEL!
    scanner = BISTScanner(telegram_notifier=telegram)
    scanner.scan_all(max_workers=8)  # 8 paralel işlem
    
    if len(scanner.df) > 0:
        print("\n📊 EN İYİ 10:")
        print("-"*70)
        print(scanner.df.head(10)[['symbol', 'price', 'change_%', 'rsi', 'score']].to_string(index=False))
        
        # CSV kaydet
        scanner.df.to_csv('bist_scan_results.csv', index=False)
        logger.info("📁 Sonuçlar kaydedildi")
        
        # Detaylı rapor gönder
        scanner.send_report(top_n=10, min_score=40)
        
        # Grafik gönder
        scanner.create_chart()
        
        if telegram:
            telegram.send_message("✅ *TARAMA TAMAMLANDI!*")
    else:
        logger.error("❌ Tarama başarısız!")
        if telegram:
            telegram.send_message("❌ Veri alınamadı")
    
    print("\n✅ Bitti!")


if __name__ == "__main__":
    main()
