"""
BIST MOMENTUM AVCISI - Strateji 1
Odak: Hacim + RSI + MACD + Bollinger Squeeze
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import requests
import io
import time
import os
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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


class DataFetcher:
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
    
    def fetch(self, symbol, period_days=90):
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
            
            return df if len(df) >= 20 else None
            
        except:
            return None


class MomentumScanner:
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
        self.fetcher = DataFetcher()
    
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
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_bollinger_squeeze(self, df, period=20, std_dev=2):
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        bandwidth = ((upper - lower) / (sma + 1e-10)) * 100
        avg_bandwidth = bandwidth.rolling(50).mean()
        
        is_squeeze = bandwidth < (avg_bandwidth * 0.7)
        percent_b = (df['Close'] - lower) / ((upper - lower) + 1e-10)
        
        return bandwidth.iloc[-1], is_squeeze.iloc[-1], percent_b.iloc[-1]
    
    def calculate_stochastic(self, df, period=14, smooth_k=3, smooth_d=3):
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        k_percent = 100 * ((df['Close'] - low_min) / ((high_max - low_min) + 1e-10))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()
        return k_percent, d_percent
    
    def analyze_stock(self, symbol):
        df = self.fetcher.fetch(symbol)
        
        if df is None or len(df) < 30:
            return None
        
        try:
            current_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            avg_volume_20 = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / (avg_volume_20 + 1e-10)
            
            rsi = float(self.calculate_rsi(df).iloc[-1])
            
            macd, signal, histogram = self.calculate_macd(df)
            macd_value = float(macd.iloc[-1])
            signal_value = float(signal.iloc[-1])
            histogram_value = float(histogram.iloc[-1])
            
            macd_bullish = macd_value > signal_value
            macd_cross = histogram_value > 0 and histogram.iloc[-2] <= 0
            macd_strong = histogram_value > histogram.iloc[-2]
            
            bb_width, is_squeeze, bb_position = self.calculate_bollinger_squeeze(df)
            
            stoch_k, stoch_d = self.calculate_stochastic(df)
            stoch_k_value = float(stoch_k.iloc[-1])
            
            score = 0
            signals = []
            risk_level = "Orta"
            
            if volume_ratio > 4:
                score += 35
                signals.append(f"💥 DEV HACİM: {volume_ratio:.1f}x")
            elif volume_ratio > 3:
                score += 30
                signals.append(f"🔥 HACİM PATLAMASI: {volume_ratio:.1f}x")
            elif volume_ratio > 2:
                score += 25
                signals.append(f"📊 Yüksek hacim: {volume_ratio:.1f}x")
            elif volume_ratio > 1.5:
                score += 15
                signals.append(f"📈 Artmış hacim: {volume_ratio:.1f}x")
            
            if 45 < rsi < 55:
                score += 25
                signals.append(f"🎯 Perfect RSI: {rsi:.0f}")
            elif 40 < rsi < 60:
                score += 20
                signals.append(f"✅ İyi RSI: {rsi:.0f}")
            elif 30 < rsi < 40:
                score += 15
                signals.append(f"📍 RSI toparlanıyor: {rsi:.0f}")
            elif rsi < 30:
                score += 10
                signals.append(f"🔽 Aşırı satım: RSI {rsi:.0f}")
            elif rsi > 70:
                signals.append(f"⚠️ AŞIRI ALIM: RSI {rsi:.0f}")
                risk_level = "Yüksek"
            
            if macd_cross:
                score += 25
                signals.append("🚀 MACD YENİ AL SİNYALİ!")
            elif macd_bullish and macd_strong:
                score += 20
                signals.append("💪 MACD güçlü pozitif")
            elif macd_bullish:
                score += 12
                signals.append("📈 MACD pozitif")
            
            if is_squeeze and bb_position < 0.3:
                score += 15
                signals.append("🎯 BB SQUEEZE + ALT BÖLGE!")
            elif is_squeeze:
                score += 10
                signals.append("⚡ Bollinger daralması")
            elif bb_position < 0.2:
                score += 8
                signals.append("📍 Alt banda yakın")
            
            if price_change > 5:
                score += 10
                signals.append(f"🚀 Güçlü yükseliş: +{price_change:.1f}%")
            elif price_change > 2:
                score += 6
                signals.append(f"📈 Artış: +{price_change:.1f}%")
            
            if score >= 80:
                risk_level = "Düşük"
            elif score >= 60 and risk_level != "Yüksek":
                risk_level = "Orta"
            
            if score >= 75 and volume_ratio > 2:
                signal_type = "🔥 GÜÇLÜ AL"
            elif score >= 60:
                signal_type = "✅ AL"
            elif score >= 45:
                signal_type = "👀 İZLE"
            else:
                signal_type = "⏸️ BEKLE"
            
            return {
                'symbol': symbol.replace('.IS', ''),
                'price': current_price,
                'change_%': price_change,
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'macd_signal': 'Pozitif' if macd_bullish else 'Negatif',
                'macd_cross': macd_cross,
                'bb_squeeze': is_squeeze,
                'bb_position': bb_position * 100,
                'stoch_k': stoch_k_value,
                'score': score,
                'risk_level': risk_level,
                'signal_type': signal_type,
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"{symbol} error: {e}")
            return None
    
    def scan_all(self, max_workers=8):
        print("="*70)
        print("🎯 MOMENTUM AVCISI - BIST TARAMASI")
        print("="*70)
        print(f"📊 Hisse: {len(self.symbols)} | 🚀 {max_workers} paralel\n")
        
        if self.telegram:
            self.telegram.send_message(
                f"🎯 *MOMENTUM AVCISI*\n\n"
                f"📊 {len(self.symbols)} hisse taranıyor\n"
                f"⏰ {datetime.now().strftime('%H:%M')}\n"
                f"🎲 Hacim + RSI + MACD + BB"
            )
        
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.analyze_stock, s): s 
                              for s in self.symbols}
            
            done = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                done += 1
                
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                        emoji = result['signal_type'].split()[0]
                        print(f"[{done}/{len(self.symbols)}] {symbol:12} {emoji} {result['score']:.0f}")
                    else:
                        self.failed.append(symbol)
                        print(f"[{done}/{len(self.symbols)}] {symbol:12} ❌")
                except:
                    self.failed.append(symbol)
                    print(f"[{done}/{len(self.symbols)}] {symbol:12} ❌")
        
        elapsed = time.time() - start
        
        if len(self.results) == 0:
            return pd.DataFrame()
        
        self.df = pd.DataFrame(self.results)
        self.df = self.df.sort_values('score', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"✅ Başarılı: {len(self.results)} | ❌ Başarısız: {len(self.failed)}")
        print(f"⏱️  Süre: {elapsed:.1f}s")
        print(f"{'='*70}\n")
        
        return self.df
    
    def print_beautiful_table(self, top_n=15):
        if len(self.df) == 0:
            return
        
        top = self.df.head(top_n)
        
        print("\n" + "="*110)
        print("🎯 MOMENTUM AVCISI - EN İYİ FIRSATLAR")
        print("="*110)
        print(f"{'HİSSE':<8} {'FİYAT':>9} {'HACİM':>7} {'RSI':>5} {'MACD':>8} {'BB':>10} {'STOCH':>7} {'SKOR':>5}  {'DURUM':<15}")
        print("-"*110)
        
        for _, row in top.iterrows():
            symbol = row['symbol']
            price = f"{row['price']:.2f}"
            volume = f"{row['volume_ratio']:.1f}x"
            rsi = f"{row['rsi']:.0f}"
            
            if row['macd_cross']:
                macd = "🚀AL"
            elif row['macd_signal'] == 'Pozitif':
                macd = "Poz"
            else:
                macd = "Neg"
            
            bb = "Squeeze⚡" if row['bb_squeeze'] else f"{row['bb_position']:.0f}%"
            stoch = f"{row['stoch_k']:.0f}"
            score = f"{row['score']:.0f}"
            durum = row['signal_type']
            
            print(f"{symbol:<8} {price:>9} {volume:>7} {rsi:>5} {macd:>8} {bb:>10} {stoch:>7} {score:>5}  {durum:<15}")
        
        print("="*110 + "\n")
    
    def send_report(self, top_n=15):
        if not self.telegram or len(self.df) == 0:
            return
        
        buy_signals = self.df[self.df['score'] >= 60].head(top_n)
        
        if len(buy_signals) == 0:
            self.telegram.send_message("⚠️ Bugün güçlü sinyal yok")
            return
        
        msg = "🎯 *MOMENTUM AVCISI*\n\n"
        msg += f"```\n"
        msg += f"{'HİSSE':<7} {'FİYAT':>7} {'HCM':>5} {'RSI':>4} {'SKOR':>4}\n"
        msg += f"{'-'*32}\n"
        
        for _, row in buy_signals.iterrows():
            symbol = row['symbol'][:7]
            price = f"{row['price']:.2f}"
            volume = f"{row['volume_ratio']:.1f}x"
            rsi = f"{row['rsi']:.0f}"
            score = f"{row['score']:.0f}"
            
            msg += f"{symbol:<7} {price:>7} {volume:>5} {rsi:>4} {score:>4}\n"
        
        msg += "```\n"
        msg += f"\n📊 {len(buy_signals)} fırsat bulundu!"
        
        self.telegram.send_message(msg)
    
    def create_chart(self):
        if len(self.df) == 0 or not self.telegram:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            top15 = self.df.head(15)
            
            colors = []
            for _, row in top15.iterrows():
                if '🔥' in row['signal_type']:
                    colors.append('#e74c3c')
                elif '✅' in row['signal_type']:
                    colors.append('#f39c12')
                else:
                    colors.append('#95a5a6')
            
            bars = ax.barh(top15['symbol'], top15['score'], color=colors)
            
            for i, (bar, row) in enumerate(zip(bars, top15.iterrows())):
                _, r = row
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                       f"{r['score']:.0f} | RSI:{r['rsi']:.0f} | {r['volume_ratio']:.1f}x",
                       va='center', fontsize=9)
            
            ax.set_xlabel('Skor', fontsize=12, fontweight='bold')
            ax.set_title('🎯 MOMENTUM AVCISI - En İyi 15 Fırsat', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            ax.set_xlim(0, 100)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            self.telegram.send_photo(buf, f"📊 {datetime.now().strftime('%d.%m %H:%M')}")
            plt.close()
            buf.close()
        except Exception as e:
            logger.error(f"Chart error: {e}")


def main():
    print("\n" + "="*70)
    print("🎯 MOMENTUM AVCISI")
    print("="*70 + "\n")
    
    telegram = None
    if BOT_TOKEN and CHAT_ID:
        telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)
        logger.info("✅ Telegram bağlı")
    else:
        logger.warning("⚠️ Telegram yok")
    
    scanner = MomentumScanner(telegram_notifier=telegram)
    scanner.scan_all(max_workers=8)
    
    if len(scanner.df) > 0:
        scanner.print_beautiful_table(top_n=15)
        scanner.df.to_csv('momentum_scan.csv', index=False)
        logger.info("📁 Sonuçlar kaydedildi")
        scanner.send_report(top_n=15)
        scanner.create_chart()
        
        if telegram:
            telegram.send_message("✅ *TARAMA TAMAM!*")
    else:
        logger.error("❌ Veri yok")
    
    print("\n✅ Bitti!")


if __name__ == "__main__":
    main()
