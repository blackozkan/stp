"""
BIST HİSSE TARAMA - PROFESYONEL VERSİYON
Tüm teknik göstergeler dahil
Yahoo Finance + Telegram entegrasyonu
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import requests
import io
import time
warnings.filterwarnings('ignore')

# Telegram
BOT_TOKEN = "8310745808:AAGpnfSna6-6AJ5I2FNKyES2Rdj_Xqu4b7o"
CHAT_ID = "1801093830"

class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, text):
        try:
            url = f"{self.base_url}/sendMessage"
            data = {'chat_id': self.chat_id, 'text': text, 'parse_mode': 'Markdown'}
            requests.post(url, data=data, timeout=10)
        except:
            pass
    
    def send_photo(self, image_buffer, caption=''):
        try:
            url = f"{self.base_url}/sendPhoto"
            image_buffer.seek(0)
            files = {'photo': image_buffer}
            data = {'chat_id': self.chat_id, 'caption': caption}
            requests.post(url, files=files, data=data, timeout=30)
        except:
            pass


class BISTScanner:
    def __init__(self, telegram_notifier=None):
        # BIST 30 + ek hisseler
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
        
    def get_stock_data(self, symbol):
        """Yahoo Finance'tan veri çek"""
        try:
            stock = yf.Ticker(symbol)
            for period in ['3mo', '6mo', '1y']:
                try:
                    df = stock.history(period=period)
                    if df is not None and len(df) >= 60:
                        return df
                except:
                    continue
            return None
        except:
            return None
    
    def calculate_rsi(self, df, period=14):
        """RSI - Relative Strength Index"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df):
        """MACD - Moving Average Convergence Divergence"""
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Bollinger Bandwidth (Squeeze tespiti)
        bandwidth = ((upper_band - lower_band) / sma) * 100
        
        # %B (Price position in bands)
        percent_b = (df['Close'] - lower_band) / (upper_band - lower_band)
        
        return upper_band, lower_band, bandwidth, percent_b
    
    def calculate_atr(self, df, period=14):
        """ATR - Average True Range (Volatilite)"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def calculate_stochastic(self, df, period=14, smooth_k=3, smooth_d=3):
        """Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()
        
        return k_percent, d_percent
    
    def calculate_obv(self, df):
        """OBV - On Balance Volume"""
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv
    
    def calculate_vwap(self, df):
        """VWAP - Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    
    def calculate_ichimoku(self, df):
        """Ichimoku Cloud - Basitleştirilmiş"""
        # Tenkan-sen (Conversion Line): 9 period
        nine_period_high = df['High'].rolling(window=9).max()
        nine_period_low = df['Low'].rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) / 2
        
        # Kijun-sen (Base Line): 26 period
        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        return tenkan_sen, kijun_sen
    
    def analyze_stock(self, symbol):
        """Gelişmiş teknik analiz"""
        df = self.get_stock_data(symbol)
        
        if df is None or len(df) < 60:
            return None
        
        try:
            # Temel Fiyat Bilgileri
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # Hacim Analizi
            avg_volume_20 = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
            
            # RSI
            rsi = self.calculate_rsi(df).iloc[-1]
            
            # MACD
            macd, signal, histogram = self.calculate_macd(df)
            macd_value = macd.iloc[-1]
            signal_value = signal.iloc[-1]
            histogram_value = histogram.iloc[-1]
            macd_cross = macd_value > signal_value
            macd_bullish = histogram_value > 0 and histogram.iloc[-2] <= 0  # Yeni AL sinyali
            
            # Hareketli Ortalamalar
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            ma200 = df['Close'].rolling(min(len(df), 200)).mean().iloc[-1]
            
            above_ma20 = current_price > ma20
            above_ma50 = current_price > ma50
            above_ma200 = current_price > ma200
            golden_cross = ma20 > ma50 and ma50 > ma200
            
            # Bollinger Bands
            upper_bb, lower_bb, bb_bandwidth, bb_percent = self.calculate_bollinger_bands(df)
            current_bb_bandwidth = bb_bandwidth.iloc[-1]
            avg_bb_bandwidth = bb_bandwidth.rolling(50).mean().iloc[-1]
            bb_squeeze = current_bb_bandwidth < avg_bb_bandwidth * 0.8  # Daralma
            bb_percent_value = bb_percent.iloc[-1]
            bb_near_lower = bb_percent_value < 0.2  # Alt banda yakın
            bb_near_upper = bb_percent_value > 0.8  # Üst banda yakın
            
            # ATR (Volatilite)
            atr = self.calculate_atr(df).iloc[-1]
            atr_percent = (atr / current_price) * 100
            
            # Stochastic
            stoch_k, stoch_d = self.calculate_stochastic(df)
            stoch_k_value = stoch_k.iloc[-1]
            stoch_d_value = stoch_d.iloc[-1]
            stoch_oversold = stoch_k_value < 20  # Aşırı satım
            stoch_overbought = stoch_k_value > 80  # Aşırı alım
            stoch_bullish_cross = stoch_k_value > stoch_d_value and stoch_k.iloc[-2] <= stoch_d.iloc[-2]
            
            # OBV (Hacim Trendi)
            obv = self.calculate_obv(df)
            obv_ma = obv.rolling(20).mean()
            obv_rising = obv.iloc[-1] > obv_ma.iloc[-1]
            
            # VWAP
            vwap = self.calculate_vwap(df).iloc[-1]
            above_vwap = current_price > vwap
            
            # Ichimoku
            tenkan, kijun = self.calculate_ichimoku(df)
            ichimoku_bullish = tenkan.iloc[-1] > kijun.iloc[-1]
            
            # 52 Haftalık Analiz
            week52_high = df['Close'].rolling(min(len(df), 252)).max().iloc[-1]
            week52_low = df['Close'].rolling(min(len(df), 252)).min().iloc[-1]
            distance_from_low = ((current_price - week52_low) / week52_low) * 100
            distance_from_high = ((week52_high - current_price) / week52_high) * 100
            
            # SKOR HESAPLAMA (Gelişmiş - 100 puan)
            score = 0
            reasons = []
            risk_level = "Orta"
            
            # 1. Hacim Analizi (Max 20)
            if volume_ratio > 3:
                score += 20
                reasons.append(f"💥 Hacim patlaması: {volume_ratio:.1f}x")
            elif volume_ratio > 2:
                score += 15
                reasons.append(f"📊 Yüksek hacim: {volume_ratio:.1f}x")
            elif volume_ratio > 1.5:
                score += 10
                reasons.append(f"📈 Artmış hacim: {volume_ratio:.1f}x")
            
            # 2. RSI (Max 15)
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
            
            # 3. MACD (Max 15)
            if macd_bullish:
                score += 15
                reasons.append("🚀 MACD yeni AL sinyali!")
            elif macd_cross:
                score += 10
                reasons.append("📈 MACD pozitif")
            
            # 4. Hareketli Ortalamalar (Max 20)
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
            
            # 5. Bollinger Bands (Max 15)
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
            
            # 6. Stochastic (Max 10)
            if stoch_bullish_cross and stoch_k_value < 50:
                score += 10
                reasons.append(f"💫 Stoch AL: {stoch_k_value:.0f}")
            elif stoch_oversold:
                score += 5
                reasons.append(f"🔽 Aşırı satım: Stoch {stoch_k_value:.0f}")
            
            # 7. OBV + VWAP (Max 10)
            if obv_rising and above_vwap:
                score += 10
                reasons.append("💪 OBV yükseliyor + VWAP üzeri")
            elif obv_rising:
                score += 5
                reasons.append("📈 OBV yükseliyor")
            elif above_vwap:
                score += 5
                reasons.append("✅ VWAP üzerinde")
            
            # 8. Ichimoku (Max 5)
            if ichimoku_bullish:
                score += 5
                reasons.append("🎌 Ichimoku bullish")
            
            # 9. 52 Haftalık Pozisyon (Max 10)
            if distance_from_low > 40 and distance_from_high > 30:
                score += 10
                reasons.append(f"📊 Dengeli pozisyon (52W: +{distance_from_low:.0f}%)")
            elif distance_from_low > 30:
                score += 8
                reasons.append(f"📍 52W düşükten +{distance_from_low:.0f}%")
            elif distance_from_low < 10:
                score += 5
                reasons.append(f"🔽 52W düşüğe yakın (+{distance_from_low:.0f}%)")
            
            # 10. Günlük Momentum (Max 10)
            if price_change > 3:
                score += 10
                reasons.append(f"💹 Güçlü artış: +{price_change:.1f}%")
            elif price_change > 1:
                score += 5
                reasons.append(f"📈 Artış: +{price_change:.1f}%")
            elif price_change < -3:
                risk_level = "Yüksek"
            
            # Risk Seviyesi Belirleme
            if score >= 80:
                risk_level = "Düşük"
            elif score >= 60:
                risk_level = "Orta"
            elif rsi > 75 or stoch_overbought or bb_near_upper:
                risk_level = "Yüksek"
            
            # Stop Loss Önerisi (ATR bazlı)
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
            print(f"  Analiz hatası: {str(e)[:30]}")
            return None
    
    def scan_all(self):
        """Tarama"""
        print("="*70)
        print("🔍 BIST PROFESYONEL TARAMA")
        print("="*70)
        print(f"📊 Taranacak: {len(self.symbols)}\n")
        
        if self.telegram:
            self.telegram.send_message(
                f"🔍 *BIST PRO TARAMA*\n\n"
                f"📊 {len(self.symbols)} hisse\n"
                f"⏰ {datetime.now().strftime('%H:%M')}\n"
                f"🎯 30+ teknik gösterge"
            )
        
        for i, symbol in enumerate(self.symbols):
            print(f"[{i+1}/{len(self.symbols)}] {symbol:12} ", end='', flush=True)
            
            result = self.analyze_stock(symbol)
            
            if result:
                self.results.append(result)
                emoji = "🔥" if result['score'] >= 80 else "⭐" if result['score'] >= 60 else "✨"
                print(f"{emoji} Skor: {result['score']:.0f} | Risk: {result['risk_level']}")
            else:
                self.failed.append(symbol)
                print(f"❌")
            
            if (i + 1) % 5 == 0:
                time.sleep(1)
        
        if len(self.results) == 0:
            print(f"\n❌ Hiçbir hisse analiz edilemedi!")
            self.df = pd.DataFrame()
            return self.df
        
        self.df = pd.DataFrame(self.results)
        self.df = self.df.sort_values('score', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"✅ BAŞARILI: {len(self.results)} | ❌ BAŞARISIZ: {len(self.failed)}")
        
        # Risk dağılımı
        risk_counts = self.df['risk_level'].value_counts()
        print(f"\n📊 RİSK DAĞILIMI:")
        for risk, count in risk_counts.items():
            print(f"   {risk}: {count} hisse")
        
        print(f"{'='*70}\n")
        
        return self.df
    
    def send_report(self, top_n=10, min_score=40):
        """Gelişmiş Telegram rapor"""
        if not self.telegram or len(self.df) == 0:
            return
        
        top = self.df[self.df['score'] >= min_score].head(top_n)
        
        if len(top) == 0:
            msg = f"⚠️ {min_score}+ skor yok\nEn yüksek: {self.df['score'].max():.0f}"
            self.telegram.send_message(msg)
            return
        
        # Özet rapor
        summary = f"✅ *PRO TARAMA TAMAM*\n\n"
        summary += f"📊 Başarılı: {len(self.df)} | Top: {len(top)}\n"
        
        # Risk istatistikleri
        low_risk = len(self.df[self.df['risk_level'] == 'Düşük'])
        medium_risk = len(self.df[self.df['risk_level'] == 'Orta'])
        high_risk = len(self.df[self.df['risk_level'] == 'Yüksek'])
        summary += f"🟢 Düşük Risk: {low_risk} | 🟡 Orta: {medium_risk} | 🔴 Yüksek: {high_risk}\n"
        summary += f"━━━━━━━━━━━━━━\n\n"
        
        for _, row in top.iterrows():
            emoji = "🔥" if row['score'] >= 80 else "⭐" if row['score'] >= 60 else "✨"
            risk_emoji = "🟢" if row['risk_level'] == 'Düşük' else "🟡" if row['risk_level'] == 'Orta' else "🔴"
            
            summary += f"{emoji} *{row['symbol']}* - {row['score']:.0f}/100 {risk_emoji}\n"
            summary += f"   💰 {row['price']:.2f} TL ({row['change_%']:+.1f}%)\n"
            summary += f"   📊 Hacim: {row['volume_ratio']:.1f}x | RSI: {row['rsi']:.0f}\n"
            summary += f"   🛑 Stop: {row['stop_loss']:.2f} TL (-{row['stop_loss_%']:.1f}%)\n\n"
        
        self.telegram.send_message(summary)
        
        # Top 3 detaylı analiz
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
        """Gelişmiş grafik"""
        if len(self.df) == 0 or not self.telegram:
            return
        
        try:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            fig.suptitle('BIST Profesyonel Tarama Analizi', fontsize=18, fontweight='bold')
            
            # 1. Top 10 Skor
            ax1 = fig.add_subplot(gs[0, :2])
            top = self.df.head(10)
            colors = ['#2ecc71' if s >= 80 else '#f39c12' if s >= 60 else '#e74c3c' 
                      for s in top['score']]
            ax1.barh(top['symbol'], top['score'], color=colors)
            ax1.set_title('Top 10 Skor', fontweight='bold')
            ax1.set_xlabel('Skor')
            ax1.invert_yaxis()
            ax1.grid(alpha=0.3)
            
            # 2. Risk Dağılımı (Pasta)
            ax2 = fig.add_subplot(gs[0, 2])
            risk_counts = self.df['risk_level'].value_counts()
            colors_risk = ['#2ecc71', '#f39c12', '#e74c3c']
            ax2.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.0f%%',
                   colors=colors_risk, startangle=90)
            ax2.set_title('Risk Dağılımı', fontweight='bold')
            
            # 3. RSI Dağılımı
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.scatter(self.df['rsi'], self.df['score'], s=100, alpha=0.6,
                       c=self.df['score'], cmap='RdYlGn')
            ax3.axvline(30, color='r', linestyle='--', alpha=0.5, label='Aşırı Satım')
            ax3.axvline(70, color='r', linestyle='--', alpha=0.5, label='Aşırı Alım')
            ax3.set_xlabel('RSI')
            ax3.set_ylabel('Skor')
            ax3.set_title('RSI vs Skor', fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)
            
            # 4. Hacim Analizi
            ax4 = fig.add_subplot(gs[1, 1])
            scatter = ax4.scatter(self.df['volume_ratio'], self.df['score'], 
                                 s=100, c=self.df['change_%'], cmap='RdYlGn', alpha=0.6)
            ax4.axvline(2, color='orange', linestyle='--', alpha=0.5, label='2x Hacim')
            ax4.set_xlabel('Hacim Oranı')
            ax4.set_ylabel('Skor')
            ax4.set_title('Hacim vs Skor', fontweight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Değişim %')
            
            # 5. Stochastic Dağılımı
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.scatter(self.df['stoch_k'], self.df['score'], s=100, alpha=0.6,
                       c=self.df['rsi'], cmap='coolwarm')
            ax5.axvline(20, color='g', linestyle='--', alpha=0.5, label='Oversold')
            ax5.axvline(80, color='r', linestyle='--', alpha=0.5, label='Overbought')
            ax5.set_xlabel('Stochastic K')
            ax5.set_ylabel('Skor')
            ax5.set_title('Stochastic vs Skor', fontweight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3)
            
            # 6. Skor Dağılımı
            ax6 = fig.add_subplot(gs[2, 0])
            ax6.hist(self.df['score'], bins=15, color='#3498db', alpha=0.7, edgecolor='black')
            ax6.axvline(self.df['score'].mean(), color='r', linestyle='--', 
                       linewidth=2, label=f'Ortalama: {self.df["score"].mean():.0f}')
            ax6.set_xlabel('Skor')
            ax6.set_ylabel('Hisse Sayısı')
            ax6.set_title('Skor Dağılımı', fontweight='bold')
            ax6.legend()
            ax6.grid(alpha=0.3)
            
            # 7. 52W Pozisyon
            ax7 = fig.add_subplot(gs[2, 1])
            ax7.scatter(self.df['distance_from_low_%'], self.df['score'], 
                       s=100, c=self.df['volume_ratio'], cmap='plasma', alpha=0.6)
            ax7.set_xlabel('52W Düşükten Uzaklık (%)')
            ax7.set_ylabel('Skor')
            ax7.set_title('52W Pozisyon vs Skor', fontweight='bold')
            ax7.grid(alpha=0.3)
            
            # 8. Teknik Gösterge Özeti
            ax8 = fig.add_subplot(gs[2, 2])
            indicators = {
                'BB Squeeze': len(self.df[self.df['bb_squeeze']]),
                'OBV Yükseliş': len(self.df[self.df['obv_rising']]),
                'VWAP Üzeri': len(self.df[self.df['above_vwap']]),
                'MA20 Üzeri': len(self.df[self.df['above_ma20']]),
                'Ichimoku+': len(self.df[self.df['ichimoku_bullish']])
            }
            ax8.barh(list(indicators.keys()), list(indicators.values()), 
                    color='#3498db', alpha=0.7)
            ax8.set_xlabel('Hisse Sayısı')
            ax8.set_title('Teknik Gösterge Özeti', fontweight='bold')
            ax8.grid(alpha=0.3)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            self.telegram.send_photo(buf, 
                f"📊 Profesyonel Analiz\n{datetime.now().strftime('%d.%m.%Y %H:%M')}")
            plt.close()
        except Exception as e:
            print(f"Grafik hatası: {e}")


def main():
    print("\n" + "="*70)
    print("BIST HİSSE TARAMA - PROFESYONEL VERSİYON")
    print("30+ Teknik Gösterge | ATR Stop Loss | Risk Analizi")
    print("="*70 + "\n")
    
    telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)
    scanner = BISTScanner(telegram_notifier=telegram)
    
    # Tarama
    scanner.scan_all()
    
    # Raporlar
    if len(scanner.df) > 0:
        print("\n" + "="*70)
        print("📊 EN İYİ 15 HİSSE (Detaylı)")
        print("="*70)
        
        display_cols = ['symbol', 'score', 'risk_level', 'price', 'change_%', 
                       'volume_ratio', 'rsi', 'macd_signal', 'stop_loss_%']
        print(scanner.df.head(15)[display_cols].to_string(index=False))
        
        # Telegram raporları
        scanner.send_report(top_n=15, min_score=35)
        scanner.create_chart()
        
        # Özel filtreler
        print("\n" + "="*70)
        print("🎯 ÖZEL FİLTRELER")
        print("="*70)
        
        # Düşük riskli + yüksek skorlu
        low_risk_high_score = scanner.df[
            (scanner.df['risk_level'] == 'Düşük') & 
            (scanner.df['score'] >= 60)
        ]
        if len(low_risk_high_score) > 0:
            print(f"\n🟢 DÜŞÜK RİSK + YÜKSEK SKOR ({len(low_risk_high_score)} adet):")
            print(low_risk_high_score[['symbol', 'score', 'price', 'stop_loss_%']].to_string(index=False))
            
            # Telegram'a özel filtre gönder
            if telegram:
                msg = f"🟢 *DÜŞÜK RİSK + YÜKSEK SKOR*\n\n"
                msg += f"📊 {len(low_risk_high_score)} adet bulundu:\n\n"
                for _, row in low_risk_high_score.head(5).iterrows():
                    msg += f"• *{row['symbol']}* - {row['score']:.0f}\n"
                    msg += f"  💰 {row['price']:.2f} TL\n"
                    msg += f"  🛑 Stop: -{row['stop_loss_%']:.1f}%\n\n"
                telegram.send_message(msg)
        
        # BB Squeeze + Düşük RSI (Patlama potansiyeli)
        squeeze_opportunities = scanner.df[
            (scanner.df['bb_squeeze']) & 
            (scanner.df['rsi'] < 50) &
            (scanner.df['score'] >= 40)
        ]
        if len(squeeze_opportunities) > 0:
            print(f"\n⚡ BOLLINGER SQUEEZE POTANSİYELİ ({len(squeeze_opportunities)} adet):")
            print(squeeze_opportunities[['symbol', 'score', 'rsi', 'price']].to_string(index=False))
            
            if telegram:
                msg = f"⚡ *PATLAMA POTANSİYELİ*\n"
                msg += f"(BB Squeeze + Düşük RSI)\n\n"
                for _, row in squeeze_opportunities.head(5).iterrows():
                    msg += f"💥 *{row['symbol']}* - Skor: {row['score']:.0f}\n"
                    msg += f"   RSI: {row['rsi']:.0f} | {row['price']:.2f} TL\n\n"
                telegram.send_message(msg)
        
        # Golden Cross
        golden_cross = scanner.df[
            (scanner.df['above_ma20']) & 
            (scanner.df['above_ma50']) & 
            (scanner.df['above_ma200'])
        ]
        if len(golden_cross) > 0:
            print(f"\n⭐ GOLDEN CROSS FORMASYON ({len(golden_cross)} adet):")
            print(golden_cross[['symbol', 'score', 'price', 'change_%']].to_string(index=False))
        
        # Final mesaj
        telegram.send_message(
            f"✅ *PROFESYONEL TARAMA TAMAMLANDI!*\n\n"
            f"📊 Analiz: {len(scanner.df)} hisse\n"
            f"🎯 30+ teknik gösterge\n"
            f"⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        )
    else:
        print("\n❌ Tarama başarısız!")
        telegram.send_message("❌ Tarama başarısız - veri çekilemedi")
    
    print("\n" + "="*70)
    print("✅ PROGRAM TAMAMLANDI!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
