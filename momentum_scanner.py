def scan_all(self, max_workers=8):
        """Paralel tarama"""
        print("="*70)
        print("🎯 MOMENTUM AVCISI - BIST TARAMASI")
        print("="*70)
        print(f"📊 Hisse: {len(self.symbols)} | 🚀 {max_workers} paralel\n")
        
        if self.telegram:
            self.telegram.send_message(
                f"🎯 *MOMENTUM AVCISI*\n\n"
                f"📊 {len(self.symbols)} hisse taranıyor\n"
                f"⏰ {datetime.now().strftime('%H:%M')}\n"
                f"🎲 Hacim + RSI + MACD + BB Squeeze"
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
                except Exception as e:
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
    
    def print_beautiful_table(self, top_n=10):
        """Güzel formatlanmış tablo yazdır"""
        if len(self.df) == 0:
            return
        
        top = self.df.head(top_n)
        
        print("\n" + "="*100)
        print("🎯 MOMENTUM AVCISI - EN İYİ FIRSATLAR")
        print("="*100)
        print(f"{'HİSSE':<8} {'FİYAT':>8} {'HACİM':>7} {'RSI':>5} {'MACD':>8} {'BB':>8} {'STOCH':>7} {'SKOR':>5}  {'DURUM':<15}")
        print("-"*100)
        
        for _, row in top.iterrows():
            symbol = row['symbol']
            price = row['price']
            volume = f"{row['volume_ratio']:.1f}x"
            rsi = f"{row['rsi']:.0f}"
            macd = "AL" if row['macd_cross'] else row['macd_signal'][:"""
BIST MOMENTUM AVCISI - Strateji 1
Odak: Hacim + RSI + MACD + Bollinger Squeeze
Basit, etkili, net sinyaller!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    """
    MOMENTUM AVCISI STRATEJİSİ
    4 Temel Gösterge:
    1. HACİM (En önemli - momentum kanıtı)
    2. RSI (Aşırı bölgeler ve momentum gücü)
    3. MACD (Trend yönü ve momentum değişimi)
    4. BOLLINGER SQUEEZE (Patlama potansiyeli)
    """
    
    def __init__(self, telegram_notifier=None):
        self.symbols = [
            'AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'YKBNK.IS', 'HALKB.IS', 'VAKBN.IS',
            'KCHOL.IS', 'SAHOL.IS', 'DOHOL.IS', 'AGHOL.IS',
            'THYAO.IS', 'TUPRS.IS', 'PETKM.IS', 'ASELS.IS', 'EREGL.IS',
            'ARCLK.IS', 'VESTL.IS', 'TOASO.IS', 'FROTO.IS', 'SISE.IS',
            'TTKOM.IS', 'TCELL.IS', 'ENKA.IS', 'AKSEN.IS',
            'KRDMD.IS', 'SODA.IS', 'SASA.IS', 'BIMAS.IS', 'ENKAI.IS',
            'TTRAK.IS', 'PGSUS.IS', 'TAVHL.IS', 'KOZAL.IS', 'EKGYO.IS',
            'MGROS.IS', 'SOKM.IS', 'ULKER.IS', 'KONTR.IS', 'AEFES.IS'
        ]
        
        self.results = []
        self.telegram = telegram_notifier
        self.failed = []
        self.fetcher = DataFetcher()
    
    def calculate_rsi(self, df, period=14):
        """RSI - Momentum gücü"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, df):
        """MACD - Trend ve momentum yönü"""
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_bollinger_squeeze(self, df, period=20, std_dev=2):
        """Bollinger Bands - Patlama potansiyeli"""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        bandwidth = ((upper - lower) / (sma + 1e-10)) * 100
        avg_bandwidth = bandwidth.rolling(50).mean()
        
        # Squeeze: Mevcut bandwidth, ortalamanın %70'inden küçükse
        is_squeeze = bandwidth < (avg_bandwidth * 0.7)
        
        # Fiyatın bantlara göre pozisyonu
        percent_b = (df['Close'] - lower) / ((upper - lower) + 1e-10)
        
        return bandwidth.iloc[-1], is_squeeze.iloc[-1], percent_b.iloc[-1]
    
    def analyze_stock(self, symbol):
        """Momentum analizi - Sadece 4 gösterge!"""
        df = self.fetcher.fetch(symbol)
        
        if df is None or len(df) < 30:
            return None
        
        try:
            # Temel fiyat bilgileri
            current_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # ═══════════════════════════════════════
            # 1️⃣ HACİM ANALİZİ (En Önemli!)
            # ═══════════════════════════════════════
            avg_volume_20 = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / (avg_volume_20 + 1e-10)
            
            # ═══════════════════════════════════════
            # 2️⃣ RSI - Momentum Gücü
            # ═══════════════════════════════════════
            rsi = float(self.calculate_rsi(df).iloc[-1])
            
            # ═══════════════════════════════════════
            # 3️⃣ MACD - Trend Yönü
            # ═══════════════════════════════════════
            macd, signal, histogram = self.calculate_macd(df)
            macd_value = float(macd.iloc[-1])
            signal_value = float(signal.iloc[-1])
            histogram_value = float(histogram.iloc[-1])
            
            # MACD sinyalleri
            macd_bullish = macd_value > signal_value  # Pozitif momentum
            macd_cross = histogram_value > 0 and histogram.iloc[-2] <= 0  # Yeni AL sinyali
            macd_strong = histogram_value > histogram.iloc[-2]  # Güçleniyor
            
            # ═══════════════════════════════════════
            # 4️⃣ BOLLINGER SQUEEZE - Patlama Potansiyeli
            # ═══════════════════════════════════════
            bb_width, is_squeeze, bb_position = self.calculate_bollinger_squeeze(df)
            
            # ═══════════════════════════════════════
            # SKORLAMA SİSTEMİ (Toplam 100 puan)
            # ═══════════════════════════════════════
            score = 0
            signals = []
            risk_level = "Orta"
            
            # HACİM SKORU (0-35 puan) - En önemli!
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
            elif volume_ratio < 0.7:
                signals.append(f"⚠️ Düşük hacim: {volume_ratio:.1f}x")
            
            # RSI SKORU (0-25 puan)
            if 45 < rsi < 55:  # İdeal momentum bölgesi
                score += 25
                signals.append(f"🎯 Perfect RSI: {rsi:.0f}")
            elif 40 < rsi < 60:  # İyi momentum
                score += 20
                signals.append(f"✅ İyi RSI: {rsi:.0f}")
            elif 30 < rsi < 40:  # Toparlanma fırsatı
                score += 15
                signals.append(f"📍 RSI toparlanıyor: {rsi:.0f}")
            elif rsi < 30:  # Aşırı satım - dikkatli olun
                score += 10
                signals.append(f"🔽 Aşırı satım: RSI {rsi:.0f}")
            elif rsi > 70:  # Aşırı alım - risk!
                signals.append(f"⚠️ AŞIRI ALIM: RSI {rsi:.0f}")
                risk_level = "Yüksek"
            
            # MACD SKORU (0-25 puan)
            if macd_cross:  # Yeni AL sinyali
                score += 25
                signals.append("🚀 MACD YENİ AL SİNYALİ!")
            elif macd_bullish and macd_strong:  # Güçlü pozitif momentum
                score += 20
                signals.append("💪 MACD güçlü pozitif")
            elif macd_bullish:  # Pozitif ama zayıf
                score += 12
                signals.append("📈 MACD pozitif")
            elif not macd_bullish and macd_strong:  # Negatif ama güçleniyor
                score += 8
                signals.append("⚡ MACD toparlanıyor")
            else:
                signals.append("📉 MACD negatif")
            
            # BOLLINGER SQUEEZE SKORU (0-15 puan)
            if is_squeeze and bb_position < 0.3:  # Alt bölgede sıkışma
                score += 15
                signals.append("🎯 BB SQUEEZE + ALT BÖLGE = PATLAMA FIRSATI!")
            elif is_squeeze:  # Genel sıkışma
                score += 10
                signals.append("⚡ Bollinger daralması - patlama yakın")
            elif bb_position < 0.2:  # Alt banta yakın
                score += 8
                signals.append("📍 Alt banda yakın")
            elif bb_position > 0.85:  # Üst banta yakın - risk!
                signals.append("⚠️ Üst banda yakın - dikkat")
                risk_level = "Yüksek"
            
            # FİYAT HAREKETİ BONUS (0-10 puan)
            if price_change > 5:
                score += 10
                signals.append(f"🚀 Güçlü yükseliş: +{price_change:.1f}%")
            elif price_change > 2:
                score += 6
                signals.append(f"📈 Artış: +{price_change:.1f}%")
            elif price_change < -5:
                risk_level = "Yüksek"
                signals.append(f"⚠️ Düşüş: {price_change:.1f}%")
            
            # Risk seviyesi düzeltmesi
            if score >= 80:
                risk_level = "Düşük"
            elif score >= 60 and risk_level != "Yüksek":
                risk_level = "Orta"
            
            # Sinyal kategorisi
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
                'bb_position': bb_position * 100,  # Yüzde olarak
                'score': score,
                'risk_level': risk_level,
                'signal_type': signal_type,
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"{symbol} error: {e}")
            return None
    
    def scan_all(self, max_workers=8):
        """Paralel tarama"""
        print("="*70)
        print("🎯 MOMENTUM AVCISI - BIST TARAMASI")
        print("="*70)
        print(f"📊 Hisse: {len(self.symbols)} | 🚀 {max_workers} paralel\n")
        
        if self.telegram:
            self.telegram.send_message(
                f"🎯 *MOMENTUM AVCISI*\n\n"
                f"📊 {len(self.symbols)} hisse taranıyor\n"
                f"⏰ {datetime.now().strftime('%H:%M')}\n"
                f"🎲 Hacim + RSI + MACD + BB Squeeze"
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
                except Exception as e:
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
        """Güzel formatlanmış tablo"""
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
            
            # MACD düzgün göster
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
        """Sadece tablo - detay yok!"""
        if not self.telegram or len(self.df) == 0:
            return
        
        buy_signals = self.df[self.df['score'] >= 60].head(top_n)
        
        if len(buy_signals) == 0:
            self.telegram.send_message("⚠️ Bugün güçlü sinyal yok")
            return
        
        # Sadece özet tablo
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
        """Basit grafik - sadece top 15"""
        if len(self.df) == 0 or not self.telegram:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            top15 = self.df.head(15)
            
            # Renkler
            colors = []
            for _, row in top15.iterrows():
                if '🔥' in row['signal_type']:
                    colors.append('#e74c3c')  # Kırmızı - Güçlü AL
                elif '✅' in row['signal_type']:
                    colors.append('#f39c12')  # Turuncu - AL
                else:
                    colors.append('#95a5a6')  # Gri
            
            # Bar chart
            bars = ax.barh(top15['symbol'], top15['score'], color=colors)
            
            # Skorları barlara yaz
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
    print("🎯 MOMENTUM AVCISI - Basit & Etkili")
    print("="*70 + "\n")
    
    telegram = None
    if BOT_TOKEN and CHAT_ID:
        telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)
        logger.info("✅ Telegram bağlı")
    else:
        logger.warning("⚠️  Telegram yok")
    
    scanner = MomentumScanner(telegram_notifier=telegram)
    scanner.scan_all(max_workers=8)
    
    if len(scanner.df) > 0:
        # Güzel tablo yazdır
        scanner.print_beautiful_table(top_n=15)
        
        # CSV kaydet
        scanner.df.to_csv('momentum_scan.csv', index=False)
        logger.info("📁 Sonuçlar kaydedildi")
        
        # Telegram'a sadece tablo gönder
        scanner.send_report(top_n=15)
        
        # Basit grafik gönder
        scanner.create_chart()
        
        if telegram:
            telegram.send_message("✅ *TARAMA TAMAM!*")
    else:
        logger.error("❌ Veri yok")
    
    print("\n✅ Bitti!")


if __name__ == "__main__":
    main()
