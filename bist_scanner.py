"""
BIST HİSSE TARAMA - GitHub Actions İçin Optimize Edilmiş
Yahoo Finance alternatifi + hata toleransı + retry mekanizması
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

# Ayarlar
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Telegram - Sadece environment variable'dan al
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
            logger.warning("Telegram credentials missing, skipping message")
            return False
        try:
            url = f"{self.base_url}/sendMessage"
            data = {'chat_id': self.chat_id, 'text': text, 'parse_mode': 'Markdown'}
            response = self.session.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram message error: {e}")
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
    """Yahoo Finance yerine alternatif veri kaynakları"""
    
    def __init__(self):
        self.session = self._create_session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'tr-TR,tr;q=0.9',
        }
    
    def _create_session(self):
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def fetch_yahoo_alternative(self, symbol, period_days=90):
        """Yahoo Finance v8 API (daha güvenilir)"""
        try:
            symbol_clean = symbol.replace('.IS', '.IS')
            end_time = int(time.time())
            start_time = end_time - (period_days * 24 * 60 * 60)
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol_clean}"
            params = {
                'period1': start_time,
                'period2': end_time,
                'interval': '1d',
                'events': 'history'
            }
            
            response = self.session.get(url, params=params, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"{symbol}: HTTP {response.status_code}")
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
            
            logger.info(f"{symbol}: {len(df)} günlük veri alındı")
            return df
            
        except Exception as e:
            logger.error(f"{symbol} fetch error: {e}")
            return None
    
    def fetch_investing_com(self, symbol):
        """Investing.com scraping (yedek)"""
        # Bu kısım daha karmaşık, şimdilik atlıyoruz
        pass


class BISTScanner:
    def __init__(self, telegram_notifier=None):
        # Genişletilmiş ve doğrulanmış BIST sembolleri
        self.symbols = [
            # Bankalar
            'AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'YKBNK.IS', 'HALKB.IS', 'VAKBN.IS',
            # Holdingler
            'KCHOL.IS', 'SAHOL.IS', 'DOHOL.IS', 'AGHOL.IS',
            # Sanayi
            'THYAO.IS', 'TUPRS.IS', 'PETKM.IS', 'ASELS.IS', 'EREGL.IS',
            'ARCLK.IS', 'VESTL.IS', 'TOASO.IS', 'FROTO.IS', 'SISE.IS',
            # Telekomünikasyon
            'TTKOM.IS', 'TCELL.IS',
            # Enerji
            'ENKA.IS', 'AKSEN.IS',
            # Diğer
            'KRDMD.IS', 'SODA.IS', 'SASA.IS', 'BIMAS.IS', 'ENKAI.IS',
            'TTRAK.IS', 'PGSUS.IS', 'TAVHL.IS', 'KOZAL.IS', 'EKGYO.IS'
        ]
        
        self.results = []
        self.telegram = telegram_notifier
        self.failed = []
        self.data_fetcher = AlternativeDataFetcher()
        
    def get_stock_data(self, symbol, max_retries=3):
        """Geliştirilmiş veri çekme - retry mekanizması ile"""
        for attempt in range(max_retries):
            try:
                # Ana yöntem: Yahoo Finance v8 API
                df = self.data_fetcher.fetch_yahoo_alternative(symbol, period_days=120)
                
                if df is not None and len(df) >= 30:
                    return df
                
                # Bekleme süresi - exponential backoff
                wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                logger.warning(f"{symbol}: Attempt {attempt+1} failed, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"{symbol} error on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def calculate_rsi(self, df, period=14):
        """RSI hesaplama"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)  # Division by zero önleme
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, df):
        """MACD hesaplama"""
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
        bandwidth = ((upper_band - lower_band) / (sma + 1e-10)) * 100
        percent_b = (df['Close'] - lower_band) / ((upper_band - lower_band) + 1e-10)
        return upper_band, lower_band, bandwidth, percent_b
    
    def calculate_stochastic(self, df, period=14, smooth_k=3, smooth_d=3):
        """Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        k_percent = 100 * ((df['Close'] - low_min) / ((high_max - low_min) + 1e-10))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()
        return k_percent, d_percent
    
    def calculate_obv(self, df):
        """On Balance Volume"""
        return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    def calculate_vwap(self, df):
        """Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        return (typical_price * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-10)
    
    def analyze_stock(self, symbol):
        """Hisse analizi - hata kontrolü geliştirilmiş"""
        df = self.get_stock_data(symbol)
        
        if df is None or len(df) < 30:
            logger.warning(f"{symbol}: Yetersiz veri")
            return None
        
        try:
            # Temel metrikler
            current_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # Hacim analizi
            avg_volume_20 = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
            
            # Teknik göstergeler
            rsi = float(self.calculate_rsi(df).iloc[-1])
            macd, signal, histogram = self.calculate_macd(df)
            macd_value = float(macd.iloc[-1])
            signal_value = float(signal.iloc[-1])
            histogram_value = float(histogram.iloc[-1])
            
            # Hareketli ortalamalar
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma20
            ma200 = df['Close'].rolling(min(len(df), 200)).mean().iloc[-1]
            
            # Bollinger Bands
            upper_bb, lower_bb, bb_bandwidth, bb_percent = self.calculate_bollinger_bands(df)
            current_bb_bandwidth = float(bb_bandwidth.iloc[-1])
            bb_percent_value = float(bb_percent.iloc[-1])
            
            # Stochastic
            stoch_k, stoch_d = self.calculate_stochastic(df)
            stoch_k_value = float(stoch_k.iloc[-1])
            stoch_d_value = float(stoch_d.iloc[-1])
            
            # OBV ve VWAP
            obv = self.calculate_obv(df)
            obv_ma = obv.rolling(20).mean()
            vwap = float(self.calculate_vwap(df).iloc[-1])
            
            # 52 hafta high/low
            week52_high = df['Close'].rolling(min(len(df), 252)).max().iloc[-1]
            week52_low = df['Close'].rolling(min(len(df), 252)).min().iloc[-1]
            distance_from_low = ((current_price - week52_low) / week52_low) * 100
            
            # Skorlama sistemi
            score = 0
            reasons = []
            risk_level = "Orta"
            
            # Hacim skoru
            if volume_ratio > 2.5:
                score += 20
                reasons.append(f"💥 Hacim patlaması: {volume_ratio:.1f}x")
            elif volume_ratio > 1.5:
                score += 12
                reasons.append(f"📊 Artmış hacim: {volume_ratio:.1f}x")
            
            # RSI skoru
            if 40 < rsi < 60:
                score += 15
                reasons.append(f"✅ İdeal RSI: {rsi:.0f}")
            elif 30 < rsi < 70:
                score += 8
            elif rsi < 30:
                score += 5
                reasons.append(f"🔽 Aşırı satım: RSI {rsi:.0f}")
            elif rsi > 75:
                risk_level = "Yüksek"
                reasons.append(f"⚠️ Aşırı alım: RSI {rsi:.0f}")
            
            # MACD skoru
            if histogram_value > 0 and histogram.iloc[-2] <= 0:
                score += 15
                reasons.append("🚀 MACD AL sinyali!")
            elif macd_value > signal_value:
                score += 8
            
            # MA skoru
            if current_price > ma20 and ma20 > ma50:
                score += 15
                reasons.append("📊 Trend güçlü")
            elif current_price > ma20:
                score += 8
            
            # Bollinger skoru
            if bb_percent_value < 0.2:
                score += 10
                reasons.append("📍 Alt banda yakın")
            
            # Stochastic skoru
            if stoch_k_value < 30:
                score += 8
                reasons.append(f"🔽 Stoch oversold: {stoch_k_value:.0f}")
            
            # Risk seviyesi belirleme
            if score >= 75:
                risk_level = "Düşük"
            elif rsi > 75 or bb_percent_value > 0.85:
                risk_level = "Yüksek"
            
            return {
                'symbol': symbol.replace('.IS', ''),
                'price': current_price,
                'change_%': price_change,
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'macd_signal': 'AL' if macd_value > signal_value else 'BEKLE',
                'score': score,
                'risk_level': risk_level,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"{symbol} analysis error: {e}")
            return None
    
    def scan_all(self):
        """Tüm hisseleri tara"""
        print("="*70)
        print("🔍 BIST PROFESYONEL TARAMA (GitHub Optimized)")
        print("="*70)
        print(f"📊 Taranacak: {len(self.symbols)}\n")
        
        if self.telegram:
            self.telegram.send_message(
                f"🔍 *BIST TARAMA BAŞLADI*\n\n"
                f"📊 {len(self.symbols)} hisse\n"
                f"⏰ {datetime.now().strftime('%H:%M')}"
            )
        
        for i, symbol in enumerate(self.symbols):
            print(f"[{i+1}/{len(self.symbols)}] {symbol:12} ", end='', flush=True)
            
            result = self.analyze_stock(symbol)
            
            if result:
                self.results.append(result)
                emoji = "🔥" if result['score'] >= 75 else "⭐" if result['score'] >= 50 else "✨"
                print(f"{emoji} Skor: {result['score']:.0f}")
            else:
                self.failed.append(symbol)
                print(f"❌")
            
            # Rate limiting - GitHub Actions için önemli
            time.sleep(1.5)
        
        if len(self.results) == 0:
            logger.error("Hiç veri alınamadı!")
            return pd.DataFrame()
        
        self.df = pd.DataFrame(self.results)
        self.df = self.df.sort_values('score', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"✅ BAŞARILI: {len(self.results)} | ❌ BAŞARISIZ: {len(self.failed)}")
        print(f"{'='*70}\n")
        
        return self.df
    
    def send_report(self, top_n=10, min_score=40):
        """Telegram raporu gönder"""
        if not self.telegram or len(self.df) == 0:
            return
        
        top = self.df[self.df['score'] >= min_score].head(top_n)
        
        if len(top) == 0:
            self.telegram.send_message(f"⚠️ {min_score}+ skor yok")
            return
        
        summary = f"✅ *TARAMA TAMAM*\n\n📊 Top {len(top)}:\n\n"
        
        for _, row in top.iterrows():
            emoji = "🔥" if row['score'] >= 75 else "⭐"
            summary += f"{emoji} *{row['symbol']}* - {row['score']:.0f}/100\n"
            summary += f"   💰 {row['price']:.2f} TL ({row['change_%']:+.1f}%)\n"
            summary += f"   📊 Hacim: {row['volume_ratio']:.1f}x | RSI: {row['rsi']:.0f}\n\n"
        
        self.telegram.send_message(summary)


def main():
    """Ana fonksiyon"""
    print("\n" + "="*70)
    print("BIST HİSSE TARAMA - GitHub Actions Optimized")
    print("="*70 + "\n")
    
    # Telegram kontrolü
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("⚠️  Telegram credentials not found - continuing without notifications")
        telegram = None
    else:
        telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)
        logger.info("✅ Telegram connected")
    
    # Tarama başlat
    scanner = BISTScanner(telegram_notifier=telegram)
    scanner.scan_all()
    
    if len(scanner.df) > 0:
        print("\n📊 EN İYİ 10:")
        print("-"*70)
        print(scanner.df.head(10)[['symbol', 'price', 'change_%', 'rsi', 'score']].to_string(index=False))
        
        # CSV kaydet
        scanner.df.to_csv('bist_scan_results.csv', index=False)
        logger.info("📁 Results saved to bist_scan_results.csv")
        
        scanner.send_report(top_n=10, min_score=35)
        
        if telegram:
            telegram.send_message("✅ *TARAMA TAMAMLANDI!*")
    else:
        logger.error("❌ Tarama başarısız - veri alınamadı!")
        if telegram:
            telegram.send_message("❌ Veri çekilemedi - tüm denemeler başarısız")
    
    print("\n✅ İşlem tamamlandı!")


if __name__ == "__main__":
    main()
