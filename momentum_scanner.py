"""
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
    
    def send_report(self, top_n=10):
        """Basitleştirilmiş rapor"""
        if not self.telegram or len(self.df) == 0:
            return
        
        # Sadece AL sinyali olanları göster
        buy_signals = self.df[self.df['score'] >= 60].head(top_n)
        
        if len(buy_signals) == 0:
            self.telegram.send_message("⚠️ Bugün güçlü sinyal yok")
            return
        
        # Özet
        summary = f"🎯 *MOMENTUM AVCISI RAPORU*\n\n"
        summary += f"📊 {len(buy_signals)} güçlü sinyal bulundu!\n\n"
        
        for _, row in buy_signals.iterrows():
            emoji = row['signal_type'].split()[0]
            risk_emoji = "🟢" if row['risk_level'] == 'Düşük' else "🟡" if row['risk_level'] == 'Orta' else "🔴"
            
            summary += f"{emoji} *{row['symbol']}* - {row['score']:.0f}/100 {risk_emoji}\n"
            summary += f"   💰 {row['price']:.2f} TL ({row['change_%']:+.1f}%)\n"
            summary += f"   📊 Hacim: {row['volume_ratio']:.1f}x | RSI: {row['rsi']:.0f}\n\n"
        
        self.telegram.send_message(summary)
        
        # Top 3 detay
        for _, row in buy_signals.head(3).iterrows():
            detail = f"📋 *{row['symbol']}* - {row['signal_type']}\n"
            detail += f"━━━━━━━━━━━━━━\n"
            detail += f"Skor: *{row['score']:.0f}/100* | Risk: {row['risk_level']}\n\n"
            detail += f"💰 Fiyat: {row['price']:.2f} TL ({row['change_%']:+.2f}%)\n\n"
            detail += f"*4 Temel Gösterge:*\n"
            detail += f"📊 Hacim: {row['volume_ratio']:.1f}x\n"
            detail += f"⚡ RSI: {row['rsi']:.0f}\n"
            detail += f"🎯 MACD: {row['macd_signal']}"
            if row['macd_cross']:
                detail += " 🚀 YENİ AL!\n"
            else:
                detail += "\n"
            
            bb_text = 'Squeeze ⚡' if row['bb_squeeze'] else f"%{row['bb_position']:.0f}"
            detail += f"🔸 BB: {bb_text}\n\n"
            
            detail += f"*Sinyaller:*\n"
            for s in row['signals'][:5]:
                detail += f"• {s}\n"
            
            self.telegram.send_message(detail)
    
    def create_chart(self):
        """Basit gösterge grafikleri"""
        if len(self.df) == 0 or not self.telegram:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('🎯 Momentum Avcısı - 4 Temel Gösterge', fontsize=16, fontweight='bold')
            
            top10 = self.df.head(10)
            
            # 1. Hacim analizi
            ax1 = axes[0, 0]
            colors = ['#e74c3c' if v < 1.5 else '#f39c12' if v < 2.5 else '#2ecc71' for v in top10['volume_ratio']]
            ax1.barh(top10['symbol'], top10['volume_ratio'], color=colors)
            ax1.axvline(2, color='orange', linestyle='--', alpha=0.5, label='2x eşik')
            ax1.set_title('📊 Hacim Oranı (En Önemli!)', fontweight='bold')
            ax1.set_xlabel('Hacim Çarpanı')
            ax1.legend()
            ax1.invert_yaxis()
            
            # 2. RSI
            ax2 = axes[0, 1]
            ax2.scatter(top10['rsi'], top10['score'], s=150, alpha=0.6, c=top10['score'], cmap='RdYlGn')
            ax2.axvline(30, color='g', linestyle='--', alpha=0.5)
            ax2.axvline(70, color='r', linestyle='--', alpha=0.5)
            ax2.axvspan(40, 60, alpha=0.1, color='green', label='İdeal bölge')
            ax2.set_title('⚡ RSI Momentum', fontweight='bold')
            ax2.set_xlabel('RSI')
            ax2.set_ylabel('Skor')
            ax2.legend()
            
            # 3. Skor dağılımı
            ax3 = axes[1, 0]
            signal_colors = {'🔥 GÜÇLÜ AL': '#2ecc71', '✅ AL': '#3498db', '👀 İZLE': '#f39c12', '⏸️ BEKLE': '#95a5a6'}
            for sig_type, color in signal_colors.items():
                mask = top10['signal_type'] == sig_type
                if mask.any():
                    ax3.barh(top10[mask]['symbol'], top10[mask]['score'], color=color, label=sig_type)
            ax3.set_title('🎯 Skor ve Sinyal Tipi', fontweight='bold')
            ax3.set_xlabel('Skor')
            ax3.legend()
            ax3.invert_yaxis()
            
            # 4. Bollinger Position
            ax4 = axes[1, 1]
            squeeze_mask = top10['bb_squeeze']
            colors = ['#e74c3c' if sq else '#95a5a6' for sq in squeeze_mask]
            ax4.scatter(top10['bb_position'], top10['volume_ratio'], s=150, c=colors, alpha=0.6)
            ax4.axvline(20, color='g', linestyle='--', alpha=0.5, label='Alt bölge')
            ax4.axvline(80, color='r', linestyle='--', alpha=0.5, label='Üst bölge')
            ax4.set_title('🔸 Bollinger Position vs Hacim', fontweight='bold')
            ax4.set_xlabel('BB Position %')
            ax4.set_ylabel('Hacim Oranı')
            ax4.legend()
            
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
    print("🎯 MOMENTUM AVCISI - Basit & Etkili Strateji")
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
        print("\n🔥 EN İYİ 10:")
        print("-"*70)
        print(scanner.df.head(10)[['symbol', 'signal_type', 'score', 'volume_ratio', 'rsi']].to_string(index=False))
        
        scanner.df.to_csv('momentum_scan.csv', index=False)
        logger.info("📁 Sonuçlar kaydedildi")
        
        scanner.send_report(top_n=10)
        scanner.create_chart()
        
        if telegram:
            telegram.send_message("✅ *TARAMA TAMAM!*")
    else:
        logger.error("❌ Veri yok")
    
    print("\n✅ Bitti!")


if __name__ == "__main__":
    main()
