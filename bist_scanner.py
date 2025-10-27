"""
BIST HİSSE TARAMA SİSTEMİ - GITHUB ACTIONS
Otomatik çalışan, Telegram bildirimi gönderen versiyon
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import requests
import io
import os
warnings.filterwarnings('ignore')

# Telegram bilgilerini environment variable'dan al
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

class TelegramNotifier:
    """Telegram Bildirim Sınıfı"""
    
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, text, parse_mode='Markdown'):
        """Telegram'a mesaj gönder"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            response = requests.post(url, data=data, timeout=10)
            return response.json()
        except Exception as e:
            print(f"Telegram mesaj hatası: {e}")
            return None
    
    def send_photo(self, image_buffer, caption=''):
        """Telegram'a fotoğraf gönder"""
        try:
            url = f"{self.base_url}/sendPhoto"
            image_buffer.seek(0)
            files = {'photo': image_buffer}
            data = {
                'chat_id': self.chat_id,
                'caption': caption
            }
            response = requests.post(url, files=files, data=data, timeout=30)
            return response.json()
        except Exception as e:
            print(f"Telegram fotoğraf hatası: {e}")
            return None


class BISTScanner:
    """BIST Hisse Tarama Sınıfı"""
    
    def __init__(self, telegram_notifier=None):
        self.symbols = [
            'CRDFA.IS', 'HDGS.IS', 'ATSYH.IS', 'BURCE.IS', 'PAPIL.IS',
            'DUNYH.IS', 'DOBUR.IS', 'BURVA.IS', 'CEOEM.IS', 'UFUK.IS',
            'EGECY.IS', 'TEKTU.IS', 'LYDHO.IS', 'IZINV.IS', 'MAGEN.IS',
            'BULGS.IS', 'PCILT.IS', 'DOGUB.IS', 'DMSAS.IS', 'BMEKS.IS',
            'RODRG.IS', 'SANFM.IS', 'PNSUT.IS', 'YAPRK.IS', 'EMKEL.IS',
            'BALAT.IS', 'OSTIM.IS', 'RALYH.IS', 'SNPAM.IS', 'MRSHL.IS',
            'ULAS.IS', 'MNDRS.IS', 'TMPOL.IS', 'DZGYO.IS', 'KSTUR.IS'
        ]
        
        self.results = []
        self.telegram = telegram_notifier
        
    def get_stock_data(self, symbol, period='6mo'):
        """Hisse verisini çek"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if len(df) > 50:
                return df
            return None
        except Exception as e:
            print(f"Veri çekme hatası ({symbol}): {e}")
            return None
    
    def calculate_rsi(self, df, period=14):
        """RSI hesapla"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df):
        """MACD hesapla"""
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, df, period=20):
        """Bollinger Bands hesapla"""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        bandwidth = (upper - lower) / sma * 100
        return upper, lower, bandwidth
    
    def analyze_stock(self, symbol):
        """Hisseyi teknik olarak analiz et"""
        df = self.get_stock_data(symbol)
        
        if df is None or len(df) < 50:
            return None
        
        try:
            # Temel bilgiler
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # Hacim analizi
            avg_volume_20 = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
            
            # RSI
            rsi = self.calculate_rsi(df).iloc[-1]
            
            # MACD
            macd, signal, histogram = self.calculate_macd(df)
            macd_value = macd.iloc[-1]
            signal_value = signal.iloc[-1]
            macd_cross = macd_value > signal_value
            
            # Hareketli Ortalamalar
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            
            above_ma20 = current_price > ma20
            above_ma50 = current_price > ma50
            golden_cross = ma20 > ma50
            
            # Bollinger Bands
            upper, lower, bandwidth = self.calculate_bollinger_bands(df)
            bb_width = bandwidth.iloc[-1]
            bb_squeeze = bb_width < bandwidth.rolling(50).mean().iloc[-1]
            
            # 52 haftalık analiz
            week52_low = df['Close'].rolling(252).min().iloc[-1]
            distance_from_low = ((current_price - week52_low) / week52_low) * 100
            
            # Taban tespiti
            last_50_std = df['Close'].iloc[-50:].std()
            last_50_mean = df['Close'].iloc[-50:].mean()
            consolidation = (last_50_std / last_50_mean) < 0.10 if last_50_mean > 0 else False
            
            # SKOR HESAPLAMA
            score = 0
            reasons = []
            
            if volume_ratio > 3:
                score += 20
                reasons.append(f"💥 Hacim patlaması: {volume_ratio:.1f}x")
            elif volume_ratio > 2:
                score += 15
                reasons.append(f"📊 Hacim artışı: {volume_ratio:.1f}x")
            elif volume_ratio > 1.5:
                score += 10
                reasons.append(f"📈 Orta hacim artışı: {volume_ratio:.1f}x")
            
            if 40 < rsi < 70:
                score += 15
                reasons.append(f"✅ İyi RSI: {rsi:.1f}")
            elif 30 < rsi < 80:
                score += 10
                reasons.append(f"⚡ Kabul edilebilir RSI: {rsi:.1f}")
            
            if macd_cross:
                score += 15
                reasons.append("🚀 MACD AL sinyali")
            
            if above_ma20 and above_ma50:
                score += 20
                reasons.append("📊 MA20 ve MA50 üzerinde")
            elif above_ma20:
                score += 10
                reasons.append("📈 MA20 üzerinde")
            
            if golden_cross:
                score += 5
                reasons.append("⭐ Golden Cross")
            
            if bb_squeeze:
                score += 10
                reasons.append("🎯 Bollinger daralması")
            
            if consolidation and distance_from_low > 20:
                score += 15
                reasons.append("🔥 Taban bölgesinden çıkış")
            elif distance_from_low > 30:
                score += 10
                reasons.append("📍 Düşük seviyelerden uzaklaşma")
            
            if price_change > 5:
                score += 5
                reasons.append(f"💹 Güçlü günlük artış: %{price_change:.1f}")
            
            return {
                'symbol': symbol.replace('.IS', ''),
                'price': current_price,
                'change_%': price_change,
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'macd_signal': 'AL' if macd_cross else 'BEKLE',
                'above_ma20': above_ma20,
                'above_ma50': above_ma50,
                'bb_squeeze': bb_squeeze,
                'distance_from_52w_low_%': distance_from_low,
                'score': score,
                'reasons': reasons
            }
        except Exception as e:
            print(f"Analiz hatası ({symbol}): {e}")
            return None
    
    def scan_all_stocks(self):
        """Tüm hisseleri tara"""
        print(f"🔍 BIST Tarama Başlatıldı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Taranacak hisse sayısı: {len(self.symbols)}\n")
        
        if self.telegram:
            start_msg = f"🔍 *BIST HİSSE TARAMASI BAŞLADI*\n\n"
            start_msg += f"📊 Taranacak Hisse: {len(self.symbols)}\n"
            start_msg += f"⏰ Başlangıç: {datetime.now().strftime('%H:%M:%S')}\n"
            start_msg += f"📅 Tarih: {datetime.now().strftime('%d.%m.%Y')}"
            self.telegram.send_message(start_msg)
        
        for i, symbol in enumerate(self.symbols):
            result = self.analyze_stock(symbol)
            if result:
                self.results.append(result)
                print(f"[{i+1}/{len(self.symbols)}] {symbol} ✓ Skor: {result['score']:.0f}")
            else:
                print(f"[{i+1}/{len(self.symbols)}] {symbol} ✗")
        
        self.df_results = pd.DataFrame(self.results)
        self.df_results = self.df_results.sort_values('score', ascending=False)
        
        print(f"\n✅ Tarama tamamlandı! Toplam analiz: {len(self.df_results)}")
        return self.df_results
    
    def send_telegram_report(self, top_n=10, min_score=50):
        """Telegram'a rapor gönder"""
        if not self.telegram:
            return
        
        top_candidates = self.df_results[self.df_results['score'] >= min_score].head(top_n)
        
        if len(top_candidates) == 0:
            msg = "❌ *Kriterlere Uygun Hisse Bulunamadı*\n\n"
            msg += f"Minimum skor: {min_score}\n"
            msg += f"En yüksek skor: {self.df_results['score'].max():.0f}"
            self.telegram.send_message(msg)
            return
        
        # Özet
        summary = f"✅ *TARAMA TAMAMLANDI*\n\n"
        summary += f"📊 Taranan: {len(self.df_results)} | Uygun: {len(top_candidates)}\n"
        summary += f"⏰ {datetime.now().strftime('%H:%M:%S')}\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━\n\n"
        summary += f"🏆 *TOP {len(top_candidates)} ADAY:*\n\n"
        
        for idx, row in top_candidates.iterrows():
            emoji = "🔥" if row['score'] >= 80 else "⭐" if row['score'] >= 65 else "✨"
            summary += f"{emoji} *{row['symbol']}* - {row['score']:.0f}/100\n"
            summary += f"   💰 {row['price']:.2f} TL ({row['change_%']:+.1f}%)\n"
            summary += f"   📊 Hacim: {row['volume_ratio']:.1f}x | RSI: {row['rsi']:.0f}\n\n"
        
        self.telegram.send_message(summary)
        
        # Detaylı analiz (sadece en iyi 3)
        for idx, row in top_candidates.head(3).iterrows():
            detail = f"📋 *{row['symbol']}* - Skor: {row['score']:.0f}/100\n\n"
            detail += f"💰 Fiyat: {row['price']:.2f} TL ({row['change_%']:+.1f}%)\n"
            detail += f"📊 Hacim: {row['volume_ratio']:.1f}x | RSI: {row['rsi']:.0f}\n"
            detail += f"🎯 MACD: {row['macd_signal']}\n\n"
            detail += f"*Güçlü Yanları:*\n"
            for reason in row['reasons'][:5]:
                detail += f"• {reason}\n"
            self.telegram.send_message(detail)
    
    def create_and_send_chart(self):
        """Grafik oluştur ve gönder"""
        if len(self.df_results) == 0 or not self.telegram:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('BIST Hisse Tarama Analizi', fontsize=16, fontweight='bold')
            
            # 1. Top 10 Skor
            top_10 = self.df_results.head(10)
            colors = ['#2ecc71' if s >= 70 else '#f39c12' if s >= 50 else '#e74c3c' 
                      for s in top_10['score']]
            axes[0, 0].barh(top_10['symbol'], top_10['score'], color=colors)
            axes[0, 0].set_xlabel('Skor')
            axes[0, 0].set_title('En Yüksek Skorlu 10 Hisse')
            axes[0, 0].invert_yaxis()
            axes[0, 0].grid(axis='x', alpha=0.3)
            
            # 2. RSI Dağılımı
            axes[0, 1].scatter(self.df_results['rsi'], self.df_results['score'], 
                              alpha=0.6, s=80)
            axes[0, 1].axvline(x=40, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].axvline(x=70, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('RSI')
            axes[0, 1].set_ylabel('Skor')
            axes[0, 1].set_title('RSI vs Skor')
            axes[0, 1].grid(alpha=0.3)
            
            # 3. Hacim Analizi
            axes[1, 0].scatter(self.df_results['volume_ratio'], self.df_results['score'], 
                              alpha=0.6, s=80, c=self.df_results['change_%'], cmap='RdYlGn')
            axes[1, 0].set_xlabel('Hacim Oranı (x)')
            axes[1, 0].set_ylabel('Skor')
            axes[1, 0].set_title('Hacim vs Skor')
            axes[1, 0].grid(alpha=0.3)
            
            # 4. Skor Dağılımı
            axes[1, 1].hist(self.df_results['score'], bins=15, 
                           color='#3498db', edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Skor')
            axes[1, 1].set_ylabel('Hisse Sayısı')
            axes[1, 1].set_title('Skor Dağılımı')
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # BytesIO'ya kaydet
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            
            caption = f"📊 {datetime.now().strftime('%d.%m.%Y %H:%M')}"
            self.telegram.send_photo(buf, caption=caption)
            
            plt.close()
            buf.close()
            
            print("📈 Grafik Telegram'a gönderildi")
        except Exception as e:
            print(f"Grafik hatası: {e}")


def main():
    """Ana fonksiyon"""
    print("="*70)
    print("BIST HİSSE TARAMA SİSTEMİ - GITHUB ACTIONS")
    print("="*70)
    
    # Telegram kontrol
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ HATA: Telegram bilgileri bulunamadı!")
        print("GitHub Secrets ayarlarını kontrol edin.")
        return
    
    telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)
    scanner = BISTScanner(telegram_notifier=telegram)
    
    # Tarama
    scanner.scan_all_stocks()
    
    # Raporlar
    scanner.send_telegram_report(top_n=10, min_score=40)
    scanner.create_and_send_chart()
    
    # Bitiş mesajı
    if telegram:
        final = "✅ *TARAMA TAMAMLANDI*\n\n"
        final += f"⏰ {datetime.now().strftime('%H:%M:%S')}\n"
        final += "Raporlar gönderildi! 📊"
        telegram.send_message(final)
    
    print("\n✅ Tüm işlemler tamamlandı!")


if __name__ == "__main__":
    main()