"""
BIST HİSSE TARAMA - PROFESYONEL VERSİYON
Tüm teknik göstergeler dahil - GitHub için optimize edilmiş
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
import os
import logging

# Ayarlar
yf.pdr_override()
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

# Telegram
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '8310745808:AAGpnfSna6-6AJ5I2FNKyES2Rdj_Xqu4b7o')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '1801093830')

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
        # Doğrulanmış çalışan BIST hisseleri
        self.symbols = [
            'AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'YKBNK.IS', 'HALKB.IS',
            'KCHOL.IS', 'SAHOL.IS', 'DOHOL.IS',
            'THYAO.IS', 'TUPRS.IS', 'PETKM.IS', 'ASELS.IS',
            'ARCLK.IS', 'VESTL.IS', 'TOASO.IS', 'FROTO.IS',
            'TTKOM.IS', 'TCELL.IS',
            'SISE.IS', 'EREGL.IS', 'KRDMD.IS', 'SODA.IS', 'SASA.IS',
            'BIMAS.IS', 'ENKAI.IS', 'TTRAK.IS', 'PGSUS.IS', 
            'TAVHL.IS', 'KOZAL.IS',
        ]
        
        self.results = []
        self.telegram = telegram_notifier
        self.failed = []
        
    def get_stock_data(self, symbol):
        """Yahoo Finance'tan veri çek"""
        try:
            time.sleep(0.5)
            stock = yf.Ticker(symbol)
            for period in ['3mo', '6mo', '1y']:
                try:
                    df = stock.history(period=period)
                    if df is not None and len(df) >= 60:
                        return df
                    time.sleep(0.3)
                except:
                    continue
            return None
        except:
            return None
    
    def calculate_rsi(self, df, period=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
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
        bandwidth = ((upper_band - lower_band) / sma) * 100
        percent_b = (df['Close'] - lower_band) / (upper_band - lower_band)
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
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()
        return k_percent, d_percent
    
    def calculate_obv(self, df):
        return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    def calculate_vwap(self, df):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    def calculate_ichimoku(self, df):
        nine_period_high = df['High'].rolling(window=9).max()
        nine_period_low = df['Low'].rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) / 2
        
        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        return tenkan_sen, kijun_sen
    
    def analyze_stock(self, symbol):
        df = self.get_stock_data(symbol)
        
        if df is None or len(df) < 60:
            return None
        
        try:
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            avg_volume_20 = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
            
            rsi = self.calculate_rsi(df).iloc[-1]
            macd, signal, histogram = self.calculate_macd(df)
            macd_value = macd.iloc[-1]
            signal_value = signal.iloc[-1]
            histogram_value = histogram.iloc[-1]
            macd_cross = macd_value > signal_value
            macd_bullish = histogram_value > 0 and histogram.iloc[-2] <= 0
            
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            ma200 = df['Close'].rolling(min(len(df), 200)).mean().iloc[-1]
            
            above_ma20 = current_price > ma20
            above_ma50 = current_price > ma50
            above_ma200 = current_price > ma200
            golden_cross = ma20 > ma50 and ma50 > ma200
            
            upper_bb, lower_bb, bb_bandwidth, bb_percent = self.calculate_bollinger_bands(df)
            current_bb_bandwidth = bb_bandwidth.iloc[-1]
            avg_bb_bandwidth = bb_bandwidth.rolling(50).mean().iloc[-1]
            bb_squeeze = current_bb_bandwidth < avg_bb_bandwidth * 0.8
            bb_percent_value = bb_percent.iloc[-1]
            bb_near_lower = bb_percent_value < 0.2
            bb_near_upper = bb_percent_value > 0.8
            
            atr = self.calculate_atr(df).iloc[-1]
            atr_percent = (atr / current_price) * 100
            
            stoch_k, stoch_d = self.calculate_stochastic(df)
            stoch_k_value = stoch_k.iloc[-1]
            stoch_d_value = stoch_d.iloc[-1]
            stoch_oversold = stoch_k_value < 20
            stoch_overbought = stoch_k_value > 80
            stoch_bullish_cross = stoch_k_value > stoch_d_value and stoch_k.iloc[-2] <= stoch_d.iloc[-2]
            
            obv = self.calculate_obv(df)
            obv_ma = obv.rolling(20).mean()
            obv_rising = obv.iloc[-1] > obv_ma.iloc[-1]
            
            vwap = self.calculate_vwap(df).iloc[-1]
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
        except:
            return None
    
    def scan_all(self):
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
            
            if (i + 1) % 3 == 0:
                time.sleep(2)
            else:
                time.sleep(0.8)
        
        if len(self.results) == 0:
            self.df = pd.DataFrame()
            return self.df
        
        self.df = pd.DataFrame(self.results)
        self.df = self.df.sort_values('score', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"✅ BAŞARILI: {len(self.results)} | ❌ BAŞARISIZ: {len(self.failed)}")
        
        risk_counts = self.df['risk_level'].value_counts()
        print(f"\n📊 RİSK DAĞILIMI:")
        for risk, count in risk_counts.items():
            print(f"   {risk}: {count} hisse")
        
        print(f"{'='*70}\n")
        
        return self.df
    
    def send_report(self, top_n=10, min_score=40):
        if not self.telegram or len(self.df) == 0:
            return
        
        top = self.df[self.df['score'] >= min_score].head(top_n)
        
        if len(top) == 0:
            msg = f"⚠️ {min_score}+ skor yok\nEn yüksek: {self.df['score'].max():.0f}"
            self.telegram.send_message(msg)
            return
        
        summary = f"✅ *PRO TARAMA TAMAM*\n\n"
        summary += f"📊 Başarılı: {len(self.df)} | Top: {len(top)}\n"
        
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
        if len(self.df) == 0 or not self.telegram:
            return
        
        try:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            fig.suptitle('BIST Profesyonel Tarama Analizi', fontsize=18, fontweight='bold')
            
            ax1 = fig.add_subplot(gs[0, :2])
            top = self.df.head(10)
            colors = ['#2ecc71' if s >= 80 else '#f39c12' if s >= 60 else '#e74c3c' for s in top['score']]
            ax1.barh(top['symbol'], top['score'], color=colors)
            ax1.set_title('Top 10 Skor', fontweight='bold')
            ax1.set_xlabel('Skor')
            ax1.invert_yaxis()
            ax1.grid(alpha=0.3)
            
            ax2 = fig.add_subplot(gs[0, 2])
            risk_counts = self.df['risk_level'].value_counts()
            colors_risk = ['#2ecc71', '#f39c12', '#e74c3c']
            ax2.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.0f%%',
                   colors=colors_risk, startangle=90)
            ax2.set_title('Risk Dağılımı', fontweight='bold')
            
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.scatter(self.df['rsi'], self.df['score'], s=100, alpha=0.6,
                       c=self.df['score'], cmap='RdYlGn')
            ax3.axvline(30, color='r', linestyle='--', alpha=0.5)
            ax3.axvline(70, color='r', linestyle='--', alpha=0.5)
            ax3.set_xlabel('RSI')
            ax3.set_ylabel('Skor')
            ax3.set_title('RSI vs Skor', fontweight='bold')
            ax3.grid(alpha=0.3)
            
            ax4 = fig.add_subplot(gs[1, 1])
            scatter = ax4.scatter(self.df['volume_ratio'], self.df['score'], 
                                 s=100, c=self.df['change_%'], cmap='RdYlGn', alpha=0.6)
            ax4.set_xlabel('Hacim Oranı')
            ax4.set_ylabel('Skor')
            ax4.set_title('Hacim vs Skor', fontweight='bold')
            ax4.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Değişim %')
            
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.scatter(self.df['stoch_k'], self.df['score'], s=100, alpha=0.6)
            ax5.axvline(20, color='g', linestyle='--', alpha=0.5)
            ax5.axvline(80, color='r', linestyle='--', alpha=0.5)
            ax5.set_xlabel('Stochastic K')
            ax5.set_ylabel('Skor')
            ax5.set_title('Stochastic vs Skor', fontweight='bold')
            ax5.grid(alpha=0.3)
            
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
        except:
            pass


def main():
    print("\n" + "="*70)
    print("BIST HİSSE TARAMA - PROFESYONEL VERSİYON")
    print("="*70 + "\n")
    
    telegram = TelegramNotifier(BOT_TOKEN, CHAT_ID)
    scanner = BISTScanner(telegram_notifier=telegram)
    
    scanner.scan_all()
    
    if len(scanner.df) > 0:
        print("\n📊 EN İYİ 10:")
        print("-"*70)
        print(scanner.df.head(10)[['symbol', 'price', 'change_%', 'rsi', 'score']].to_string(index=False))
        
        scanner.send_report(top_n=15, min_score=30)
        scanner.create_chart()
        
        telegram.send_message("✅ *TARAMA TAMAMLANDI!*")
    else:
        print("\n❌ Tarama başarısız!")
        telegram.send_message("❌ Veri çekilemedi")
    
    print("\n✅ Tamamlandı!")


if __name__ == "__main__":
    main()
