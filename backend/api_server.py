#!/usr/bin/env python3
"""
知識蒸留APIサーバー - 企業評判分析版 v4.0
株式会社Luupのシェアライドサービスに関する多角的分析
各エージェントが異なるデータソースを専門的に扱い、トーナメントで知識を統合
"""
import os
import re
import sys
import time
import glob
import json
import random
import logging
import hashlib
import asyncio
import datetime
import pdfplumber
import unicodedata
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# FastAPI関連
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# LLM API imports
import openai
import anthropic
import google.generativeai as genai
import requests  # For OpenRouter

# Tavily for web search
from tavily import TavilyClient

# For text analysis
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# NLTK downloads
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed
random.seed(42)
np.random.seed(42)

# Load environment variables
load_dotenv()

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message="Could get FontBBox")
warnings.filterwarnings("ignore", category=UserWarning)

# プロジェクトのパスを設定
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")

# デバッグ情報を出力
logger.info(f"Backend directory: {BACKEND_DIR}")
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"PDF directory: {PDF_DIR}")
logger.info(f"PDF directory exists: {os.path.exists(PDF_DIR)}")

# PDFディレクトリが存在しない場合は作成
if not os.path.exists(PDF_DIR):
    try:
        os.makedirs(PDF_DIR, exist_ok=True)
        logger.info(f"Created PDF directory: {PDF_DIR}")
    except Exception as e:
        logger.error(f"Failed to create PDF directory: {e}")

# ====================
# Enums
# ====================
class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    UNKNOWN = "unknown"

class PoliticalOrientation(Enum):
    LIBERAL = "liberal"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    PROGRESSIVE = "progressive"
    LIBERTARIAN = "libertarian"
    UNKNOWN = "unknown"

class UserAttitude(Enum):
    """ユーザーのシェアライドサービスへの態度"""
    POSITIVE = "positive"  # サービス支持者
    NEGATIVE = "negative"  # サービス批判者
    NEUTRAL = "neutral"    # 中立的
    CONCERNED = "concerned"  # 懸念を持つが利用者
    UNKNOWN = "unknown"

class StakeholderType(Enum):
    """ステークホルダーの種類"""
    USER = "user"               # サービス利用者
    NON_USER = "non_user"       # 非利用者
    LOCAL_RESIDENT = "local_resident"  # 地域住民
    DRIVER = "driver"           # ドライバー
    PEDESTRIAN = "pedestrian"   # 歩行者
    BUSINESS_OWNER = "business_owner"  # 店舗経営者
    POLICY_MAKER = "policy_maker"  # 政策立案者

class ReputationCluster(Enum):
    """評判クラスターの種類（PDFレポートに基づく）"""
    CM_CAMPAIGN = "cm_campaign"  # 二宮和也のCM出演
    SIGNAL_VIOLATION = "signal_violation"  # 信号無視
    BATHHOUSE_EXPERIENCE = "bathhouse_experience"  # 銭湯とリラックス体験
    FIRE_SAFETY_VIOLATION = "fire_safety_violation"  # マンション避難器具設置と消防法違反
    ANTI_SENTIMENT = "anti_sentiment"  # 反感と倒産要求
    CEO_CRITICISM = "ceo_criticism"  # 岡井大輝への批判
    PORT_LOCATION = "port_location"  # ポート設置場所問題

class InfluencerType(Enum):
    """インフルエンサーのタイプ"""
    ANTI_LUUP = "anti_luup"  # 反Luup教
    NEUTRAL_CRITIC = "neutral_critic"  # 中立的批判者
    VIRAL_AMPLIFIER = "viral_amplifier"  # バイラル拡散者
    NEWS_MEDIA = "news_media"  # ニュースメディア

class TreatmentCondition(Enum):
    GENERIC = "generic"
    PERSONALIZATION = "personalization"
    COMMUNITY_ALIGNED = "community_aligned"

class DataSourceType(Enum):
    PDF = "pdf"
    WEB_SEARCH = "web_search"
    VIDEO_DATA = "video_data"
    CSV = "csv"  # 新規追加

# ====================
# Configuration Classes (改善案8: 設定の外部化)
# ====================
@dataclass
class TournamentConfig:
    """トーナメント設定"""
    agents_per_source: int = 10
    learning_enabled: bool = True
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        'base': 1.0,
        'integration': 2.0,
        'learning': 0.5,
        'depth': 1.5
    })
    max_learning_depth: int = 3
    parallel_processing: bool = True
    max_retries: int = 3
    retry_delay: float = 2.0
    cache_enabled: bool = True
    memory_compression: bool = True
    max_history_size: int = 5

# ====================
# Data Classes
# ====================
@dataclass
class DetailedScore:
    """詳細なスコア情報（改善案5: 評価の透明性向上）"""
    total: float
    breakdown: Dict[str, float]
    winning_factors: List[str]
    missing_elements: List[str]
    tournament_round: int = 0

@dataclass
class UserProfile:
    """Inferred user attributes from posting history"""
    username: str
    user_attitude: UserAttitude
    gender: Gender  # 追加
    age_range: Tuple[int, int]
    ethnicity: Optional[str]
    location: Optional[str]
    political_orientation: PoliticalOrientation  # 追加
    stakeholder_type: StakeholderType 
    interests: List[str]
    writing_style: Dict[str, float]

@dataclass
class RedditPost:
    """Represents a Reddit post"""
    post_id: str
    subreddit: str
    title: str
    body: str
    author: str
    timestamp: datetime
    score: int

@dataclass
class PersuasiveResponse:
    """Generated persuasive response"""
    content: str
    treatment_condition: TreatmentCondition
    persuasion_score: float
    model_used: str
    generation_params: Dict
    user_profile: Optional[UserProfile] = None
    pdf_context: Optional[str] = None
    learning_history: List[Dict] = field(default_factory=list)
    data_source_type: Optional[DataSourceType] = None
    source_specific_content: Optional[Dict[str, Any]] = None
    tournament_round: int = 0
    detailed_score: Optional[DetailedScore] = None

# ====================
# Learning Cache (改善案4: 学習効率の改善)
# ====================
class LearningCache:
    """学習パターンをキャッシュして再利用"""
    def __init__(self):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get_integration_pattern(self, winner_source: DataSourceType, 
                               loser_source: DataSourceType) -> Optional[Dict]:
        key = f"{winner_source.value}_{loser_source.value}"
        pattern = self.cache.get(key)
        if pattern:
            self.hit_count += 1
            logger.debug(f"Cache hit for {key} (hit rate: {self.get_hit_rate():.2%})")
        else:
            self.miss_count += 1
        return pattern
    
    def store_integration_pattern(self, winner_source: DataSourceType,
                                 loser_source: DataSourceType, 
                                 pattern: Dict):
        key = f"{winner_source.value}_{loser_source.value}"
        self.cache[key] = pattern
        logger.debug(f"Stored integration pattern for {key}")
    
    def get_hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

# ====================
# API Configuration
# ====================
class APIConfig:
    def __init__(self):
        # OpenAI
        self.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        ) if os.getenv('OPENAI_API_KEY') else None
        
        # Anthropic
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        ) if os.getenv('ANTHROPIC_API_KEY') else None
        
        # Google Gemini
        if os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        else:
            self.gemini_model = None
        
        # OpenRouter
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        # Tavily Search API
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key) if self.tavily_api_key else None

        # 要約用の設定を追加
        self.summarization_model = "o3-2025-04-16"
        self.max_prompt_length = 50000

        # Log API status
        logger.info("API Keys Status:")
        logger.info(f"OpenAI: {'✓' if self.openai_client else '✗'}")
        logger.info(f"Anthropic: {'✓' if self.anthropic_client else '✗'}")
        logger.info(f"Gemini: {'✓' if self.gemini_model else '✗'}")
        logger.info(f"OpenRouter: {'✓' if self.openrouter_api_key else '✗'}")
        logger.info(f"Tavily: {'✓' if self.tavily_client else '✗'}")

# ====================
# PDF Text Cleaner
# ====================
class PDFTextCleaner:
    """PDFから抽出したテキストをクリーンアップするクラス"""
    
    def __init__(self):
        self.noise_patterns = [
            r'←この[^→\n]*への',
            r'ページ\s*\d+',
            r'^\d+$',
            r'✔',
            r'\[.*?\]「.*?」',
            r'Copyright.*\d{4}',
            r'All Rights Reserved',
        ]
        
        # Luup分析レポート特有のパターン
        self.report_patterns = {
            'cluster_headers': r'\d+_[^のクラスター]+のクラスター',
            'user_mentions': r'@[a-zA-Z0-9_]+',
            'urls': r'https?://[^\s]+',
            'impression_data': r'impression[数を]?\d+',
            'date_ranges': r'\d{4}-\d{2}(?:-\d{2})?',
        }
        
    def clean_text(self, text: str) -> str:
        """PDFテキストをクリーンアップ"""
        text = text.replace('　', ' ')
        text = self._fix_split_words(text)
        
        for pattern in self.noise_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        text = self._merge_short_lines(text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = self._remove_headers_footers(text)
        text = self._normalize_punctuation(text)
        
        return text.strip()
    
    def _fix_split_words(self, text: str) -> str:
        pattern = r'([ぁ-んァ-ヶー一-龥々])\n([ぁ-んァ-ヶー一-龥々])'
        text = re.sub(pattern, r'\1\2', text)
        pattern = r'([がのにをはでとも])\n'
        text = re.sub(pattern, r'\1', text)
        return text
    
    def _merge_short_lines(self, text: str) -> str:
        lines = text.split('\n')
        merged_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                merged_lines.append('')
                i += 1
                continue
            if len(line) <= 3 and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line:
                    merged_lines.append(line + next_line)
                    i += 2
                    continue
            merged_lines.append(line)
            i += 1
        return '\n'.join(merged_lines)
    
    def _remove_headers_footers(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if re.match(r'^[-\s]*\d+[-\s]*$', line.strip()):
                continue
            if re.match(r'^\d+/\d+$', line.strip()):
                continue
            if re.match(r'^第\d+ページ$', line.strip()):
                continue
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    def _normalize_punctuation(self, text: str) -> str:
        text = text.replace('､', '、')
        text = text.replace('｡', '。')
        text = text.replace('･', '・')
        return text
    
    def extract_critical_users(self, text: str) -> List[Dict[str, Any]]:
        """重要なユーザーアカウントを抽出"""
        critical_users = []
        
        # PDFレポートで特定された要注意アカウント
        anti_luup_accounts = [
            '@chinniisan',  # 「ちんにい」- 反Luup教の中心人物
            '@stopluup',
            '@onshanow',  # 「人間の鑑」
            '@hamusoku',
            '@ikeda11123',
            '@mrnobuyuki'
        ]
        
        for account in anti_luup_accounts:
            if account in text:
                critical_users.append({
                    'account': account,
                    'type': 'anti_luup',
                    'risk_level': 'high' if account in ['@chinniisan', '@stopluup'] else 'medium'
                })
        
        return critical_users

    def extract_risk_clusters(self, text: str) -> List[Dict[str, Any]]:
        """リスククラスターを抽出"""
        risk_clusters = []
        
        # 最大の争点となるクラスター
        if 'マンションの避難器具設置と消防法違反' in text:
            risk_clusters.append({
                'type': ReputationCluster.FIRE_SAFETY_VIOLATION,
                'risk_level': 'critical',
                'description': '避難器具設置問題は最大の炎上リスク',
                'timeline': '10月にバースト発生'
            })
        
        if '信号無視' in text:
            risk_clusters.append({
                'type': ReputationCluster.SIGNAL_VIOLATION,
                'risk_level': 'high',
                'description': '利用者マナーに関する批判の中心'
            })
        
        return risk_clusters
    
# ====================
# Video Data Analyzer
# ====================
class VideoDataAnalyzer:
    """YouTube/TikTok動画データを詳細分析するクラス（Luup関連）"""
        
    def __init__(self):
        self.json_cache = {}
        
        # PDFレポートから抽出したクラスターパターン
        self.cluster_patterns = {
            'cm_campaign': ['二宮和也', 'CM', '新CM', 'Luup新CM'],
            'signal_violation': ['信号無視', '赤信号', '違反', '危険運転'],
            'bathhouse': ['銭湯', 'サウナ', '#Luupで銭湯'],
            'fire_safety': ['消防法', '避難器具', 'マンション', '違法'],
            'anti_sentiment': ['反Luup', '倒産', '廃止', '消えて']
        }
        
        # バーストリスクの評価基準
        self.burst_risk_keywords = {
            'critical': ['消防法違反', '避難器具', '違法設置'],
            'high': ['事故', '怪我', '転倒', '衝突'],
            'medium': ['マナー', '迷惑', '邪魔', '放置']
        }
        
        # センチメント分析用のキーワード（Luup向け）
        self.sentiment_keywords = {
            'positive': [
                '便利', 'エコ', '環境に優しい', '快適', '楽しい', '効率的', 
                '革新的', 'スマート', '未来', '素晴らしい', 'おすすめ', 
                '時短', '移動が楽', '気軽', 'サステナブル'
            ],
            'negative': [
                '危険', '事故', 'マナー悪い', '邪魔', '迷惑', '違反', '怖い',
                '歩道', '逆走', '信号無視', 'ぶつかる', '転倒', '怪我',
                '放置', '乗り捨て', '騒音', '暴走'
            ],
            'neutral': [
                'ルール', '規制', '法律', '対策', '改善', '検討', '議論',
                '報告', '調査', '分析', '統計', 'データ'
            ]
        }
        
        # Luup関連キーワード
        self.luup_keywords = {
            'service': ['Luup', 'ループ', 'シェアライド', '電動キックボード', 
                    'キックボード', 'マイクロモビリティ', 'ラストワンマイル'],
            'issues': ['事故', 'マナー', '違反', '放置', '歩道走行', '信号無視',
                    '飲酒運転', 'ヘルメット', '二人乗り', '危険運転'],
            'stakeholders': ['利用者', 'ユーザー', '歩行者', 'ドライバー', '警察',
                        '自治体', '住民', '商店街', '駅前'],
            'solutions': ['安全講習', 'ルール', '規制', '取り締まり', '啓発',
                        'マナー向上', '専用レーン', 'ポート', '駐輪場']
        }

    def analyze_cluster_distribution(self, videos: List[Dict]) -> Dict[str, Any]:
        """動画のクラスター分布を分析"""
        cluster_distribution = {cluster: [] for cluster in self.cluster_patterns.keys()}
        
        for video in videos:
            title = video.get('title', '').lower()
            description = video.get('description', '').lower()
            text = title + ' ' + description
            
            for cluster_name, keywords in self.cluster_patterns.items():
                if any(keyword.lower() in text for keyword in keywords):
                    cluster_distribution[cluster_name].append({
                        'title': video.get('title', ''),
                        'views': video.get('view_count', 0),
                        'channel': video.get('channel', ''),
                        'id': video.get('id', '')
                    })
        
        # 各クラスターのリスク評価
        cluster_risks = {}
        for cluster_name, videos_in_cluster in cluster_distribution.items():
            total_views = sum(v['views'] for v in videos_in_cluster)
            
            risk_level = 'low'
            if cluster_name == 'fire_safety' and total_views > 100000:
                risk_level = 'critical'
            elif cluster_name in ['signal_violation', 'anti_sentiment'] and total_views > 50000:
                risk_level = 'high'
            elif total_views > 10000:
                risk_level = 'medium'
            
            cluster_risks[cluster_name] = {
                'video_count': len(videos_in_cluster),
                'total_views': total_views,
                'risk_level': risk_level,
                'top_videos': sorted(videos_in_cluster, key=lambda x: x['views'], reverse=True)[:3]
            }
        
        return cluster_risks
    
    def load_json_data(self, json_path: str) -> List[Dict]:
        """JSONファイルを読み込んで動画データのリストを返す"""
        if json_path in self.json_cache:
            return self.json_cache[json_path]
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # データの正規化
            if isinstance(data, dict):
                # videos キーがある場合
                if 'videos' in data:
                    data = data['videos']
                # data キーがある場合
                elif 'data' in data:
                    data = data['data']
                else:
                    # 辞書を配列に変換
                    data = [data]
            
            self.json_cache[json_path] = data
            logger.info(f"Successfully loaded JSON with {len(data)} videos")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            return []
    
    def analyze_sentiment(self, videos: List[Dict]) -> Dict[str, Any]:
        """動画のセンチメント分析を実行"""
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'unknown': 0}
        sentiment_videos = {'positive': [], 'negative': [], 'neutral': [], 'unknown': []}
        sentiment_engagement = {'positive': {'views': 0, 'count': 0}, 
                               'negative': {'views': 0, 'count': 0},
                               'neutral': {'views': 0, 'count': 0}}
        
        for video in videos:
            title = video.get('title', '').lower()
            description = video.get('description', '').lower()
            text = title + ' ' + description
            
            # センチメント判定
            sentiment_scores = {
                'positive': sum(1 for kw in self.sentiment_keywords['positive'] if kw.lower() in text),
                'negative': sum(1 for kw in self.sentiment_keywords['negative'] if kw.lower() in text),
                'neutral': sum(1 for kw in self.sentiment_keywords['neutral'] if kw.lower() in text)
            }
            
            # 最も高いスコアのセンチメントを選択
            if max(sentiment_scores.values()) > 0:
                sentiment = max(sentiment_scores, key=sentiment_scores.get)
            else:
                sentiment = 'unknown'
            
            sentiment_counts[sentiment] += 1
            sentiment_videos[sentiment].append({
                'title': video.get('title', '')[:100],
                'channel': video.get('channel', ''),
                'views': video.get('view_count', 0),
                'id': video.get('id', '')
            })
            
            # エンゲージメント統計
            if sentiment != 'unknown':
                sentiment_engagement[sentiment]['views'] += video.get('view_count', 0)
                sentiment_engagement[sentiment]['count'] += 1
        
        # 平均視聴回数を計算
        for sent in sentiment_engagement:
            if sentiment_engagement[sent]['count'] > 0:
                sentiment_engagement[sent]['avg_views'] = (
                    sentiment_engagement[sent]['views'] / sentiment_engagement[sent]['count']
                )
            else:
                sentiment_engagement[sent]['avg_views'] = 0
        
        total_videos = len(videos)
        
        return {
            'distribution': sentiment_counts,
            'percentages': {
                k: (v / total_videos * 100) if total_videos > 0 else 0 
                for k, v in sentiment_counts.items()
            },
            'engagement_stats': sentiment_engagement,
            'top_videos_by_sentiment': {
                k: sorted(v, key=lambda x: x['views'], reverse=True)[:3]
                for k, v in sentiment_videos.items()
            }
        }
    
    def identify_influencers(self, videos: List[Dict], top_n: int = 10) -> List[Dict]:
        """影響力のあるチャンネルを特定"""
        channel_stats = {}
        
        for video in videos:
            channel = video.get('channel', 'Unknown')
            
            if channel not in channel_stats:
                channel_stats[channel] = {
                    'channel_name': channel,
                    'channel_id': video.get('channel_id', ''),
                    'follower_count': video.get('channel_follower_count', 0),
                    'total_views': 0,
                    'video_count': 0,
                    'videos': [],
                    'avg_views': 0
                }
            
            channel_stats[channel]['total_views'] += video.get('view_count', 0)
            channel_stats[channel]['video_count'] += 1
            channel_stats[channel]['videos'].append({
                'title': video.get('title', '')[:80],
                'views': video.get('view_count', 0),
                'id': video.get('id', '')
            })
        
        # 影響力スコアを計算
        for channel in channel_stats.values():
            channel['avg_views'] = (
                channel['total_views'] / channel['video_count'] 
                if channel['video_count'] > 0 else 0
            )
            
            # 影響力スコア = フォロワー数 * 0.3 + 平均視聴回数 * 0.5 + 総視聴回数 * 0.2
            channel['influence_score'] = (
                channel['follower_count'] * 0.3 +
                channel['avg_views'] * 0.5 +
                channel['total_views'] * 0.0002
            )
            
            # 最も人気の動画を特定
            channel['top_video'] = max(channel['videos'], key=lambda x: x['views']) if channel['videos'] else None
        
        # 影響力スコアでソート
        sorted_channels = sorted(channel_stats.values(), 
                                key=lambda x: x['influence_score'], 
                                reverse=True)
        
        return sorted_channels[:top_n]
    
    def analyze_temporal_trends(self, videos: List[Dict]) -> Dict[str, Any]:
        """時系列トレンドを分析"""
        if not videos:
            return {}
        
        # タイムスタンプをdatetimeに変換
        for video in videos:
            if 'timestamp' in video:
                video['datetime'] = datetime.fromtimestamp(video['timestamp'])
        
        # 日付でソート
        videos_with_date = [v for v in videos if 'datetime' in v]
        if not videos_with_date:
            return {}
        
        videos_sorted = sorted(videos_with_date, key=lambda x: x['datetime'])
        
        # 日別の統計を計算
        daily_stats = {}
        for video in videos_sorted:
            date_key = video['datetime'].date()
            
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    'date': date_key.strftime('%Y-%m-%d'),
                    'video_count': 0,
                    'total_views': 0,
                    'channels': set(),
                    'keywords': []
                }
            
            daily_stats[date_key]['video_count'] += 1
            daily_stats[date_key]['total_views'] += video.get('view_count', 0)
            daily_stats[date_key]['channels'].add(video.get('channel', ''))
            daily_stats[date_key]['keywords'].extend(video.get('matched_keywords', []))
        
        # setをリストに変換
        for date_key in daily_stats:
            daily_stats[date_key]['channels'] = list(daily_stats[date_key]['channels'])
            daily_stats[date_key]['unique_channels'] = len(daily_stats[date_key]['channels'])
        
        # 統計のリストを作成
        daily_list = list(daily_stats.values())
        
        # トレンド分析
        if len(daily_list) > 1:
            first_week_views = sum(d['total_views'] for d in daily_list[:7])
            last_week_views = sum(d['total_views'] for d in daily_list[-7:])
            trend_direction = 'increasing' if last_week_views > first_week_views else 'decreasing'
        else:
            trend_direction = 'insufficient_data'
        
        # ピーク日を特定
        peak_day = max(daily_list, key=lambda x: x['total_views']) if daily_list else None
        
        return {
            'daily_stats': daily_list,
            'date_range': {
                'start': videos_sorted[0]['datetime'].strftime('%Y-%m-%d'),
                'end': videos_sorted[-1]['datetime'].strftime('%Y-%m-%d')
            },
            'peak_day': peak_day,
            'trend_direction': trend_direction,
            'total_period_views': sum(video.get('view_count', 0) for video in videos),
            'avg_daily_videos': len(videos) / max(len(daily_list), 1)
        }
    
    def extract_key_topics(self, videos: List[Dict], max_topics: int = 15) -> Dict[str, Any]:
        """主要なトピックとキーワードを抽出"""
        # タイトルと説明文を結合
        all_text = []
        keyword_videos = {}
        hashtag_counter = Counter()
        
        for video in videos:
            text = video.get('title', '') + ' ' + video.get('description', '')
            all_text.append(text)
            
            # マッチしたキーワードを収集
            for keyword in video.get('matched_keywords', []):
                if keyword not in keyword_videos:
                    keyword_videos[keyword] = []
                keyword_videos[keyword].append({
                    'title': video.get('title', '')[:80],
                    'channel': video.get('channel', ''),
                    'views': video.get('view_count', 0)
                })
            
            # ハッシュタグを抽出
            hashtags = re.findall(r'#([^\s]+)', text)
            hashtag_counter.update(hashtags)
        
        # 政治関連トピックの分析
        political_topics = {
            'parties_mentioned': {},
            'politicians_mentioned': {},
            'topics_discussed': {}
        }
        
        full_text = ' '.join(all_text)
        
        # 政党の言及回数
        for party in self.political_keywords['parties']:
            count = full_text.count(party)
            if count > 0:
                political_topics['parties_mentioned'][party] = count
        
        # 政治家の言及回数
        for politician in self.political_keywords['politicians']:
            count = full_text.count(politician)
            if count > 0:
                political_topics['politicians_mentioned'][politician] = count
        
        # トピックの言及回数
        for topic in self.political_keywords['topics']:
            count = full_text.count(topic)
            if count > 0:
                political_topics['topics_discussed'][topic] = count
        
        # キーワード別の統計
        keyword_stats = []
        for keyword, vids in keyword_videos.items():
            total_views = sum(v['views'] for v in vids)
            avg_views = total_views / len(vids) if vids else 0
            
            keyword_stats.append({
                'keyword': keyword,
                'video_count': len(vids),
                'total_views': total_views,
                'avg_views': avg_views,
                'top_video': max(vids, key=lambda x: x['views']) if vids else None
            })
        
        # 視聴回数でソート
        keyword_stats.sort(key=lambda x: x['total_views'], reverse=True)
        
        return {
            'matched_keywords': keyword_stats[:max_topics],
            'top_hashtags': dict(hashtag_counter.most_common(10)),
            'political_analysis': political_topics,
            'trending_topics': self._identify_trending_topics(videos)
        }
    
    def _identify_trending_topics(self, videos: List[Dict]) -> List[str]:
        """トレンドトピックを特定"""
        recent_videos = sorted(videos, 
                             key=lambda x: x.get('timestamp', 0), 
                             reverse=True)[:10]
        
        trending = []
        for video in recent_videos:
            # タイトルから重要そうな部分を抽出
            title = video.get('title', '')
            # 【】内のテキストを抽出
            brackets = re.findall(r'【([^】]+)】', title)
            trending.extend(brackets)
        
        # 重複を除いて頻度順にソート
        trending_counter = Counter(trending)
        return [topic for topic, _ in trending_counter.most_common(5)]
    
    def calculate_engagement_metrics(self, videos: List[Dict]) -> Dict[str, Any]:
        """エンゲージメント指標を計算"""
        if not videos:
            return {}
        
        view_counts = [v.get('view_count', 0) for v in videos]
        follower_counts = [v.get('channel_follower_count', 0) for v in videos]
        durations = [v.get('duration', 0) for v in videos]
        
        # 基本統計
        metrics = {
            'total_videos': len(videos),
            'total_views': sum(view_counts),
            'avg_views': sum(view_counts) / len(view_counts) if view_counts else 0,
            'median_views': sorted(view_counts)[len(view_counts)//2] if view_counts else 0,
            'max_views': max(view_counts) if view_counts else 0,
            'min_views': min(view_counts) if view_counts else 0,
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'total_reach': sum(follower_counts)
        }
        
        # パフォーマンス別の分類
        performance_tiers = {
            'viral': [],      # 10万再生以上
            'high': [],       # 5万〜10万再生
            'medium': [],     # 1万〜5万再生  
            'low': []         # 1万再生未満
        }
        
        for video in videos:
            views = video.get('view_count', 0)
            video_info = {
                'title': video.get('title', '')[:80],
                'channel': video.get('channel', ''),
                'views': views
            }
            
            if views >= 100000:
                performance_tiers['viral'].append(video_info)
            elif views >= 50000:
                performance_tiers['high'].append(video_info)
            elif views >= 10000:
                performance_tiers['medium'].append(video_info)
            else:
                performance_tiers['low'].append(video_info)
        
        metrics['performance_distribution'] = {
            tier: {
                'count': len(videos_list),
                'percentage': (len(videos_list) / len(videos) * 100) if videos else 0,
                'examples': sorted(videos_list, key=lambda x: x['views'], reverse=True)[:2]
            }
            for tier, videos_list in performance_tiers.items()
        }
        
        return metrics
    
    def generate_comprehensive_analysis(self, json_path: str, query: str = "") -> Dict[str, Any]:
        """動画データの包括的な分析を生成"""
        videos = self.load_json_data(json_path)
        
        if not videos:
            return {'error': 'Failed to load video data'}
        
        analysis = {
            'summary': {
                'total_videos': len(videos),
                'total_views': sum(v.get('view_count', 0) for v in videos),
                'unique_channels': len(set(v.get('channel', '') for v in videos)),
                'languages': list(set(v.get('language', 'unknown') for v in videos))
            },
            'sentiment_analysis': self.analyze_sentiment(videos),
            'influencer_analysis': self.identify_influencers(videos, top_n=5),
            'temporal_trends': self.analyze_temporal_trends(videos),
            'topic_analysis': self.extract_key_topics(videos),
            'engagement_metrics': self.calculate_engagement_metrics(videos),
            'platform_distribution': self._analyze_platform_distribution(videos)
        }
        
        # クエリに関連する動画を特定
        if query:
            analysis['query_relevant_videos'] = self._find_relevant_videos(videos, query)
        
        return analysis
    
    def _analyze_platform_distribution(self, videos: List[Dict]) -> Dict[str, Any]:
        """プラットフォーム別の分布を分析"""
        # YouTube Shortsかどうかを判定（aspect_ratio = 0.56 は縦型動画）
        shorts_count = sum(1 for v in videos if v.get('aspect_ratio', 0) < 1)
        regular_count = len(videos) - shorts_count
        
        return {
            'shorts': {
                'count': shorts_count,
                'percentage': (shorts_count / len(videos) * 100) if videos else 0
            },
            'regular': {
                'count': regular_count,
                'percentage': (regular_count / len(videos) * 100) if videos else 0
            }
        }
    
    def _find_relevant_videos(self, videos: List[Dict], query: str, max_results: int = 5) -> List[Dict]:
        """クエリに関連する動画を検索"""
        query_lower = query.lower()
        relevant = []
        
        for video in videos:
            title = video.get('title', '').lower()
            description = video.get('description', '').lower()
            
            # 関連性スコアを計算
            relevance_score = 0
            if query_lower in title:
                relevance_score += 2
            if query_lower in description:
                relevance_score += 1
            
            # キーワードマッチ
            for keyword in video.get('matched_keywords', []):
                if keyword.lower() in query_lower or query_lower in keyword.lower():
                    relevance_score += 1.5
            
            if relevance_score > 0:
                relevant.append({
                    'title': video.get('title', ''),
                    'channel': video.get('channel', ''),
                    'views': video.get('view_count', 0),
                    'relevance_score': relevance_score,
                    'id': video.get('id', ''),
                    'url': f"https://youtube.com/watch?v={video.get('id', '')}" if video.get('id') else ''
                })
        
        # 関連性スコアでソート
        relevant.sort(key=lambda x: (x['relevance_score'], x['views']), reverse=True)
        
        return relevant[:max_results]

# ====================
# Twitter CSV Analyzer
# ====================
class TwitterCSVAnalyzer:
    """X（Twitter）CSVデータを分析するクラス（Luup評判分析強化版）"""
    
    def __init__(self):
        self.csv_cache = {}
        self.sentiment_keywords = {
            'positive': ['便利', '良い', 'いい', '最高', '快適', '満足', 
                        'おすすめ', '楽しい', 'エコ', '環境'],
            'negative': ['危険', '最悪', '迷惑', '邪魔', '事故', 'マナー', 
                        '違反', '怖い', '不安', '問題']
        }
        
        # Luup特有の評判キーワード
        self.reputation_keywords = {
            'safety': ['事故', '安全', '危険', '怪我', 'ヘルメット'],
            'manner': ['マナー', '迷惑', '邪魔', '放置', '乗り捨て'],
            'regulation': ['規制', 'ルール', '法律', '違反', '取り締まり'],
            'convenience': ['便利', '快適', '時短', '効率', 'アクセス']
        }
        
        # PDFレポートから抽出した重要キーワード
        self.critical_keywords = {
            'anti_luup': ['#反Luup教', '倒産', '消えて', '廃止'],
            'safety_issues': ['事故', '信号無視', '危険', '迷惑', '転倒'],
            'port_issues': ['消防法違反', '避難器具', '水道メーター', 'ポート', '放置'],
            'authority_criticism': ['天下り', '元警視総監', '東大卒', '岡井']
        }
        
        # 要監視アカウント（PDFレポートより）
        self.watch_list = {
            'high_risk': ['chinniisan', 'stopluup', 'onshanow'],
            'medium_risk': ['hamusoku', 'ikeda11123', 'mrnobuyuki'],
            'influencers': ['livedoornews', 'item87177']
        }
    
    def analyze_anti_luup_movement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """反Luup教の活動を分析"""
        if df.empty:
            return {}
        
        anti_luup_data = {
            'hashtag_usage': 0,
            'key_accounts': [],
            'viral_posts': [],
            'engagement_total': 0
        }
        
        # #反Luup教のハッシュタグを含む投稿を分析
        if 'Opening Text' in df.columns:
            anti_luup_posts = df[df['Opening Text'].str.contains('#反Luup教|反Luup', na=False)]
            anti_luup_data['hashtag_usage'] = len(anti_luup_posts)
            
            if not anti_luup_posts.empty and 'Engagement' in anti_luup_posts.columns:
                anti_luup_data['engagement_total'] = anti_luup_posts['Engagement'].sum()
                
                # バイラル投稿を特定
                viral_threshold = anti_luup_posts['Engagement'].quantile(0.9)
                viral_posts = anti_luup_posts[anti_luup_posts['Engagement'] > viral_threshold]
                
                for _, post in viral_posts.iterrows():
                    anti_luup_data['viral_posts'].append({
                        'text': post.get('Opening Text', '')[:200],
                        'engagement': post.get('Engagement', 0),
                        'author': post.get('Author Handle', '')
                    })
        
        return anti_luup_data
    
    def identify_risk_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """リスクパターンを特定（PDFレポートの知見を活用）"""
        patterns = {
            'fire_safety_burst': False,  # 消防法違反関連のバースト
            'authority_criticism': False,  # 権威批判（天下り、東大卒）
            'port_complaints': False,  # ポート設置場所への不満
            'signal_violation': False  # 信号無視関連
        }
        
        if 'Opening Text' in df.columns:
            text_combined = ' '.join(df['Opening Text'].dropna().astype(str))
            
            # 各パターンの検出
            if '消防法' in text_combined or '避難器具' in text_combined:
                patterns['fire_safety_burst'] = True
            
            if '天下り' in text_combined or '元警視総監' in text_combined:
                patterns['authority_criticism'] = True
            
            if 'ポート' in text_combined and ('邪魔' in text_combined or '迷惑' in text_combined):
                patterns['port_complaints'] = True
            
            if '信号無視' in text_combined or '赤信号' in text_combined:
                patterns['signal_violation'] = True
        
        return patterns
    
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """CSVファイルを読み込んで DataFrame として返す"""
        if csv_path in self.csv_cache:
            return self.csv_cache[csv_path]
        
        try:
            # UTF-16LEエンコーディングで読み込み
            df = pd.read_csv(csv_path, encoding='utf-16le', sep='\t')
            
            # 日付をdatetime型に変換
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # 数値型に変換
            numeric_columns = ['Likes', 'Reposts', 'Replies', 'Quotes', 'Engagement', 
                              'Views', 'Reach', 'Global Reach', 'National Reach', 'Local Reach']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            self.csv_cache[csv_path] = df
            logger.info(f"Successfully loaded CSV with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """センチメント分布を分析"""
        if 'Sentiment' not in df.columns:
            return {}
        
        sentiment_counts = df['Sentiment'].value_counts().to_dict()
        total_posts = len(df)
        
        # センチメント別のエンゲージメント統計
        sentiment_engagement = {}
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_df = df[df['Sentiment'] == sentiment]
            if not sentiment_df.empty:
                sentiment_engagement[sentiment] = {
                    'count': len(sentiment_df),
                    'percentage': (len(sentiment_df) / total_posts * 100) if total_posts > 0 else 0,
                    'avg_likes': sentiment_df['Likes'].mean() if 'Likes' in sentiment_df.columns else 0,
                    'avg_reposts': sentiment_df['Reposts'].mean() if 'Reposts' in sentiment_df.columns else 0,
                    'avg_engagement': sentiment_df['Engagement'].mean() if 'Engagement' in sentiment_df.columns else 0,
                    'total_reach': sentiment_df['Reach'].sum() if 'Reach' in sentiment_df.columns else 0
                }
        
        return {
            'distribution': sentiment_counts,
            'total_posts': total_posts,
            'by_sentiment': sentiment_engagement
        }
    
    def identify_influencers(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
        """影響力のあるアカウントを特定"""
        if df.empty:
            return []
        
        # エンゲージメントでグループ化
        influencer_stats = df.groupby(['Author Name', 'Author Handle']).agg({
            'Likes': 'sum',
            'Reposts': 'sum', 
            'Engagement': 'sum',
            'Reach': 'sum' if 'Reach' in df.columns else 'count',
            'Date': 'count'  # 投稿数
        }).reset_index()
        
        influencer_stats.columns = ['author_name', 'author_handle', 'total_likes', 
                                   'total_reposts', 'total_engagement', 'total_reach', 'post_count']
        
        # 影響力スコアを計算
        influencer_stats['influence_score'] = (
            influencer_stats['total_likes'] * 1.0 +
            influencer_stats['total_reposts'] * 2.0 +  # リポストは拡散力が高いので重み付け
            influencer_stats['total_reach'] * 0.001  # リーチ数も考慮
        )
        
        # トップインフルエンサーを選出
        top_influencers = influencer_stats.nlargest(top_n, 'influence_score').to_dict('records')
        
        # 各インフルエンサーの代表的な投稿を追加
        for influencer in top_influencers:
            author_posts = df[df['Author Name'] == influencer['author_name']]
            if not author_posts.empty:
                # 最もエンゲージメントの高い投稿を取得
                top_post = author_posts.nlargest(1, 'Engagement').iloc[0]
                influencer['top_post'] = {
                    'text': top_post.get('Opening Text', ''),
                    'likes': top_post.get('Likes', 0),
                    'reposts': top_post.get('Reposts', 0),
                    'sentiment': top_post.get('Sentiment', 'unknown'),
                    'date': top_post.get('Date', '').strftime('%Y-%m-%d') if pd.notna(top_post.get('Date')) else ''
                }
        
        return top_influencers
    
    def analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """時系列トレンドを分析"""
        if 'Date' not in df.columns or df.empty:
            return {}
        
        # 日付でソート
        df_sorted = df.sort_values('Date')
        
        # 日別の投稿数とエンゲージメント
        daily_stats = df_sorted.groupby(df_sorted['Date'].dt.date).agg({
            'Date': 'count',
            'Likes': 'sum',
            'Reposts': 'sum',
            'Engagement': 'sum',
            'Sentiment': lambda x: x.value_counts().to_dict() if 'Sentiment' in df.columns else {}
        }).reset_index()
        
        daily_stats.columns = ['date', 'post_count', 'total_likes', 'total_reposts', 
                               'total_engagement', 'sentiment_dist']
        
        # トレンド情報
        trends = {
            'daily_stats': daily_stats.to_dict('records'),
            'peak_day': {
                'date': daily_stats.loc[daily_stats['total_engagement'].idxmax(), 'date'].strftime('%Y-%m-%d'),
                'engagement': daily_stats['total_engagement'].max()
            } if not daily_stats.empty else {},
            'average_daily_posts': daily_stats['post_count'].mean(),
            'trend_direction': 'increasing' if len(daily_stats) > 1 and 
                             daily_stats.iloc[-1]['total_engagement'] > daily_stats.iloc[0]['total_engagement'] 
                             else 'decreasing'
        }
        
        return trends
    
    def extract_key_topics(self, df: pd.DataFrame, max_topics: int = 10) -> List[Dict]:
        """主要なトピックやキーフレーズを抽出"""
        if 'Opening Text' not in df.columns:
            return []
        
        # テキストから頻出単語を抽出（簡易版）
        import re
        from collections import Counter
        
        all_text = ' '.join(df['Opening Text'].dropna().astype(str))
        
        # 日本語の単語を抽出（簡易的な方法）
        japanese_words = re.findall(r'[ぁ-んァ-ヶー一-龥々]+', all_text)
        
        # ストップワードを除外
        stopwords = ['です', 'ます', 'した', 'こと', 'もの', 'これ', 'それ', 'あれ', 
                    'ここ', 'そこ', 'どこ', 'いる', 'ある', 'する', 'なる']
        
        filtered_words = [w for w in japanese_words if len(w) > 1 and w not in stopwords]
        
        # 単語の頻度をカウント
        word_counts = Counter(filtered_words)
        
        # トピックごとのセンチメント分析
        topics = []
        for word, count in word_counts.most_common(max_topics):
            # その単語を含む投稿のセンチメント分布
            word_posts = df[df['Opening Text'].str.contains(word, na=False)]
            if not word_posts.empty and 'Sentiment' in word_posts.columns:
                sentiment_dist = word_posts['Sentiment'].value_counts().to_dict()
                avg_engagement = word_posts['Engagement'].mean() if 'Engagement' in word_posts.columns else 0
            else:
                sentiment_dist = {}
                avg_engagement = 0
            
            topics.append({
                'keyword': word,
                'frequency': count,
                'sentiment_distribution': sentiment_dist,
                'avg_engagement': avg_engagement,
                'post_count': len(word_posts)
            })
        
        return topics
    
    def generate_analysis_summary(self, csv_path: str, query: str = "") -> Dict[str, Any]:
        """CSV分析の総合サマリーを生成"""
        df = self.load_csv_data(csv_path)
        
        if df.empty:
            return {'error': 'Failed to load CSV data'}
        
        summary = {
            'total_posts': len(df),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d') if 'Date' in df.columns and not df['Date'].isna().all() else 'unknown',
                'end': df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns and not df['Date'].isna().all() else 'unknown'
            },
            'sentiment_analysis': self.analyze_sentiment_distribution(df),
            'top_influencers': self.identify_influencers(df, top_n=5),
            'temporal_trends': self.analyze_temporal_trends(df),
            'key_topics': self.extract_key_topics(df, max_topics=5),
            'engagement_stats': {
                'total_likes': df['Likes'].sum() if 'Likes' in df.columns else 0,
                'total_reposts': df['Reposts'].sum() if 'Reposts' in df.columns else 0,
                'avg_engagement': df['Engagement'].mean() if 'Engagement' in df.columns else 0,
                'max_engagement_post': {
                    'text': df.loc[df['Engagement'].idxmax(), 'Opening Text'] if 'Engagement' in df.columns and not df.empty else '',
                    'engagement': df['Engagement'].max() if 'Engagement' in df.columns else 0,
                    'author': df.loc[df['Engagement'].idxmax(), 'Author Name'] if 'Engagement' in df.columns and not df.empty else ''
                }
            }
        }
        
        return summary

# ====================
# User Profiler
# ====================
class UserProfiler:
    """Analyzes user's posting history to infer demographic attributes"""
        
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
        self.attitude_indicators = {
            UserAttitude.POSITIVE: ['便利', '愛用', '素晴らしい', 'おすすめ', '快適'],
            UserAttitude.NEGATIVE: ['危険', '迷惑', '反対', '廃止すべき', '規制強化'],
            UserAttitude.CONCERNED: ['心配', '不安', '改善希望', '気をつけて'],
            UserAttitude.NEUTRAL: ['どちらとも', '一長一短', '場合による'],
            UserAttitude.UNKNOWN: []
        }
        
        self.stakeholder_indicators = {
            StakeholderType.USER: ['利用している', '乗っている', 'ユーザーです'],
            StakeholderType.DRIVER: ['運転中', 'ドライバーとして', '車から見て'],
            StakeholderType.PEDESTRIAN: ['歩いていたら', '歩行者として', '歩道で'],
            StakeholderType.LOCAL_RESIDENT: ['近所に住んで', '地域住民', '家の前']
        }
    
    def _infer_user_attitude(self, text: str) -> UserAttitude:
        """テキストからユーザーの態度を推論"""
        text_lower = text.lower()
        attitude_scores = {attitude: 0 for attitude in UserAttitude}
        
        for attitude, indicators in self.attitude_indicators.items():
            for indicator in indicators:
                attitude_scores[attitude] += text_lower.count(indicator)
        
        max_attitude = max(attitude_scores, key=attitude_scores.get)
        return max_attitude if attitude_scores[max_attitude] > 0 else UserAttitude.NEUTRAL

    def _infer_stakeholder_type(self, text: str) -> StakeholderType:
        """テキストからステークホルダータイプを推論"""
        text_lower = text.lower()
        stakeholder_scores = {stakeholder: 0 for stakeholder in StakeholderType}
        
        for stakeholder, indicators in self.stakeholder_indicators.items():
            for indicator in indicators:
                stakeholder_scores[stakeholder] += text_lower.count(indicator)
        
        max_stakeholder = max(stakeholder_scores, key=stakeholder_scores.get)
        return max_stakeholder if stakeholder_scores[max_stakeholder] > 0 else StakeholderType.NON_USER

    def analyze_posting_history(self, posts: List[RedditPost]) -> UserProfile:
        """Analyze user's posts to build profile"""
        if not posts:
            return self.create_political_researcher_profile()
        
        all_text = ' '.join([p.title + ' ' + p.body for p in posts])
        
        gender = self._infer_gender(all_text)
        age_range = self._infer_age_range(all_text)
        location = self._infer_location(all_text)
        political = self._infer_political_orientation(all_text)
        interests = self._extract_interests(posts)
        writing_style = self._analyze_writing_style(posts)

        user_attitude = self._infer_user_attitude(all_text)
        stakeholder_type = self._infer_stakeholder_type(all_text)
        

        return UserProfile(
            username=posts[0].author if posts else "unknown",
            user_attitude=user_attitude,  # 追加
            gender=gender,
            age_range=age_range,
            ethnicity=None,
            location=location,
            political_orientation=political,
            stakeholder_type=stakeholder_type,  # 追加
            interests=interests,
            writing_style=writing_style
        )
    
    def _infer_gender(self, text: str) -> Gender:
        text_lower = text.lower()
        gender_scores = {gender: 0 for gender in Gender}
        
        for gender, indicators in self.gender_indicators.items():
            for indicator in indicators:
                gender_scores[gender] += text_lower.count(indicator)
        
        max_gender = max(gender_scores, key=gender_scores.get)
        return max_gender if gender_scores[max_gender] > 0 else Gender.UNKNOWN
    
    def _infer_age_range(self, text: str) -> Tuple[int, int]:
        age_patterns = [
            r"I'm (\d{1,2}) years old",
            r"as a (\d{1,2}) year old",
            r"I'm (\d{1,2})\b"
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text)
            if match:
                age = int(match.group(1))
                return (age - 2, age + 2)
        
        return (25, 35)
    
    def _infer_location(self, text: str) -> Optional[str]:
        for pattern in self.location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _infer_political_orientation(self, text: str) -> PoliticalOrientation:
        text_lower = text.lower()
        political_scores = {orientation: 0 for orientation in PoliticalOrientation}
        
        for orientation, indicators in self.political_indicators.items():
            for indicator in indicators:
                political_scores[orientation] += text_lower.count(indicator)
        
        max_orientation = max(political_scores, key=political_scores.get)
        return max_orientation if political_scores[max_orientation] > 0 else PoliticalOrientation.UNKNOWN
    
    def _extract_interests(self, posts: List[RedditPost]) -> List[str]:
        subreddit_counts = Counter([p.subreddit for p in posts])
        return [sub for sub, _ in subreddit_counts.most_common(5)]
    
    def _analyze_writing_style(self, posts: List[RedditPost]) -> Dict[str, float]:
        if not posts:
            return {}
        
        all_text = ' '.join([p.body for p in posts if p.body])
        
        if not all_text:
            return {}
        
        sentences = [s.strip() for s in all_text.split('.') if s.strip()]
        
        style = {
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'flesch_reading_ease': textstat.flesch_reading_ease(all_text),
            'sentiment_compound': self.sia.polarity_scores(all_text)['compound'],
            'question_ratio': all_text.count('?') / max(len(all_text.split()), 1),
            'exclamation_ratio': all_text.count('!') / max(len(all_text.split()), 1)
        }
        
        return style
    
    def create_political_researcher_profile(self, username: str = "political_researcher_2025") -> UserProfile:
        """政治学研究者の固定プロフィールを生成"""
        return UserProfile(
            username=username,
            user_attitude=UserAttitude.NEUTRAL,  # 追加：中立的な態度
            gender=Gender.UNKNOWN,
            age_range=(35, 45),
            ethnicity=None,
            location="東京都",
            political_orientation=PoliticalOrientation.MODERATE,
            stakeholder_type=StakeholderType.POLICY_MAKER,  # 追加：政策立案者として
            interests=['政治学', '選挙分析', 'SNS政治', '世論調査', 'メディア研究'],
            writing_style={
                'avg_sentence_length': 25.0,
                'flesch_reading_ease': 40.0,
                'sentiment_compound': 0.0,
                'question_ratio': 0.05,
                'exclamation_ratio': 0.0
            }
        )
# ====================
# Tournament Selector with Learning
# ====================
class TournamentSelector:
    """Implements single-elimination tournament for response selection with enhanced cross-source learning"""
    
    def __init__(self, persuasion_optimizer=None, config: TournamentConfig = None):
        self.sia = SentimentIntensityAnalyzer()
        self.persuasion_optimizer = persuasion_optimizer
        self.config = config or TournamentConfig()
        self.learning_cache = LearningCache() if self.config.cache_enabled else None
        # 統合された知識を蓄積
        self.integrated_knowledge = {
            'pdf': [],
            'web_search': [],
            'video_data': []
        }
    
    def _refine_response_with_llm(self, content: str, winner_source: DataSourceType, 
                                loser_source: DataSourceType, loser_insights: List[str],
                                winner_response: PersuasiveResponse) -> str:
        """LLMを使用して応答文を洗練させる（具体的データ重視版）"""
        
        # データから固有名と統計情報を抽出
        data_points = []
        video_examples = []
        
        # 勝者のソース特有のデータを抽出（既存のコード維持）
        if winner_source == DataSourceType.PDF and winner_response.source_specific_content:
            # 既存のPDFデータ抽出コード...
            import re
            for key, value in winner_response.source_specific_content.items():
                if 'pdf' in key and isinstance(value, dict):
                    sections = value.get('sections', [])
                    for section in sections:
                        party_names = re.findall(r'(自民党|公明党|参政党|国民民主党|立憲民主党|日本維新の会)', section)
                        data_points.extend(list(set(party_names)))
                        numbers = re.findall(r'\d+[％%]|\d+議席|\d+票', section)
                        data_points.extend(numbers[:3])
        
        elif winner_source == DataSourceType.VIDEO_DATA and winner_response.source_specific_content:
            # 既存の動画データ抽出コード...
            analysis = winner_response.source_specific_content.get('video_analysis', {})
            if analysis and isinstance(analysis, dict):
                if 'total_views' in analysis:
                    data_points.append(f"総視聴回数{analysis['total_views']:,}回")
                if 'top_viewed_videos' in analysis:
                    for video in analysis['top_viewed_videos'][:3]:
                        if isinstance(video, dict):
                            video_examples.append({
                                'title': video.get('title', ''),
                                'channel': video.get('channel', ''),
                                'views': video.get('view_count', 0)
                            })
        
        elif winner_source == DataSourceType.WEB_SEARCH and winner_response.source_specific_content:
            # 既存のWeb検索データ抽出コード...
            for key, value in winner_response.source_specific_content.items():
                if 'search' in key and isinstance(value, list):
                    for result in value[:2]:
                        if isinstance(result, dict):
                            title = result.get('title', '')
                            if title:
                                data_points.append(f"「{title[:30]}...」")
        
        # 敗者の洞察をテキスト化（この部分が抜けていました）
        loser_insights_text = ""
        if loser_insights:
            loser_insights_text = "\\n".join([f"- {insight}" for insight in loser_insights[:3]])
        
        # 敗者の具体的データを抽出（新規追加）
        loser_concrete_examples = []
        if winner_response.source_specific_content:
            loser_data = winner_response.source_specific_content.get(f'learned_from_{loser_source.value}', {})
            if loser_data and 'insights' in loser_data:
                # メタデータから具体例を取得
                for insight in loser_data['insights']:
                    if '「' in insight and '」' in insight:
                        loser_concrete_examples.append(insight)
        
        # 動画例を文章化
        video_examples_text = ""
        if video_examples:
            video_examples_text = "\\n具体的に引用すべき動画例:\\n"
            for v in video_examples[:3]:
                video_examples_text += f"- 「{v['title'][:50]}」（{v['channel']}、{v['views']:,}回視聴）\\n"
        
        refinement_prompt = f"""
        以下の文章を、日本の新聞記事として自然で読みやすい文章に改善してください。
        特に第2段落で敗者のデータソースから学んだ具体的な情報を含めることが重要です。

        【あなたが利用可能なデータソース】
        {winner_source.value}のデータのみ

        【敗者から学んだ新しい視点】
        {loser_source.value}のデータから得られた以下の洞察を、
        あなたのデータソースの観点から解釈し、統合してください：
        {loser_insights_text}
        {f"具体例: {', '.join(loser_concrete_examples)}" if loser_concrete_examples else ""}

        【改善が必要な文章】
        {content}

        【必ず組み込むべきデータ】
        {', '.join(data_points[:10]) if data_points else 'なし'}
        {video_examples_text}

        【改善の指示】
        あなたは{winner_source.value}のデータソースのみにアクセスできるため、以下の構造で文章を構成してください：

        1. 第1段落：あなたのデータソース（{winner_source.value}）からの具体的な分析
        - 具体的なデータや数値を必ず含める
        - データソースの特性を活かした分析を行う

        2. 第2段落：敗者の視点との統合的考察
        - 敗者（{loser_source.value}）から学んだ洞察を、あなたのデータソースの観点から再解釈
        - 動画データの場合：具体的な動画タイトル、チャンネル名、視聴回数を言及
        - Web検索の場合：具体的な記事タイトル、メディア名、専門家の見解を言及
        - 両者の視点がどのように補完し合うかを説明

        3. 第3段落：結論と新たな視点の提示
        - 単一のデータソースでは見えなかった新しい発見
        - 統合によって得られたメタ的な視点や洞察
        - 参政党躍進の真の要因についての深い考察

        【重要な制約】
        - あなたは{winner_source.value}のデータにのみアクセスできます
        - 他のデータソースの詳細を勝手に創作してはいけません
        - 敗者から学んだ「視点」や「解釈の枠組み」は活用できますが、具体的なデータは引用できません
        - 400-600文字程度にまとめる
        - 同じ表現や結論を繰り返さない
        - 各段落は明確に区別し、自然な流れを保つ

        改善された文章のみを出力してください。
        """
        
        try:
            if self.persuasion_optimizer and hasattr(self.persuasion_optimizer, 'api_config'):
                api_config = self.persuasion_optimizer.api_config
                
                if api_config.openai_client:
                    response = api_config.openai_client.chat.completions.create(
                        model="gpt-4.1-2025-04-14",
                        messages=[
                            {"role": "system", "content": "あなたは日本の一流新聞社の編集者です。構造化された分析記事を作成してください。"},
                            {"role": "user", "content": refinement_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                    refined = response.choices[0].message.content
                    logger.info(f"Response refined with structured format: {len(content)} -> {len(refined)} chars")
                    return refined
        except Exception as e:
            logger.error(f"Failed to refine response: {e}")
        
        return content

    def _remove_duplicate_content(self, content: str) -> str:
        """重複する文章を削除"""
        sentences = content.split('。')
        seen_sentences = []
        unique_content = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 完全一致をチェック
            if sentence in seen_sentences:
                continue
            
            # 部分一致をチェック（80%以上の類似度）
            is_duplicate = False
            for seen in seen_sentences:
                if self._calculate_similarity(sentence, seen) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_content.append(sentence)
                seen_sentences.append(sentence)
        
        return '。'.join(unique_content) + '。' if unique_content else content

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """2つの文字列の類似度を計算"""
        if not str1 or not str2:
            return 0.0
        
        # 簡易的な文字列類似度計算
        set1 = set(str1)
        set2 = set(str2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

    def _validate_content_structure(self, content: str) -> bool:
        """コンテンツの構造が適切かチェック"""
        # 最小文字数チェック
        if len(content) < 300:
            return False
        
        # 段落数チェック
        paragraphs = content.split('\\n\\n')
        if len(paragraphs) < 2:
            return False
        
        # 各段落の長さチェック
        for para in paragraphs:
            if len(para.strip()) < 50:
                return False
        
        return True

    def _merge_all_source_contents(self, winner_content: Optional[Dict], 
                                loser_content: Optional[Dict],
                                winner_source: DataSourceType, 
                                loser_source: DataSourceType) -> Dict:
        """勝者と敗者の全てのソース固有コンテンツをマージ"""
        merged = winner_content.copy() if winner_content else {}
        
        # 敗者のデータを統合用として追加
        if loser_content:
            # 敗者の生データを保持（学習用）
            merged[f'integrated_from_{loser_source.value}'] = {
                'source_type': loser_source.value,
                'original_content': loser_content,
                'integration_timestamp': datetime.now().isoformat()
            }
            
            # 具体的なデータを抽出して保存
            if loser_source == DataSourceType.VIDEO_DATA and 'video_analysis' in loser_content:
                analysis = loser_content['video_analysis']
                merged[f'video_data_integration'] = {
                    'total_views': analysis.get('total_views', 0),
                    'total_videos': analysis.get('total_videos', 0),
                    'top_videos': analysis.get('top_viewed_videos', [])[:10]
                }
            
            elif loser_source == DataSourceType.PDF:
                merged[f'pdf_data_integration'] = {
                    'sections': [],
                    'parties_mentioned': [],
                    'policies': []
                }
                for key, value in loser_content.items():
                    if 'pdf' in key and isinstance(value, dict):
                        if 'sections' in value:
                            merged[f'pdf_data_integration']['sections'].extend(value['sections'])
            
            elif loser_source == DataSourceType.WEB_SEARCH:
                merged[f'web_search_integration'] = {
                    'articles': []
                }
                for key, value in loser_content.items():
                    if 'search' in key and isinstance(value, list):
                        merged[f'web_search_integration']['articles'].extend(value[:10])
        
        return merged

    def run_tournament_with_learning_tracking(self, responses: List[PersuasiveResponse], 
                                            enable_learning: bool = True,
                                            user_question: str = None) -> Tuple[PersuasiveResponse, List[Dict]]:
        """学習機能付きトーナメントを実行し、各ラウンドの結果を記録"""
        tournament_log = []
        
        # ユーザーの質問を保存（ここに挿入）
        if user_question:
            self._current_user_question = user_question
        
        # 詳細スコアを初期化
        for response in responses:
            response.detailed_score = self._score_response_with_details(response)
            response.persuasion_score = response.detailed_score.total
            response.tournament_round = 0
        
        initial_scores = [
            (r.model_used, r.persuasion_score, r.data_source_type.value if r.data_source_type else 'unknown') 
            for r in responses
        ]
        tournament_log.append({
            'round': 0,
            'type': 'initial',
            'participants': initial_scores
        })
        
        contestants = responses.copy()
        random.shuffle(contestants)
        
        round_num = 1
        while len(contestants) > 1:
            round_results = []
            next_round = []
            
            # ラウンド番号を設定
            for contestant in contestants:
                contestant.tournament_round = round_num
            
            for i in range(0, len(contestants), 2):
                if i + 1 < len(contestants):
                    contestant1 = contestants[i]
                    contestant2 = contestants[i + 1]
                    
                    if enable_learning:
                        winner = self._compare_responses_with_learning(contestant1, contestant2)
                    else:
                        winner = self._compare_responses(contestant1, contestant2)
                    
                    round_results.append({
                        'match': f"{contestant1.model_used} ({contestant1.data_source_type.value if contestant1.data_source_type else 'unknown'}) vs {contestant2.model_used} ({contestant2.data_source_type.value if contestant2.data_source_type else 'unknown'})",
                        'scores': f"{contestant1.persuasion_score:.2f} vs {contestant2.persuasion_score:.2f}",
                        'winner': f"{winner.model_used} ({winner.data_source_type.value if winner.data_source_type else 'unknown'})",
                        'winner_improved_score': f"{winner.persuasion_score:.2f}",
                        'learning_applied': enable_learning and hasattr(winner, 'learning_history') and len(winner.learning_history) > 0,
                        'score_details': winner.detailed_score.breakdown if winner.detailed_score else {}
                    })
                    
                    next_round.append(winner)
                else:
                    next_round.append(contestants[i])
                    round_results.append({
                        'match': f"{contestants[i].model_used} ({contestants[i].data_source_type.value if contestants[i].data_source_type else 'unknown'}) (bye)",
                        'scores': f"{contestants[i].persuasion_score:.2f}",
                        'winner': f"{contestants[i].model_used} ({contestants[i].data_source_type.value if contestants[i].data_source_type else 'unknown'})",
                        'learning_applied': False
                    })
            
            tournament_log.append({
                'round': round_num,
                'type': 'elimination',
                'matches': round_results
            })
            
            contestants = next_round
            round_num += 1
        
        final_winner = contestants[0]
        
        # 学習履歴を圧縮
        if self.config.memory_compression:
            final_winner.learning_history = self._compress_learning_history(final_winner.learning_history)
        
        # 学習履歴のサマリーを追加
        if hasattr(final_winner, 'learning_history') and final_winner.learning_history:
            successful_learnings = [h for h in final_winner.learning_history if h.get('status') in ['success', 'success_alternative']]
            failed_learnings = [h for h in final_winner.learning_history if h.get('status') == 'failed_no_improvement']
            
            tournament_log.append({
                'round': 'final',
                'type': 'learning_summary',
                'learning_history': final_winner.learning_history,
                'total_attempts': len(final_winner.learning_history),
                'successful_learnings': len(successful_learnings),
                'failed_learnings': len(failed_learnings),
                'total_score_improvement': sum(h.get('improved_score', 0) - h.get('original_score', 0) for h in successful_learnings),
                'cache_hit_rate': self.learning_cache.get_hit_rate() if self.learning_cache else 0.0
            })
        
        return final_winner, tournament_log
    
    def _compress_learning_history(self, history: List[Dict]) -> List[Dict]:
        """学習履歴を圧縮"""
        if len(history) <= self.config.max_history_size:
            return history
        
        # 最も重要な学習のみを保持
        sorted_history = sorted(
            history, 
            key=lambda x: x.get('improved_score', 0) - x.get('original_score', 0),
            reverse=True
        )
        return sorted_history[:self.config.max_history_size]
    
    def _score_response_with_details(self, response: PersuasiveResponse) -> DetailedScore:
        """詳細なスコアリング - PDFレポート知見反映版"""
        breakdown = {
            'base_quality': 0.0,
            'data_richness': 0.0,
            'source_coverage': 0.0,
            'integration_quality': 0.0,
            'solution_oriented': 0.0,
            'stakeholder_balance': 0.0,
            'reputation_insights': 0.0,
            'actionable_advice': 0.0,
            'risk_awareness': 0.0,  # リスク認識の深さ
            'cluster_understanding': 0.0,  # クラスター理解度
            'influencer_strategy': 0.0,  # インフルエンサー対策
            'timeline_awareness': 0.0,  # タイムライン認識
            'depth_bonus': 0.0,  # 追加
            'knowledge_synthesis': 0.0,  # 追加
            'cross_platform_synthesis': 0.0,  # 追加（これが不足していました）
            'comprehensive_coverage': 0.0,  # 追加
            'learning_bonus': 0.0,  # 追加
            # CSV分析関連（必要に応じて初期化）
            'csv_sentiment_quality': 0.0,
            'csv_influencer_depth': 0.0,
            'csv_temporal_analysis': 0.0,
            'csv_engagement_metrics': 0.0,
            # 動画分析関連（必要に応じて初期化）
            'video_sentiment_quality': 0.0,
            'video_influencer_depth': 0.0,
            'video_topic_richness': 0.0,
            'video_performance_analysis': 0.0,
        }

        content = response.content
        
        # ========================================
        # 1. 基本品質評価
        # ========================================
        
        # 基本的な文字数評価
        char_count = len(content)
        if 1800 <= char_count <= 2200:
            breakdown['base_quality'] += 1.0
        else:
            breakdown['base_quality'] += max(0, 1.0 - abs(char_count - 500) / 500)
        
        # 学術的な表現の評価
        academic_terms = ['分析', '考察', '要因', '背景', '傾向', '影響', '構造', '動向', 
                        '評判', '反応', '評価', '洞察', '示唆', '観点', '視点']
        academic_score = sum(1 for term in academic_terms if term in content)
        breakdown['base_quality'] += min(1.0, academic_score / 4)
        
        # データキーワードの評価
        data_keywords = ['データ', '統計', '調査', '%', '割合', '票', '議席', '視聴回数',
                        'いいね', 'リポスト', 'フォロワー', 'エンゲージメント', 'リーチ']
        data_score = sum(1 for keyword in data_keywords if keyword in content)
        breakdown['data_richness'] = min(2.0, data_score / 2)

        # 企業分析用語の評価
        business_terms = ['評判', 'イメージ', '戦略', 'ブランド', 'PR', '広報', 
                        'ステークホルダー', 'CSR', 'コミュニケーション', '信頼']
        business_score = sum(1 for term in business_terms if term in content)
        breakdown['base_quality'] += min(1.0, business_score / 4)
        
        # 解決策の評価
        solution_keywords = ['改善', '対策', '提案', '施策', 'すべき', '必要', 
                            'ポイント', '重要', '戦略', 'アプローチ']
        solution_score = sum(1 for keyword in solution_keywords if keyword in content)
        breakdown['solution_oriented'] = min(2.0, solution_score / 2)
        
        # ステークホルダーバランスの評価
        stakeholders_mentioned = 0
        for stakeholder in ['利用者', 'ユーザー', '歩行者', 'ドライバー', '住民', '自治体']:
            if stakeholder in content:
                stakeholders_mentioned += 1
        breakdown['stakeholder_balance'] = min(2.0, stakeholders_mentioned * 0.5)
        
        # 評判改善の洞察
        if any(term in content for term in ['SNS', 'メディア', '発信', 'コミュニケーション']):
            breakdown['reputation_insights'] += 1.0
        if any(term in content for term in ['炎上', '批判', 'ネガティブ', '改善']):
            breakdown['reputation_insights'] += 0.5
        
        # 実行可能なアドバイス
        if '具体的' in content or '実際に' in content:
            breakdown['actionable_advice'] += 1.0
        if any(term in content for term in ['べき', 'ポイント', '重要', '必要']):
            breakdown['actionable_advice'] += 0.5
        
        # ========================================
        # 2. データソースカバレッジ評価（4ソース対応）
        # ========================================
        
        # 各データソースの存在確認
        pdf_mentioned = self._check_pdf_content(content)
        web_mentioned = self._check_web_content(content)
        video_mentioned = self._check_video_content(content)
        csv_mentioned = self._check_csv_content(content)
        
        source_count = sum([pdf_mentioned, web_mentioned, video_mentioned, csv_mentioned])
        
        # 各データソースの深度をチェック
        pdf_depth = self._evaluate_source_depth(content, 'pdf')
        web_depth = self._evaluate_source_depth(content, 'web')
        video_depth = self._evaluate_source_depth(content, 'video')
        csv_depth = self._evaluate_source_depth(content, 'csv')
        
        # ソースカバレッジスコア（4つ全て含む場合に最高点）
        breakdown['source_coverage'] = source_count * 0.75
        if source_count == 4:
            breakdown['source_coverage'] += 3.0  # 完全統合ボーナス
        elif source_count == 3:
            breakdown['source_coverage'] += 1.5  # 部分統合ボーナス
        
        # 各ソースの深度を評価
        breakdown['depth_bonus'] = (pdf_depth + web_depth + video_depth + csv_depth) / 4
        
        # ========================================
        # 3. CSV（Twitter）分析特有の評価
        # ========================================
        
        if response.data_source_type == DataSourceType.CSV or csv_mentioned:
            source_specific = response.source_specific_content if hasattr(response, 'source_specific_content') else {}
            csv_analysis = source_specific.get('csv_analysis', {})
            
            # センチメント分析の品質評価
            if csv_analysis.get('sentiment_analysis'):
                sentiment = csv_analysis['sentiment_analysis']
                # センチメント分布の詳細度
                if sentiment.get('by_sentiment'):
                    breakdown['csv_sentiment_quality'] += 1.0
                    # 具体的な数値が含まれているか
                    for sent_type, data in sentiment['by_sentiment'].items():
                        if 'avg_likes' in data and 'avg_reposts' in data:
                            breakdown['csv_sentiment_quality'] += 0.3
                
                # コンテンツ内でセンチメント結果が言及されているか
                if 'ポジティブ' in content or 'ネガティブ' in content or 'センチメント' in content:
                    breakdown['csv_sentiment_quality'] += 0.5
            
            # インフルエンサー分析の深さ
            if csv_analysis.get('top_influencers'):
                influencers = csv_analysis['top_influencers']
                breakdown['csv_influencer_depth'] = min(2.0, len(influencers) * 0.3)
                
                # 具体的なインフルエンサー名が言及されているか
                for influencer in influencers[:3]:
                    if influencer.get('author_name', '') in content:
                        breakdown['csv_influencer_depth'] += 0.3
                
                # 影響力の数値が含まれているか
                if any(str(inf.get('total_likes', 0)) in content for inf in influencers):
                    breakdown['csv_influencer_depth'] += 0.5
            
            # 時系列トレンド分析の活用
            if csv_analysis.get('temporal_trends'):
                trends = csv_analysis['temporal_trends']
                if trends.get('peak_day'):
                    breakdown['csv_temporal_analysis'] += 1.0
                if trends.get('trend_direction'):
                    breakdown['csv_temporal_analysis'] += 0.5
                    if trends['trend_direction'] in content:
                        breakdown['csv_temporal_analysis'] += 0.5
            
            # エンゲージメント指標の活用
            if csv_analysis.get('engagement_stats'):
                engagement = csv_analysis['engagement_stats']
                if 'total_likes' in engagement or 'total_reposts' in engagement:
                    breakdown['csv_engagement_metrics'] += 1.0
                    # 具体的な数値が含まれているか
                    if any(str(v) in content for v in engagement.values() if isinstance(v, (int, float))):
                        breakdown['csv_engagement_metrics'] += 0.5
        
        # ========================================
        # 4. 動画分析特有の評価
        # ========================================
        
        if response.data_source_type == DataSourceType.VIDEO_DATA or video_mentioned:
            source_specific = response.source_specific_content if hasattr(response, 'source_specific_content') else {}
            video_analysis = source_specific.get('video_analysis', {})
            
            # センチメント分析の品質
            if video_analysis.get('sentiment_analysis'):
                sentiment = video_analysis['sentiment_analysis']
                if sentiment.get('distribution'):
                    breakdown['video_sentiment_quality'] += 1.0
                if sentiment.get('top_videos_by_sentiment'):
                    breakdown['video_sentiment_quality'] += 0.5
                    # 具体的な動画タイトルが言及されているか
                    for sent_type, videos in sentiment['top_videos_by_sentiment'].items():
                        for video in videos[:2]:
                            if video.get('title', '')[:30] in content:
                                breakdown['video_sentiment_quality'] += 0.3
            
            # インフルエンサー（チャンネル）分析の深さ
            if video_analysis.get('influencer_analysis'):
                channels = video_analysis['influencer_analysis']
                breakdown['video_influencer_depth'] = min(2.0, len(channels) * 0.3)
                
                # 具体的なチャンネル名が言及されているか
                for channel in channels[:3]:
                    if channel.get('channel_name', '') in content:
                        breakdown['video_influencer_depth'] += 0.3
                
                # フォロワー数や視聴回数が含まれているか
                if any(str(ch.get('follower_count', 0)) in content for ch in channels):
                    breakdown['video_influencer_depth'] += 0.5
            
            # トピック分析の豊富さ
            if video_analysis.get('topic_analysis'):
                topics = video_analysis['topic_analysis']
                if topics.get('matched_keywords'):
                    breakdown['video_topic_richness'] += min(1.5, len(topics['matched_keywords']) * 0.2)
                if topics.get('political_analysis'):
                    political = topics['political_analysis']
                    # 政党や政治家の言及
                    if political.get('parties_mentioned') or political.get('politicians_mentioned'):
                        breakdown['video_topic_richness'] += 0.5
                if topics.get('trending_topics'):
                    breakdown['video_topic_richness'] += 0.5
            
            # パフォーマンス分析の活用
            if video_analysis.get('engagement_metrics'):
                metrics = video_analysis['engagement_metrics']
                if metrics.get('performance_distribution'):
                    breakdown['video_performance_analysis'] += 1.0
                    # バイラル動画の言及
                    if 'バイラル' in content or '10万' in content:
                        breakdown['video_performance_analysis'] += 0.5
                if metrics.get('total_views'):
                    if str(metrics['total_views']) in content:
                        breakdown['video_performance_analysis'] += 0.5

        # ========================================
        # 5. 統合的な分析の評価
        # ========================================
        
        # リスク認識の評価
        critical_risks = ['消防法違反', '避難器具', 'マンション', '10月にバースト']
        high_risks = ['信号無視', '反Luup教', 'ちんにい', '岡井批判']
        
        for risk in critical_risks:
            if risk in content:
                breakdown['risk_awareness'] += 1.0
        
        for risk in high_risks:
            if risk in content:
                breakdown['risk_awareness'] += 0.5
        
        # クラスター理解度の評価
        clusters = ['CM出演', '信号無視', '銭湯', '消防法', '反感', '岡井', 'ポート']
        cluster_count = sum(1 for cluster in clusters if cluster in content)
        breakdown['cluster_understanding'] = min(2.0, cluster_count * 0.5)
        
        # インフルエンサー対策の評価
        if any(account in content for account in ['ちんにい', '@chinniisan', '反Luup教']):
            breakdown['influencer_strategy'] += 1.0
        if 'インフルエンサー' in content or 'Pagerank' in content:
            breakdown['influencer_strategy'] += 0.5
        if '要監視' in content or 'リスト化' in content:
            breakdown['influencer_strategy'] += 0.5
        
        # タイムライン認識の評価
        timeline_keywords = ['10月', 'バースト', '9月', '8月', '時系列', 'タイミング']
        timeline_score = sum(0.3 for keyword in timeline_keywords if keyword in content)
        breakdown['timeline_awareness'] = min(1.5, timeline_score)

        # 統合的な分析を示す表現の評価
        integration_keywords = [
            '統合', '総合的', '多角的', '横断的', '複合的', '包括的',
            '一方で', 'これに対し', 'さらに', '加えて', '組み合わせ',
            'X（Twitter）', 'YouTube', 'TikTok', 'SNS', 'ソーシャルメディア'
        ]
        integration_score = sum(1 for keyword in integration_keywords if keyword in content)
        breakdown['integration_quality'] = min(2.5, integration_score * 0.4)
        
        # 知識統合の評価
        synthesis_indicators = [
            'PDFデータ.*Web検索',
            'Web検索.*動画',
            'PDF.*動画',
            'CSV.*動画',
            'Twitter.*YouTube',
            'X.*TikTok',
            '4つのデータソース',
            '複数の観点',
            'それぞれのデータ',
            'SNS分析.*政策文書',
            'メディア報道.*投稿データ'
        ]
        import re
        synthesis_score = sum(1 for pattern in synthesis_indicators 
                            if re.search(pattern, content))
        breakdown['knowledge_synthesis'] = min(3.0, synthesis_score * 0.75)
        
        # クロスプラットフォーム統合の評価
        if (csv_mentioned and video_mentioned) or ('Twitter' in content and 'YouTube' in content):
            breakdown['cross_platform_synthesis'] += 1.5
            # 具体的な比較や関連付けがあるか
            comparison_terms = ['比較', '対照的', '共通', '異なる', '一致', '相違']
            if any(term in content for term in comparison_terms):
                breakdown['cross_platform_synthesis'] += 1.0
        
        # 包括的カバレッジの評価（全データソースからの洞察統合）
        if source_count >= 3:
            breakdown['comprehensive_coverage'] = source_count * 0.5
            # 各ソースからの具体的データが含まれているか
            concrete_data_count = 0
            if any(term in content for term in ['議席', '公約', 'マニフェスト']):  # PDF
                concrete_data_count += 1
            if any(term in content for term in ['報道', '専門家', 'ニュース']):  # Web
                concrete_data_count += 1
            if any(term in content for term in ['視聴回数', 'チャンネル', '動画']):  # Video
                concrete_data_count += 1
            if any(term in content for term in ['いいね', 'リポスト', 'ツイート']):  # CSV
                concrete_data_count += 1
            
            breakdown['comprehensive_coverage'] += concrete_data_count * 0.5
        
        # ========================================
        # 6. 学習履歴による加点
        # ========================================
        
        if hasattr(response, 'learning_history') and response.learning_history:
            learning_bonus = len(response.learning_history) * 0.5
            # クロスソース学習の場合は追加ボーナス
            cross_source_learnings = [h for h in response.learning_history 
                                    if h.get('cross_source_learning')]
            learning_bonus += len(cross_source_learnings) * 1.0
            
            # CSV/動画間の学習は特にボーナス
            for learning in response.learning_history:
                source_types = learning.get('source_types', {})
                if source_types:
                    winner = source_types.get('winner', '')
                    loser = source_types.get('loser', '')
                    if ('csv' in winner and 'video' in loser) or ('video' in winner and 'csv' in loser):
                        learning_bonus += 0.5
            
            breakdown['learning_bonus'] = min(4.0, learning_bonus)
        
        # ========================================
        # 7. 重み付けを適用（4ソース統合を最重視）
        # ========================================

        weights = {
            'base_quality': 0.4,
            'data_richness': 0.8,
            'source_coverage': 2.5,  # 最重要
            'solution_oriented': 2.0,
            'stakeholder_balance': 1.5,
            'reputation_insights': 2.0,
            'actionable_advice': 2.0,
            'risk_awareness': 2.5,  # 高い重要度
            'cluster_understanding': 1.8,
            'influencer_strategy': 2.0,
            'timeline_awareness': 1.2,
            'integration_quality': 1.5,
            'depth_bonus': 1.0,
            'knowledge_synthesis': 2.0,  # 重要
            # CSV分析
            'csv_sentiment_quality': 0.6,
            'csv_influencer_depth': 0.7,
            'csv_temporal_analysis': 0.5,
            'csv_engagement_metrics': 0.6,
            # 動画分析
            'video_sentiment_quality': 0.6,
            'video_influencer_depth': 0.7,
            'video_topic_richness': 0.8,
            'video_performance_analysis': 0.6,
            # 統合評価
            'cross_platform_synthesis': 1.8,  # 重要
            'comprehensive_coverage': 2.0,  # 重要
            'learning_bonus': 1.2
        }
        
        weighted_total = sum(
            breakdown[key] * weights.get(key, 1.0)
            for key in breakdown
        )
        
        # ========================================
        # 8. 強みと弱みの特定
        # ========================================
        
        winning_factors = []
        missing_elements = []
        
        # 強みの特定
        if breakdown['source_coverage'] >= 4.0:
            winning_factors.append("全4データソースの完全統合")
        elif breakdown['source_coverage'] >= 3.0:
            winning_factors.append("3つのデータソースの統合")
        
        if breakdown['knowledge_synthesis'] >= 2.0:
            winning_factors.append("優れた知識統合")
        
        if breakdown['cross_platform_synthesis'] >= 2.0:
            winning_factors.append("SNSプラットフォーム横断分析")
        
        if breakdown['csv_sentiment_quality'] >= 1.5 or breakdown['video_sentiment_quality'] >= 1.5:
            winning_factors.append("詳細なセンチメント分析")
        
        if breakdown['csv_influencer_depth'] >= 1.5 or breakdown['video_influencer_depth'] >= 1.5:
            winning_factors.append("インフルエンサー影響力分析")
        
        if breakdown['data_richness'] >= 1.5:
            winning_factors.append("豊富な具体的データ")
        
        # 弱みの特定
        if breakdown['source_coverage'] < 2.0:
            missing_elements.append("データソースの不足")
        
        if breakdown['knowledge_synthesis'] < 1.0:
            missing_elements.append("知識統合の欠如")
        
        if breakdown['cross_platform_synthesis'] < 0.5:
            missing_elements.append("プラットフォーム間の関連付け不足")
        
        if csv_mentioned and breakdown['csv_sentiment_quality'] < 0.5:
            missing_elements.append("Twitter感情分析の活用不足")
        
        if video_mentioned and breakdown['video_topic_richness'] < 0.5:
            missing_elements.append("動画トピック分析の活用不足")
        
        if breakdown['comprehensive_coverage'] < 1.0 and source_count >= 2:
            missing_elements.append("具体的データの引用不足")

        if breakdown['solution_oriented'] >= 1.5:
            winning_factors.append("具体的な解決策の提示")

        if breakdown['stakeholder_balance'] >= 1.5:
            winning_factors.append("バランスの取れたステークホルダー分析")

        if breakdown['reputation_insights'] >= 1.5:
            winning_factors.append("評判改善への深い洞察")
        
        if breakdown['solution_oriented'] < 1.0:
            missing_elements.append("解決策の不足")

        if breakdown['actionable_advice'] < 1.0:
            missing_elements.append("実行可能なアドバイスの欠如")
        
        if breakdown['risk_awareness'] >= 2.0:
            winning_factors.append("消防法違反リスクの正確な認識")

        if breakdown['influencer_strategy'] >= 1.5:
            winning_factors.append("反Luup教への適切な対策")

        if breakdown['cluster_understanding'] >= 1.5:
            winning_factors.append("世論クラスターの包括的理解")
        
        if breakdown['risk_awareness'] < 1.0:
            missing_elements.append("重大リスクの認識不足")

        if breakdown['timeline_awareness'] < 0.5:
            missing_elements.append("炎上タイミングの理解不足")

        if breakdown['influencer_strategy'] < 1.0:
            missing_elements.append("キーインフルエンサー対策の欠如")

        return DetailedScore(
            total=weighted_total,
            breakdown=breakdown,
            winning_factors=winning_factors,
            missing_elements=missing_elements,
            tournament_round=response.tournament_round
        )

    # ========================================
    # 補助メソッド（追加が必要）
    # ========================================

# ====================================
# TournamentSelector クラスの補助メソッド
# ====================================

    def _check_csv_content(self, content: str) -> bool:
        """CSV（X/Twitter）関連コンテンツの存在確認
        
        Args:
            content (str): 評価対象のテキストコンテンツ
            
        Returns:
            bool: CSV/Twitter関連の内容が含まれている場合True
        """
        # Twitter/X関連の直接的な言及
        twitter_platforms = [
            'CSV', 'csv',
            'Twitter', 'twitter', 'TWITTER',
            'X（Twitter）', 'X(Twitter)', 'X（旧Twitter）',
            'ツイッター', 'ツイート', 'つぶやき',
            'X上で', 'X上の', 'Xで', 'Xの',
            'SNS投稿', 'ソーシャルメディア投稿'
        ]
        
        # Twitter特有の機能・用語
        twitter_features = [
            'リポスト', 'リツイート', 'RT', 'repost', 'retweet',
            'いいね', 'ライク', 'Like', 'likes',
            'フォロワー', 'フォロー', 'follower', 'following',
            'リプライ', 'リプ', 'reply', 'replies',
            'タイムライン', 'TL', 'timeline',
            'トレンド', 'trending', 'バズ', 'バイラル',
            'ハッシュタグ', '#', 'hashtag',
            'メンション', '@', 'mention',
            'インプレッション', 'impression',
            'エンゲージメント', 'engagement'
        ]
        
        # Twitter分析に関連する用語
        twitter_analytics = [
            'ツイート分析', 'ポスト分析', '投稿分析',
            'X分析', 'Twitter分析',
            'センチメント', 'sentiment',
            'ポジティブな投稿', 'ネガティブな投稿',
            '肯定的な反応', '否定的な反応',
            '投稿数', 'ツイート数', 'ポスト数',
            '拡散力', '影響力', 'インフルエンサー',
            '投稿者', 'アカウント', 'ユーザー'
        ]
        
        # 統計・数値表現との組み合わせ
        statistical_patterns = [
            r'\d+件の(投稿|ツイート|ポスト)',
            r'\d+人の(フォロワー|ユーザー)',
            r'\d+(いいね|リポスト|リツイート)',
            r'(投稿|ツイート).*\d+%',
            r'(ポジティブ|ネガティブ).*\d+%',
            r'平均.*いいね',
            r'総(リポスト|リツイート)数'
        ]
        
        # プラットフォーム、機能、分析用語のいずれかが含まれているかチェック
        for term in twitter_platforms:
            if term in content:
                return True
        
        for feature in twitter_features:
            if feature in content:
                return True
        
        for analytics_term in twitter_analytics:
            if analytics_term in content:
                return True
        
        # 正規表現パターンのチェック
        import re
        for pattern in statistical_patterns:
            if re.search(pattern, content):
                return True
        
        # 文脈的な判定（複数の関連語が含まれる場合）
        context_indicators = 0
        weak_indicators = ['投稿', 'ポスト', '反応', '意見', 'SNS', '評判', '評価']
        for indicator in weak_indicators:
            if indicator in content:
                context_indicators += 1
        
        # 2つ以上の弱い指標がある場合はCSV関連とみなす
        if context_indicators >= 2:
            return True
        
        return False

    def _evaluate_source_depth(self, content: str, source_type: str) -> float:
        """各データソースの深度を評価（CSV対応版）
        
        Args:
            content (str): 評価対象のテキストコンテンツ
            source_type (str): データソースタイプ ('pdf', 'web', 'video', 'csv')
            
        Returns:
            float: 深度スコア（0.0-2.0）
        """
        depth_score = 0.0
        
        if source_type == 'pdf':
            # PDF（政策文書）の深度評価
            if '政策' in content and '具体的' in content:
                depth_score += 0.5
            if 'マニフェスト' in content or '公約' in content:
                depth_score += 0.5
            if '制度' in content or '改革' in content:
                depth_score += 0.5
            if '法案' in content or '提案' in content:
                depth_score += 0.3
            if '施策' in content or '計画' in content:
                depth_score += 0.2
        
        elif source_type == 'web':
            # Web検索（ニュース・専門家分析）の深度評価
            if '報道' in content or 'ニュース' in content:
                depth_score += 0.5
            if '専門家' in content or '分析' in content:
                depth_score += 0.5
            if '最新' in content or '動向' in content:
                depth_score += 0.5
            if '記事' in content or 'メディア' in content:
                depth_score += 0.3
            if '発表' in content or '速報' in content:
                depth_score += 0.2
        
        elif source_type == 'video':
            # 動画データ（YouTube/TikTok）の深度評価
            if '視聴回数' in content or '再生' in content:
                depth_score += 0.5
            if 'YouTube' in content or 'TikTok' in content:
                depth_score += 0.5
            if 'チャンネル' in content or '動画' in content:
                depth_score += 0.4
            if 'トレンド' in content or '人気' in content:
                depth_score += 0.3
            if 'ショート' in content or 'Shorts' in content:
                depth_score += 0.3
            if 'バイラル' in content or '急上昇' in content:
                depth_score += 0.3
            if '配信者' in content or 'YouTuber' in content:
                depth_score += 0.2
        
        elif source_type == 'csv':
            # CSV（X/Twitter）の深度評価
            
            # 基本的なTwitter機能への言及
            if 'いいね' in content or 'リポスト' in content or 'リツイート' in content:
                depth_score += 0.5
            
            # センチメント分析への言及
            if 'センチメント' in content or '感情' in content or '評判' in content:
                depth_score += 0.4
            if 'ポジティブ' in content or 'ネガティブ' in content:
                depth_score += 0.3
            
            # インフルエンサー・影響力分析への言及
            if 'インフルエンサー' in content or '影響力' in content:
                depth_score += 0.4
            if 'フォロワー' in content:
                depth_score += 0.3
            
            # エンゲージメント分析への言及
            if 'エンゲージメント' in content or '反応' in content:
                depth_score += 0.4
            if '拡散' in content or 'バズ' in content:
                depth_score += 0.3
            
            # プラットフォーム名の明示的な言及
            if 'Twitter' in content or 'X（Twitter）' in content or 'X上' in content:
                depth_score += 0.3
            
            # 時系列・トレンド分析への言及
            if 'トレンド' in content or '推移' in content or '傾向' in content:
                depth_score += 0.3
            
            # 具体的な数値データへの言及
            import re
            # 数値を含むTwitter関連の表現
            numeric_patterns = [
                r'\d+件の(投稿|ツイート)',
                r'\d+人のフォロワー',
                r'\d+(いいね|リポスト)',
                r'\d+%.*(?:ポジティブ|ネガティブ)',
                r'平均\d+',
                r'合計\d+'
            ]
            
            for pattern in numeric_patterns:
                if re.search(pattern, content):
                    depth_score += 0.2
                    break  # 1つ見つかれば十分
            
            # 具体的なアカウント名やハッシュタグへの言及
            if '@' in content or '#' in content:
                depth_score += 0.2
        
        # スコアの上限を2.0に制限
        return min(2.0, depth_score)

    def _check_pdf_content(self, content: str) -> bool:
        """PDF関連コンテンツの存在確認（既存メソッド）"""
        pdf_indicators = [
            'PDF', 'pdf', '公約', '政策文書', '参考資料', 'マニフェスト',
            '政策提案', '公式文書', '党の方針', '制度改革', '施策',
            '法案', '白書', '報告書', '提言', '政策集'
        ]
        return any(indicator in content for indicator in pdf_indicators)


    def _check_web_content(self, content: str) -> bool:
        """Web検索関連コンテンツの存在確認（既存メソッド）"""
        web_indicators = [
            'Web', 'web', 'ウェブ', '検索', 'ニュース', '報道', '新聞',
            '専門家', '分析記事', 'メディア', 'オンライン', '記事',
            '最新情報', '速報', 'ネット', 'サイト', 'ページ',
            'URL', 'リンク', '配信', '掲載'
        ]
        return any(indicator in content for indicator in web_indicators)

    def _check_video_content(self, content: str) -> bool:
        """動画関連コンテンツの存在確認（既存メソッド）"""
        video_indicators = [
            '動画', 'YouTube', 'youtube', 'TikTok', 'tiktok', '視聴', 
            'チャンネル', 'ショート', 'Shorts', '再生回数', '視聴者', 
            'トレンド', '配信', 'ビデオ', 'video', '投稿動画',
            'YouTuber', 'ユーチューバー', 'バイラル', '急上昇',
            'サムネ', 'サムネイル', '概要欄', 'コメント欄'
        ]
        return any(indicator in content for indicator in video_indicators)

    def _identify_winning_factors(self, breakdown: Dict[str, float], concrete_count: int) -> List[str]:
        """勝因を特定"""
        factors = []
        if concrete_count > 30:
            factors.append("豊富な具体的データ")
        if breakdown['source_coverage'] >= 3.0:
            factors.append("全データソースの完全統合")
        if breakdown['length_appropriateness'] >= 1.5:
            factors.append("適切な分析の深さ")
        return factors

    def _identify_missing_elements(self, breakdown: Dict[str, float], concrete_count: int) -> List[str]:
        """不足要素を特定"""
        missing = []
        if concrete_count < 15:
            missing.append("具体的データの不足")
        if breakdown['length_appropriateness'] < 1.0:
            missing.append("分析の深さ不足")
        return missing
    
    def _check_pdf_content(self, content: str) -> bool:
        """PDF関連コンテンツの存在確認"""
        pdf_indicators = [
            'PDF', '公約', '政策文書', '参考資料', 'マニフェスト',
            '政策提案', '公式文書', '党の方針', '制度改革'
        ]
        return any(indicator in content for indicator in pdf_indicators)
    
    def _check_web_content(self, content: str) -> bool:
        """Web検索関連コンテンツの存在確認"""
        web_indicators = [
            'Web', 'ウェブ', '検索', 'ニュース', '報道', '新聞',
            '専門家', '分析記事', 'メディア', 'オンライン'
        ]
        return any(indicator in content for indicator in web_indicators)
    
    def _check_video_content(self, content: str) -> bool:
        """動画関連コンテンツの存在確認"""
        video_indicators = [
            '動画', 'YouTube', 'TikTok', '視聴', 'チャンネル',
            'ショート', '再生回数', '視聴者', 'トレンド', '配信'
        ]
        return any(indicator in content for indicator in video_indicators)
    
    def _integrate_with_source_constraints(self, winner: PersuasiveResponse, loser: PersuasiveResponse,
                                        winner_source: DataSourceType, loser_source: DataSourceType) -> str:
        """データソース制約を守りながら知識を統合"""
        
        # 敗者から具体的な情報を抽出（構造化）
        loser_structured_data = self._extract_structured_data(loser.content, loser_source, loser.source_specific_content)
        
        # 勝者のデータソースに応じた応答構造を決定
        if winner_source == DataSourceType.PDF:
            return self._create_pdf_based_response(winner, loser_structured_data, loser_source)
        elif winner_source == DataSourceType.WEB_SEARCH:
            return self._create_web_based_response(winner, loser_structured_data, loser_source)
        elif winner_source == DataSourceType.VIDEO_DATA:
            return self._create_video_based_response(winner, loser_structured_data, loser_source)
        else:
            return winner.content

    def _create_web_based_response(self, winner: PersuasiveResponse,
                                loser_data: Dict[str, Any], loser_source: DataSourceType) -> str:
        """Web検索データソースを基にした3段落構造の応答を生成"""
        
        web_content = winner.content
        web_specific = winner.source_specific_content or {}
        
        # 第1段落：Web検索データの分析
        paragraph1 = self._extract_first_paragraph(web_content)
        if not paragraph1:
            paragraph1 = "Web検索の結果によると、2025年参議院選挙における参政党の躍進は、従来の政治報道では捉えきれない複合的な要因が関係しています。"
        
        # 第2段落：統合的考察
        paragraph2 = "最新のメディア報道と専門家の分析から、参政党の成功要因が浮かび上がります。"
        
        if loser_source == DataSourceType.VIDEO_DATA and loser_data.get('specific_data'):
            views = loser_data['specific_data'].get('total_views', 0)
            if views > 0:
                paragraph2 += f"動画プラットフォームでの{views:,}回という視聴回数は、新しいメディア環境での情報拡散力を示しています。"
            else:
                paragraph2 += "SNS動画による拡散は、従来のメディアとは異なる層に訴求していることが分かります。"
        elif loser_source == DataSourceType.PDF:
            paragraph2 += "政策文書の分析が示す既存政党の枠組みと、メディアが報じる有権者の期待のギャップが重要な要因です。"
        
        paragraph2 += "このような複数の視点を統合すると、参政党の躍進は単純な選挙戦術の成功ではなく、構造的な変化の表れです。"
        
        # 第3段落：結論
        paragraph3 = "報道分析から明らかになったのは、参政党の躍進が従来の政治分析の枠組みでは説明できない現象だということです。"
        paragraph3 += "SNSやショート動画の戦略は確かに重要ですが、それは有権者の潜在的な不満や期待を可視化する手段に過ぎません。"
        paragraph3 += "真の要因は、既存の政治システムへの不信と新しいメディア環境が相互作用した結果にあります。"
        
        return f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}"

    def _create_video_based_response(self, winner: PersuasiveResponse,
                                    loser_data: Dict[str, Any], loser_source: DataSourceType) -> str:
        """動画データソースを基にした3段落構造の応答を生成"""
        
        video_specific = winner.source_specific_content or {}
        video_analysis = video_specific.get('video_analysis', {})
        
        # 第1段落：動画データの整理
        paragraph1 = "【動画データの整理】\n"
        
        if video_analysis:
            total_videos = video_analysis.get('total_videos', 0)
            total_views = video_analysis.get('total_views', 0)
            
            paragraph1 += f"YouTube/TikTokの分析対象{total_videos:,}件の動画は、累計{total_views:,}回の視聴を記録しています。"
            
            # トップ動画の具体例
            if 'top_viewed_videos' in video_analysis:
                top_videos = video_analysis['top_viewed_videos'][:2]
                for video in top_videos:
                    if isinstance(video, dict):
                        title = video.get('title', '')[:30]
                        channel = video.get('channel', '')
                        views = video.get('view_count', 0)
                        if title and channel:
                            paragraph1 += f"特に「{title}...」（{channel}）は{views:,}回視聴され、大きな注目を集めました。"
                            break
        else:
            paragraph1 += "動画データの分析によると、SNSでの政治コンテンツが前例のない拡散力を示しています。"
        
        # 第2段落：統合的考察
        paragraph2 = "【統合的な考察】\n"
        paragraph2 += "これらの動画トレンドを"
        
        if loser_source == DataSourceType.PDF:
            paragraph2 += "政策文書の観点から見ると、既存政党の公約では触れられていない論点が動画で頻繁に取り上げられていることが分かります。"
        elif loser_source == DataSourceType.WEB_SEARCH:
            paragraph2 += "専門家の分析と照らし合わせると、メディアが指摘する社会的分断が動画コンテンツにも反映されていることが明らかです。"
        else:
            paragraph2 += "分析すると、従来のメディアでは見落とされがちな視点が浮かび上がります。"
        
        paragraph2 += "この乖離こそが、参政党が独自のナラティブを構築できた背景にあると考えられます。"
        
        # 第3段落：結論
        paragraph3 = "【結論】\n"
        paragraph3 += "動画データが示す圧倒的な視聴回数と拡散力は、単なる選挙戦術の成功ではなく、"
        paragraph3 += "情報消費パターンの根本的な変化を示しています。"
        paragraph3 += "CMVの投稿者が指摘する「SNSやショート動画の戦略」は確かに重要ですが、"
        paragraph3 += "それは表層的な現象であり、より深い社会的・政治的変化の一端に過ぎません。"
        
        return f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}"

    def _create_pdf_based_response(self, winner: PersuasiveResponse, 
                                loser_data: Dict[str, Any], loser_source: DataSourceType) -> str:
        """PDFデータソースを基にした3段落構造の応答を生成（改善版）"""
        
        # PDFデータから具体的な内容を抽出
        pdf_content = winner.content
        pdf_specific = winner.source_specific_content or {}
        
        # 第1段落：PDFデータの分析（自分のデータソース）
        paragraph1 = self._extract_first_paragraph(pdf_content)
        if not paragraph1:
            paragraph1 = "PDFデータによると、2025年参議院選挙において主要政党が掲げた公約は、経済対策や子育て支援、政治資金の透明化など、従来型の政策テーマが中心でした。"
        
        # 第2段落：統合的考察（敗者の具体的データを含める）
        paragraph2 = "政策文書の分析から見えてくるのは、既存政党が提供する政策だけでは説明できない有権者層の存在です。"
        
        # 敗者のデータソースに応じて具体的な情報を追加
        if loser_source == DataSourceType.VIDEO_DATA and loser_data.get('concrete_examples'):
            # 動画の具体例を含める
            video_examples = loser_data['concrete_examples'][:3]
            if video_examples:
                paragraph2 += f"実際に、{video_examples[0]}などの動画が大きな注目を集め、"
                if loser_data['specific_data'].get('total_views', 0) > 0:
                    paragraph2 += f"関連動画全体で{loser_data['specific_data']['total_views']:,}回もの視聴を記録しています。"
                else:
                    paragraph2 += "多くの視聴者の関心を集めています。"
            paragraph2 += "これらの動画で取り上げられているテーマは、既存政党の公約では触れられていない論点が多く、"
        
        elif loser_source == DataSourceType.WEB_SEARCH and loser_data.get('concrete_examples'):
            # Web検索の具体例を含める
            article_examples = loser_data['concrete_examples'][:2]
            if article_examples:
                paragraph2 += f"メディア分析では、{article_examples[0]}など複数の記事が「既存の枠組みへの不信感」を指摘しており、"
            paragraph2 += "専門家らは参政党の躍進を単なる選挙戦術ではなく構造的変化として分析しています。"
        
        paragraph2 += "このような多面的な視点から見ると、参政党の躍進は政策文書に表れない深層的な社会変化の表れと解釈できます。"
        
        # 第3段落：結論
        paragraph3 = "以上の分析から、参政党の躍進はSNS戦略だけでなく、既存政党の政策が捉えきれない有権者の不満や期待、"
        paragraph3 += "そして新しいメディア環境における情報伝達の変化が複合的に作用した結果と言えるでしょう。"
        
        # 敗者から学んだ具体的な洞察を結論に反映
        if loser_data.get('insights'):
            paragraph3 += f"特に、{loser_data['insights'][0]}という点は重要です。"
        
        paragraph3 += "あなたの考える「SNSやショート動画の戦略」という単一要因では、この現象の本質は説明できません。"
        
        return f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}"

    def _extract_structured_data(self, content: str, source: DataSourceType, 
                                source_specific_content: Optional[Dict]) -> Dict[str, Any]:
        """コンテンツから構造化データを抽出（改善版）"""
        structured_data = {
            'source_type': source.value,
            'key_points': [],
            'specific_data': {},
            'insights': [],
            'concrete_examples': []  # 新規追加：具体例を保存
        }
        
        if source == DataSourceType.VIDEO_DATA and source_specific_content:
            # 動画データから具体的な情報を抽出
            video_analysis = source_specific_content.get('video_analysis', {})
            if video_analysis:
                structured_data['specific_data'] = {
                    'total_videos': video_analysis.get('total_videos', 0),
                    'total_views': video_analysis.get('total_views', 0),
                    'top_videos': []
                }
                
                # トップ動画の具体的な情報を抽出
                if 'top_viewed_videos' in video_analysis:
                    for video in video_analysis['top_viewed_videos'][:5]:  # 上位5件に増やす
                        if isinstance(video, dict):
                            video_info = {
                                'title': video.get('title', ''),
                                'channel': video.get('channel', ''),
                                'views': video.get('view_count', 0)
                            }
                            structured_data['specific_data']['top_videos'].append(video_info)
                            # 具体例として保存
                            structured_data['concrete_examples'].append(
                                f"「{video_info['title'][:40]}」（{video_info['channel']}、{video_info['views']:,}回視聴）"
                            )
                
                # 動画トレンドから洞察を抽出
                structured_data['insights'].append("SNSでの拡散力が既存メディアを凌駕")
        
        elif source == DataSourceType.WEB_SEARCH and source_specific_content:
            # Web検索結果から専門家の分析を抽出
            for key, value in source_specific_content.items():
                if 'search' in key and isinstance(value, list):
                    for result in value[:5]:  # 上位5件を処理
                        if isinstance(result, dict):
                            article_info = {
                                'title': result.get('title', ''),
                                'source': result.get('url', '').split('/')[2] if result.get('url') else '',
                                'key_point': result.get('content', '')[:200]
                            }
                            structured_data['specific_data'].setdefault('articles', []).append(article_info)
                            # 具体例として保存
                            if article_info['title']:
                                structured_data['concrete_examples'].append(
                                    f"『{article_info['title'][:40]}』（{article_info['source']}）"
                                )
            
            structured_data['insights'].append("専門家による多角的な要因分析")
        
        return structured_data     

    def _extract_first_paragraph(self, content: str) -> str:
        """コンテンツから最初の段落を抽出"""
        paragraphs = content.split('\n\n')
        if paragraphs:
            return paragraphs[0].strip()
        
        # 改行がない場合は最初の2文を抽出
        sentences = content.split('。')
        if len(sentences) >= 2:
            return '。'.join(sentences[:2]) + '。'

    def _evaluate_source_depth(self, content: str, source_type: str) -> float:
        """各データソースの深度を評価"""
        depth_score = 0.0
        
        if source_type == 'pdf':
            if '政策' in content and '具体的' in content:
                depth_score += 0.5
            if 'マニフェスト' in content or '公約' in content:
                depth_score += 0.5
            if '制度' in content or '改革' in content:
                depth_score += 0.5
        elif source_type == 'web':
            if '報道' in content or 'ニュース' in content:
                depth_score += 0.5
            if '専門家' in content or '分析' in content:
                depth_score += 0.5
            if '最新' in content or '動向' in content:
                depth_score += 0.5
        elif source_type == 'video':
            if '視聴回数' in content or '再生' in content:
                depth_score += 0.5
            if 'トレンド' in content or '人気' in content:
                depth_score += 0.5
            if 'SNS' in content or 'バイラル' in content:
                depth_score += 0.5
        
        return min(1.5, depth_score)
    
    def _extract_insights_from_content(self, content: Dict, source: DataSourceType) -> List[str]:
        """コンテンツから洞察のみを抽出（具体的データは除外）"""
        insights = []
        
        if source == DataSourceType.VIDEO_DATA:
            if 'video_analysis' in content:
                analysis = content['video_analysis']
                if analysis.get('total_views', 0) > 100000000:
                    insights.append("億単位の視聴回数が示す情報拡散力")
                if analysis.get('total_videos', 0) > 1000:
                    insights.append("大量の関連動画が存在する話題性")
        
        elif source == DataSourceType.PDF:
            insights.append("政策文書に基づく制度的観点")
            insights.append("既存政党の公約分析")
        
        elif source == DataSourceType.WEB_SEARCH:
            insights.append("専門家による要因分析")
            insights.append("メディア報道の論調")

        elif source == DataSourceType.CSV:
            if 'csv_analysis' in content:
                analysis = content['csv_analysis']
                if analysis.get('sentiment_analysis', {}).get('by_sentiment', {}).get('positive', {}).get('percentage', 0) > 50:
                    insights.append("SNS上での肯定的な評価が多数")
                if analysis.get('top_influencers'):
                    insights.append("影響力のあるアカウントによる拡散")
        
        return insights

    def _merge_source_contents(self, winner_content: Optional[Dict], loser_content: Optional[Dict],
                            winner_source: DataSourceType, loser_source: DataSourceType) -> Dict:
        """勝者と敗者のソース固有コンテンツをマージ（メタデータとして保持）"""
        merged = winner_content.copy() if winner_content else {}
        
        # 敗者のデータをメタデータとして追加（直接アクセスはできないが、統合の参考として保持）
        if loser_content:
            merged[f'learned_from_{loser_source.value}'] = {
                'source_type': loser_source.value,
                'metadata_only': True,
                'insights': self._extract_insights_from_content(loser_content, loser_source)
            }
        
        return merged

    def _count_concrete_examples(self, content: str) -> int:
        """文章中の具体例（動画タイトル、記事名、数値データ）の数をカウント"""
        import re
        count = 0
        
        # 「」で囲まれたタイトル
        count += len(re.findall(r'「[^」]+」', content))
        
        # 『』で囲まれた記事タイトル
        count += len(re.findall(r'『[^』]+』', content))
        
        # 数値データ（視聴回数など）
        count += len(re.findall(r'\d{1,3}(?:,\d{3})*回', content))
        
        # チャンネル名や出典の言及
        count += len(re.findall(r'（[^）]+、\d+', content))
        
        return count
    
    def _manually_integrate_response(self, winner: PersuasiveResponse, 
                                loser: PersuasiveResponse,
                                integrated_data: Dict[str, Any]) -> str:
        """手動で統合応答を作成（2000文字×3段落版）"""
        
        winner_data = integrated_data['winner_data']
        loser_data = integrated_data['loser_data']
        
        # 第1段落：勝者のデータソースの詳細分析（2000文字）
        paragraph1 = self._create_detailed_first_paragraph(winner, winner_data, 2000)
        
        # 第2段落：マルチソース統合（2000文字）
        paragraph2 = self._create_integrated_second_paragraph(winner_data, loser_data, 2000)
        
        # 第3段落：深層的な結論（2000文字）
        paragraph3 = self._create_deep_conclusion(winner_data, loser_data, 2000)
        
        return f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}"

    def _create_detailed_first_paragraph(self, response: PersuasiveResponse, 
                                    data: Dict[str, Any], target_length: int) -> str:
        """詳細な第1段落を作成"""
        source_type = response.data_source_type
        
        if source_type == DataSourceType.WEB_SEARCH:
            paragraph = "【Web検索による包括的分析】\n"
            paragraph += "最新のメディア報道と専門家の分析を総合的に検証した結果、2025年参議院選挙における参政党の躍進には、"
            paragraph += "従来の政治学的枠組みでは説明困難な複合的要因が存在することが明らかになりました。"
            
            # 具体的な記事と分析を追加
            if data.get('specific_examples', {}).get('articles'):
                for article in data['specific_examples']['articles'][:5]:
                    paragraph += f"\n\n『{article['title']}』（{article['source']}）では、"
                    paragraph += f"{article['content'][:200]}...という分析が提示されています。"
            
            # データポイントを織り込む
            for dp in data.get('all_data_points', [])[:10]:
                if '専門家' in dp:
                    paragraph += f"\n\n{dp}"
        
        # 必要に応じて文章を拡張
        while len(paragraph) < target_length - 200:
            paragraph += self._add_analytical_content(source_type, data)
        
        return paragraph[:target_length]

    def _create_integrated_second_paragraph(self, winner_data: Dict[str, Any], 
                                        loser_data: Dict[str, Any], 
                                        target_length: int) -> str:
        """統合的な第2段落を作成"""
        paragraph = "【マルチソース統合分析】\n"
        paragraph += "3つの異なるデータソース（政策文書、Web検索、動画データ）を統合的に分析することで、"
        paragraph += "参政党躍進の多面的な要因が浮かび上がってきます。\n\n"
        
        # PDFデータの統合
        paragraph += "まず政策文書の分析から、"
        # ここに具体的な政党名、政策を追加
        
        # 動画データの統合
        paragraph += "\n\n次に動画プラットフォームの分析では、"
        # ここに具体的な動画タイトル、視聴回数を追加
        
        # Web検索データの統合
        paragraph += "\n\nさらに専門家の分析によると、"
        # ここに具体的な専門家の見解を追加
        
        return paragraph[:target_length]

    def _create_deep_conclusion(self, winner_data: Dict[str, Any], 
                            loser_data: Dict[str, Any], 
                            target_length: int) -> str:
        """深層的な結論を作成"""
        paragraph = "【統合的洞察と結論】\n"
        paragraph += "以上の3つのデータソースを統合的に分析した結果、参政党の躍進は単なるSNS戦略の成功ではなく、"
        paragraph += "より深層的な社会構造の変化を反映した現象であることが明らかになりました。\n\n"
        
        # 各データソースから得られた証拠を統合
        # 深い分析と洞察を追加
        
        return paragraph[:target_length]

    def _add_analytical_content(self, source_type: DataSourceType, data: Dict[str, Any]) -> str:
        """分析的な内容を追加"""
        additional = "\n\nさらに詳細に分析すると、"
        
        # データソースに応じた分析を追加
        if source_type == DataSourceType.WEB_SEARCH:
            additional += "メディアの論調には明確な変化が見られ、"
        elif source_type == DataSourceType.VIDEO_DATA:
            additional += "動画コンテンツの拡散パターンには特定の傾向があり、"
        elif source_type == DataSourceType.PDF:
            additional += "政策文書の詳細な検証により、"
        
        return additional
    
    def _format_concrete_data_for_prompt(self, data: Dict[str, Any]) -> str:
        """具体的データをプロンプト用にフォーマット"""
        formatted = []
        
        if data['source_type'] == 'video_data':
            examples = data.get('specific_examples', {})
            if examples.get('total_views'):
                formatted.append(f"総視聴回数: {examples['total_views']:,}回")
            if examples.get('total_videos'):
                formatted.append(f"分析動画数: {examples['total_videos']:,}件")
            
            if examples.get('videos'):
                formatted.append("\n具体的な動画:")
                for i, video in enumerate(examples['videos'][:10], 1):
                    formatted.append(f"{i}. 「{video['title']}」")
                    formatted.append(f"   チャンネル: {video['channel']}")
                    formatted.append(f"   視聴回数: {video['views']:,}回")
                    if video.get('likes'):
                        formatted.append(f"   いいね数: {video['likes']:,}")
        
        elif data['source_type'] == 'pdf':
            examples = data.get('specific_examples', {})
            if examples.get('parties'):
                formatted.append(f"言及政党: {', '.join(set(examples['parties']))}")
            
            if examples.get('policies'):
                formatted.append("\n具体的な政策:")
                for i, policy in enumerate(examples['policies'][:10], 1):
                    formatted.append(f"{i}. {policy}")
            
            if data.get('all_data_points'):
                numbers = [dp for dp in data['all_data_points'] if any(c.isdigit() for c in dp)]
                if numbers:
                    formatted.append(f"\n数値データ: {', '.join(numbers[:10])}")
        
        elif data['source_type'] == 'web_search':
            examples = data.get('specific_examples', {})
            if examples.get('articles'):
                formatted.append("\n報道記事:")
                for i, article in enumerate(examples['articles'][:10], 1):
                    formatted.append(f"{i}. 『{article['title']}』")
                    formatted.append(f"   メディア: {article['source']}")
                    if article.get('content'):
                        formatted.append(f"   要約: {article['content'][:100]}...")
        
        # 全てのデータポイント
        if data.get('all_data_points'):
            formatted.append(f"\nその他の具体的データ（{len(data['all_data_points'])}件）:")
            for dp in data['all_data_points'][:20]:
                formatted.append(f"- {dp}")
        
        return "\n".join(formatted)

    def _extract_all_concrete_data(self, response: PersuasiveResponse) -> Dict[str, Any]:
        """応答から全ての具体的データを抽出（強化版）"""
        concrete_data = {
            'source_type': response.data_source_type.value if response.data_source_type else 'unknown',
            'all_data_points': [],
            'specific_examples': {},
            'raw_content': response.content
        }
        
        if response.data_source_type == DataSourceType.VIDEO_DATA and response.source_specific_content:
            analysis = response.source_specific_content.get('video_analysis', {})
            if analysis:
                # 基本統計
                concrete_data['specific_examples']['total_videos'] = analysis.get('total_videos', 0)
                concrete_data['specific_examples']['total_views'] = analysis.get('total_views', 0)
                
                # 全ての動画情報を抽出
                if 'top_viewed_videos' in analysis:
                    concrete_data['specific_examples']['videos'] = []
                    for video in analysis['top_viewed_videos'][:10]:  # 上位10件まで拡大
                        if isinstance(video, dict):
                            video_data = {
                                'title': video.get('title', ''),
                                'channel': video.get('channel', ''),
                                'views': video.get('view_count', 0),
                                'likes': video.get('likes', 0)
                            }
                            concrete_data['specific_examples']['videos'].append(video_data)
                            # データポイントとして追加
                            concrete_data['all_data_points'].append(
                                f"「{video_data['title']}」（{video_data['channel']}、{video_data['views']:,}回視聴）"
                            )
        
        elif response.data_source_type == DataSourceType.PDF and response.source_specific_content:
            import re
            concrete_data['specific_examples']['parties'] = []
            concrete_data['specific_examples']['policies'] = []
            
            for key, value in response.source_specific_content.items():
                if 'pdf' in key and isinstance(value, dict):
                    sections = value.get('sections', [])
                    for section in sections:
                        # 政党名を全て抽出
                        parties = re.findall(r'(自民党|公明党|参政党|国民民主党|立憲民主党|日本維新の会|れいわ新選組|社民党)', section)
                        concrete_data['specific_examples']['parties'].extend(parties)
                        
                        # 政策キーワードを抽出
                        policies = re.findall(r'([^。]{0,20}(?:政策|公約|改革|支援|規制|推進|強化|無償化|削減|廃止)[^。]{0,20})', section)
                        concrete_data['specific_examples']['policies'].extend(policies[:5])
                        
                        # 数値データを抽出
                        numbers = re.findall(r'(\d+[％%]|\d+議席|\d+万?円|\d+兆?円)', section)
                        concrete_data['all_data_points'].extend(numbers)
                        
                        # 具体的な政策文を抽出
                        for policy in policies[:3]:
                            concrete_data['all_data_points'].append(policy.strip())
        
        elif response.data_source_type == DataSourceType.WEB_SEARCH and response.source_specific_content:
            concrete_data['specific_examples']['articles'] = []
            
            for key, value in response.source_specific_content.items():
                if 'search' in key and isinstance(value, list):
                    for result in value[:10]:  # 上位10件まで拡大
                        if isinstance(result, dict):
                            article_data = {
                                'title': result.get('title', ''),
                                'url': result.get('url', ''),
                                'source': result.get('url', '').split('/')[2] if result.get('url') else '',
                                'content': result.get('content', '')[:500]  # より長い内容を保持
                            }
                            concrete_data['specific_examples']['articles'].append(article_data)
                            
                            # 専門家の発言を抽出
                            import re
                            expert_quotes = re.findall(r'「([^」]{10,100})」', article_data['content'])
                            for quote in expert_quotes[:2]:
                                concrete_data['all_data_points'].append(f"専門家の見解：「{quote}」")
                            
                            # 記事タイトルを追加
                            if article_data['title']:
                                concrete_data['all_data_points'].append(
                                    f"『{article_data['title']}』（{article_data['source']}）"
                                )
        
        return concrete_data

    def _create_fully_integrated_response(self, winner: PersuasiveResponse, 
                                        loser: PersuasiveResponse,
                                        integrated_data: Dict[str, Any]) -> str:
        """完全に統合された応答を生成（2000文字×3段落版）"""
        
        winner_source = integrated_data['winner_source']
        loser_source = integrated_data['loser_source']
        winner_data = integrated_data['winner_data']
        loser_data = integrated_data['loser_data']
        
        # ユーザーの質問内容を取得（追加）
        user_question = integrated_data.get('user_question', '')
        
        # プロンプトを構築
        integration_prompt = f"""
        以下の指示に従って、約6000文字（各段落2000文字程度）の包括的な分析記事を作成してください。

        【ユーザーの質問】
        {user_question}

        【あなたのメインデータソース】
        {winner_source}

        【統合すべき追加データソース】
        {loser_source}

        【利用可能な具体的データ】
        
        ■{winner_source}からのデータ：
        {self._format_concrete_data_for_prompt(winner_data)}
        
        ■{loser_source}からのデータ：
        {self._format_concrete_data_for_prompt(loser_data)}

        【作成する文章の構造】

        第1段落（2000文字程度）：ユーザーの質問に対する関連データの包括的要約
        - ユーザーの質問「{user_question[:100]}...」に直接関連する情報を全データソースから抽出
        - 参政党の躍進に関連する具体的なデータ（動画タイトル、視聴回数、政策文書の内容、Web記事など）を優先的に提示
        - 各データソースから得られた参政党関連の重要な事実、数値、トレンドを網羅的に列挙
        - 単なるデータの羅列ではなく、質問への回答に向けた構造化された情報提示
        - 公明党など他党の情報は、参政党との比較において必要な場合のみ言及

        第2段落（2000文字程度）：マルチソース統合分析
        - 3つのデータソース（PDF、Web検索、動画データ）全ての情報を統合
        - 参政党躍進の要因を多角的に分析
        - PDFデータ：参政党と他党の政策比較、独自性の分析
        - 動画データ：参政党関連動画の視聴傾向、拡散パターン、具体的な動画タイトルと視聴回数
        - Web検索：専門家による参政党躍進の分析、メディア報道の論調
        - 各データソースがどのように相互補完し、参政党躍進の全体像を形成するかを詳述

        第3段落（2000文字程度）：深層的な統合分析と結論
        - 3つのデータソースを統合することで初めて見えてくる参政党躍進の真の要因
        - 表面的な要因（SNS戦略）と深層的な要因（社会的背景、有権者心理）の区別
        - 参政党が「反ワクチンナラティブや排外主義的なナラティブを融合して党独自の新しいストーリーを生成できた理由」への具体的な回答
        - CMVの投稿者の「SNSやショート動画の戦略」という見解への建設的な反論
        - 各データソースから得られた証拠を統合した説得力のある結論

        【重要な制約】
        - 第1段落は必ずユーザーの質問（参政党の躍進）に直接関連する情報から始める
        - 公明党などの他党の政策は、参政党との比較や背景説明に必要な場合のみ言及
        - 各段落は必ず1800-2200文字程度にする
        - 具体的なデータ（数値、タイトル、名前等）を豊富に含める
        - 同じ内容の繰り返しを避ける
        - 「です・ます」調で統一
        - 段落間は明確に区切る（改行2つ）

        上記の指示に従って、6000文字程度の包括的な分析記事を作成してください。
        """
        
        try:
            if hasattr(self, 'persuasion_optimizer') and self.persuasion_optimizer:
                api_config = self.persuasion_optimizer.api_config
                
                if api_config.openai_client:
                    response = api_config.openai_client.chat.completions.create(
                        model="gpt-4.1-2025-04-14",
                        messages=[
                            {"role": "system", "content": "あなたは日本の著名な政治アナリストです。詳細なデータ分析と深い洞察を提供してください。"},
                            {"role": "user", "content": integration_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=8000  # 大幅に増加
                    )
                    integrated_content = response.choices[0].message.content
                    logger.info(f"Created fully integrated response: {len(integrated_content)} chars")
                    return integrated_content
        except Exception as e:
            logger.error(f"Failed to create integrated response: {e}")
        
        # フォールバック：手動で統合
        return self._manually_integrate_response(winner, loser, integrated_data)

    def _compare_responses_with_learning(self, resp1: PersuasiveResponse, resp2: PersuasiveResponse) -> PersuasiveResponse:
        """強化された知識統合を伴う応答比較（スコア低下防止版）"""
        
        # まず通常の比較で勝者を決定
        if resp1.persuasion_score > resp2.persuasion_score:
            basic_winner = resp1
            loser = resp2
        elif resp2.persuasion_score > resp1.persuasion_score:
            basic_winner = resp2
            loser = resp1
        else:
            basic_winner = random.choice([resp1, resp2])
            loser = resp2 if basic_winner == resp1 else resp1
        
        # 元のスコアと内容を保存
        original_score = basic_winner.persuasion_score
        original_content = basic_winner.content
        original_detailed_score = basic_winner.detailed_score
        
        # スコアが既に高い場合（例：17.00）は学習をスキップ
        SCORE_CEILING_THRESHOLD = 16.5  # このスコア以上の場合は慎重に
        if original_score >= SCORE_CEILING_THRESHOLD:
            logger.info(f"High score detected ({original_score:.2f}), applying cautious refinement")
        
        # 勝者と敗者のデータソースタイプを確認
        winner_source = basic_winner.data_source_type
        loser_source = loser.data_source_type
        
        logger.info(f"Knowledge integration: {basic_winner.model_used} ({winner_source.value if winner_source else 'unknown'}) learning from {loser.model_used} ({loser_source.value if loser_source else 'unknown'})")
        
        # 異なるデータソースの場合、知識を統合
        if winner_source and loser_source and winner_source != loser_source:
            # 強化された知識統合を試みる
            improved_content = self._integrate_cross_source_knowledge_enhanced(basic_winner, loser)
            
            # 改善された応答を作成
            improved_winner = PersuasiveResponse(
                content=improved_content,
                treatment_condition=basic_winner.treatment_condition,
                persuasion_score=0.0,
                model_used=basic_winner.model_used,
                generation_params=basic_winner.generation_params,
                user_profile=basic_winner.user_profile,
                pdf_context=basic_winner.pdf_context,
                data_source_type=basic_winner.data_source_type,
                source_specific_content=basic_winner.source_specific_content,
                tournament_round=basic_winner.tournament_round
            )
            
            # 学習履歴を引き継ぎ
            improved_winner.learning_history = getattr(basic_winner, 'learning_history', []).copy()
            
            # 改善後のスコアを計算
            improved_winner.detailed_score = self._score_response_with_details(improved_winner)
            improved_winner.persuasion_score = improved_winner.detailed_score.total
            
            logger.info(f"Score after knowledge integration: {improved_winner.persuasion_score:.2f} (original: {original_score:.2f})")
            
            # === 重要な修正: スコアが下がった場合の処理 ===
            MINIMUM_IMPROVEMENT = 0.1  # 最小改善スコア
            
            # スコアが元より低い、または改善が不十分な場合
            if improved_winner.persuasion_score < original_score:
                logger.warning(f"Refinement decreased score from {original_score:.2f} to {improved_winner.persuasion_score:.2f}, reverting to original")
                
                # 元の応答を維持
                basic_winner.learning_history.append({
                    'learned_from': f"{loser.model_used} ({loser_source.value})",
                    'original_score': original_score,
                    'attempted_score': improved_winner.persuasion_score,
                    'opponent_score': loser.persuasion_score,
                    'cross_source_learning': True,
                    'source_types': {'winner': winner_source.value, 'loser': loser_source.value},
                    'status': 'rejected_score_decrease',
                    'integration_type': 'skipped'
                })
                
                return basic_winner
                
            # 高スコアの場合、改善が小さくても慎重に判断
            elif original_score >= SCORE_CEILING_THRESHOLD and improved_winner.persuasion_score < original_score + MINIMUM_IMPROVEMENT:
                logger.info(f"High score refinement: insufficient improvement ({improved_winner.persuasion_score:.2f} vs {original_score:.2f}), keeping original")
                
                basic_winner.learning_history.append({
                    'learned_from': f"{loser.model_used} ({loser_source.value})",
                    'original_score': original_score,
                    'attempted_score': improved_winner.persuasion_score,
                    'opponent_score': loser.persuasion_score,
                    'cross_source_learning': True,
                    'source_types': {'winner': winner_source.value, 'loser': loser_source.value},
                    'status': 'rejected_insufficient_improvement',
                    'integration_type': 'skipped'
                })
                
                return basic_winner
            
            # スコアが改善された場合のみ、改善された応答を使用
            else:
                logger.info(f"Refinement successful: score improved from {original_score:.2f} to {improved_winner.persuasion_score:.2f}")
                
                improved_winner.learning_history.append({
                    'learned_from': f"{loser.model_used} ({loser_source.value})",
                    'original_score': original_score,
                    'improved_score': improved_winner.persuasion_score,
                    'opponent_score': loser.persuasion_score,
                    'cross_source_learning': True,
                    'source_types': {'winner': winner_source.value, 'loser': loser_source.value},
                    'status': 'success',
                    'integration_type': 'enhanced'
                })
                
                # 統合された知識を蓄積
                self.integrated_knowledge[winner_source.value].append({
                    'from': loser_source.value,
                    'content': self._extract_key_insights(loser.content, loser_source)
                })
                
                return improved_winner
        
        # 同じデータソースの場合も、蓄積された知識を活用（ただし慎重に）
        elif winner_source:
            if self.integrated_knowledge.get(winner_source.value):
                improved_content = self._add_accumulated_knowledge(basic_winner.content, winner_source)
                improved_winner = PersuasiveResponse(
                    content=improved_content,
                    treatment_condition=basic_winner.treatment_condition,
                    persuasion_score=0.0,
                    model_used=basic_winner.model_used,
                    generation_params=basic_winner.generation_params,
                    user_profile=basic_winner.user_profile,
                    pdf_context=basic_winner.pdf_context,
                    data_source_type=basic_winner.data_source_type,
                    source_specific_content=basic_winner.source_specific_content,
                    tournament_round=basic_winner.tournament_round
                )
                improved_winner.detailed_score = self._score_response_with_details(improved_winner)
                improved_winner.persuasion_score = improved_winner.detailed_score.total
                
                # スコアが改善された場合のみ使用
                if improved_winner.persuasion_score > original_score:
                    logger.info(f"Accumulated knowledge improved score from {original_score:.2f} to {improved_winner.persuasion_score:.2f}")
                    return improved_winner
                else:
                    logger.info(f"Accumulated knowledge did not improve score, keeping original")
        
        return basic_winner

    def _integrate_cross_source_knowledge_enhanced(self, winner: PersuasiveResponse, loser: PersuasiveResponse) -> str:
        """強化された異なるデータソースの知識統合（品質保証版）"""
        winner_content = winner.content
        loser_content = loser.content
        winner_source = winner.data_source_type
        loser_source = loser.data_source_type
        
        # 高スコアの応答の場合は、より慎重な統合を行う
        if winner.persuasion_score >= 16.5:
            logger.info(f"High-score content detected ({winner.persuasion_score:.2f}), applying conservative integration")
            
            # LLMによる洗練処理を優先
            loser_insights = self._extract_key_insights(loser_content, loser_source)
            if loser_insights:
                refined_content = self._refine_response_with_llm(
                    content=winner_content,
                    winner_source=winner_source,
                    loser_source=loser_source,
                    loser_insights=loser_insights,
                    winner_response=winner
                )
                
                # 洗練された内容の品質をチェック
                if self._validate_content_structure(refined_content):
                    # 洗練された内容が元の内容と大きく異ならないかチェック
                    if self._is_content_coherent(winner_content, refined_content):
                        return refined_content
                    else:
                        logger.warning("Refined content diverged too much from original, using conservative approach")
            
            # LLM refinementが失敗した場合は、最小限の変更のみ
            return self._minimal_integration(winner_content, loser_insights, winner_source, loser_source)
        
        # 通常スコアの場合は、既存の処理を使用
        else:
            # 既存の_integrate_cross_source_knowledge_enhancedの処理
            # （前回提供したコードと同じ）
            winner_content = self._remove_duplicate_content(winner_content)
            loser_insights = self._extract_key_insights(loser_content, loser_source)
            
            if loser_insights:
                refined_content = self._refine_response_with_llm(
                    content=winner_content,
                    winner_source=winner_source,
                    loser_source=loser_source,
                    loser_insights=loser_insights,
                    winner_response=winner
                )
                
                if self._validate_content_structure(refined_content):
                    return refined_content
            
            # フォールバック処理
            return self._fallback_integration(winner, loser, winner_content, loser_content, 
                                            winner_source, loser_source, loser_insights)


    def _minimal_integration(self, winner_content: str, loser_insights: List[str], 
                            winner_source: DataSourceType, loser_source: DataSourceType) -> str:
        """最小限の統合（高スコアコンテンツ用）"""
        # 勝者の内容をほぼそのまま維持し、最小限の洞察のみ追加
        if not loser_insights:
            return winner_content
        
        # 最も重要な1つの洞察のみを選択
        most_important_insight = loser_insights[0] if loser_insights else ""
        
        # 段落の区切りを探す
        paragraphs = winner_content.split('\n\n')
        
        # 適切な位置に1文だけ追加
        if len(paragraphs) > 2:
            # 中間位置に簡潔な参照を追加
            insertion_point = len(paragraphs) // 2
            connector = self._get_minimal_connector(winner_source, loser_source)
            minimal_addition = f"{connector}{most_important_insight}。"
            
            # 既存の段落に自然に統合
            paragraphs[insertion_point] = paragraphs[insertion_point] + minimal_addition
        
        return '\n\n'.join(paragraphs)


    def _is_content_coherent(self, original: str, refined: str) -> bool:
        """洗練された内容が元の内容と一貫性があるかチェック"""
        # 文字数の大幅な変化をチェック
        len_ratio = len(refined) / len(original) if len(original) > 0 else 1.0
        if len_ratio < 0.7 or len_ratio > 1.3:
            logger.warning(f"Content length changed significantly: {len(original)} -> {len(refined)}")
            return False
        
        # 主要なキーワードが維持されているかチェック
        original_keywords = set(self._extract_keywords(original))
        refined_keywords = set(self._extract_keywords(refined))
        
        # 元のキーワードの70%以上が保持されているか
        retention_rate = len(original_keywords.intersection(refined_keywords)) / len(original_keywords) if original_keywords else 0
        if retention_rate < 0.7:
            logger.warning(f"Too many keywords lost: retention rate = {retention_rate:.2f}")
            return False
        
        return True


    def _extract_keywords(self, text: str) -> List[str]:
        """テキストから重要なキーワードを抽出"""
        import re
        # 重要な名詞や数値を抽出
        keywords = []
        
        # 政党名
        party_names = re.findall(r'(自民党|公明党|参政党|国民民主党|立憲民主党|日本維新の会)', text)
        keywords.extend(party_names)
        
        # 数値データ
        numbers = re.findall(r'\d+[％%]|\d+議席|\d+万?回|\d+億?円', text)
        keywords.extend(numbers)
        
        # 重要な政策用語
        policy_terms = ['政策', '選挙', 'SNS', '動画', 'YouTube', 'PDF', 'Web', '分析', '要因', '背景']
        for term in policy_terms:
            if term in text:
                keywords.append(term)
        
        return keywords


    def _get_minimal_connector(self, winner_source: DataSourceType, loser_source: DataSourceType) -> str:
        """最小限の接続表現を返す"""
        connectors = {
            (DataSourceType.PDF, DataSourceType.WEB_SEARCH): "報道分析も踏まえると、",
            (DataSourceType.PDF, DataSourceType.VIDEO_DATA): "動画トレンドを考慮すると、",
            (DataSourceType.WEB_SEARCH, DataSourceType.PDF): "政策文書の観点では、",
            (DataSourceType.WEB_SEARCH, DataSourceType.VIDEO_DATA): "SNS動向を見ると、",
            (DataSourceType.VIDEO_DATA, DataSourceType.PDF): "政策面では、",
            (DataSourceType.VIDEO_DATA, DataSourceType.WEB_SEARCH): "メディア分析によれば、"
        }
        return connectors.get((winner_source, loser_source), "また、")


    def _fallback_integration(self, winner: PersuasiveResponse, loser: PersuasiveResponse,
                            winner_content: str, loser_content: str,
                            winner_source: DataSourceType, loser_source: DataSourceType,
                            loser_insights: List[str]) -> str:
        """フォールバック統合処理（既存のロジック）"""
        # 既存の統合ロジックをここに配置
        # （前回提供したコードの該当部分）
        existing_elements = set()
        paragraphs = winner_content.split('\\n\\n') if '\\n\\n' in winner_content else [winner_content]
        
        for para in paragraphs:
            sentences = para.split('。')
            for sentence in sentences:
                if sentence.strip():
                    existing_elements.add(sentence.strip())
        
        integrated_response = []
        
        if paragraphs:
            intro = paragraphs[0]
            if winner_source and loser_source:
                source_mention = self._create_source_mention(winner_source, loser_source)
                if source_mention not in intro:
                    intro = self._enhance_introduction(intro, source_mention)
            integrated_response.append(intro)
        
        if len(paragraphs) > 1:
            for para in paragraphs[1:]:
                if not self._is_duplicate_paragraph(para, integrated_response):
                    integrated_response.append(para)
        
        if loser_insights:
            new_insights = []
            for insight in loser_insights:
                if insight not in existing_elements:
                    new_insights.append(insight)
                    existing_elements.add(insight)
            
            if new_insights:
                insight_paragraph = self._create_insight_paragraph(new_insights, loser_source, winner_source)
                
                if not self._is_duplicate_paragraph(insight_paragraph, integrated_response):
                    insertion_point = min(len(integrated_response) // 2 + 1, len(integrated_response))
                    integrated_response.insert(insertion_point, insight_paragraph)
        
        conclusion = self._create_integrated_conclusion(winner_source, loser_source, winner_content, loser_content)
        
        has_similar_conclusion = False
        for para in integrated_response:
            if self._calculate_similarity(conclusion, para) > 0.7:
                has_similar_conclusion = True
                break
        
        if not has_similar_conclusion:
            integrated_response.append(conclusion)
        
        final_content = '\\n\\n'.join(integrated_response)
        final_content = self._remove_duplicate_content(final_content)
        final_content = self._adjust_content_length_improved(final_content, target_length=500)
        
        return final_content
    
    def _extract_key_insights(self, content: str, source: DataSourceType) -> List[str]:
        """データソースから重要な洞察を抽出（具体的情報を含む）"""
        insights = []
        sentences = content.split('。')
        
        # データソース特有の重要情報を優先的に抽出
        priority_keywords = {
            DataSourceType.PDF: ['政策', '公約', 'マニフェスト', '制度', '改革', '提案'],
            DataSourceType.WEB_SEARCH: ['報道', '専門家', '分析', '最新', 'ニュース', '動向'],
            DataSourceType.VIDEO_DATA: ['視聴回数', '動画', 'トレンド', 'SNS', 'バイラル', '再生']
        }
        
        source_keywords = priority_keywords.get(source, [])
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # データソース特有のキーワードを含む文を優先
            if any(keyword in sentence for keyword in source_keywords):
                # 数値データも含む場合は特に重要
                if any(char.isdigit() for char in sentence):
                    insights.insert(0, sentence)  # 最優先
                else:
                    insights.append(sentence)
            # 一般的な重要情報
            elif any(keyword in sentence for keyword in ['要因', '背景', '影響', '結果', '理由']):
                insights.append(sentence)
        
        # 最大3つの洞察を返す
        return insights[:3]
    
    def _create_source_mention(self, source1: DataSourceType, source2: DataSourceType) -> str:
        """データソースの言及を作成"""
        source_names = {
            DataSourceType.PDF: "政策文書分析",
            DataSourceType.WEB_SEARCH: "最新のメディア報道",
            DataSourceType.VIDEO_DATA: "SNS動画トレンド",
            DataSourceType.CSV: "X（Twitter）投稿分析"  # 新規追加
        }
        return f"{source_names.get(source1, 'データ')}と{source_names.get(source2, 'データ')}"
    
    def _enhance_introduction(self, intro: str, source_mention: str) -> str:
        """導入部を強化"""
        if len(intro) > 100:
            # 既存の導入が長い場合は、データソース言及を追加
            return f"{intro}本分析では{source_mention}を統合的に検討します。"
        else:
            return intro
    
    def _create_insight_paragraph(self, insights: List[str], source: DataSourceType, winner_source: DataSourceType) -> str:
        """洞察を段落として構成"""
        source_intro = {
            DataSourceType.PDF: "政策文書の分析から",
            DataSourceType.WEB_SEARCH: "最新の報道分析によると",
            DataSourceType.VIDEO_DATA: "SNS動画の傾向を見ると"
        }
        
        winner_relation = {
            DataSourceType.PDF: "この政策的観点は",
            DataSourceType.WEB_SEARCH: "このメディア分析は",
            DataSourceType.VIDEO_DATA: "この世論動向は"
        }
        
        paragraph = f"{source_intro.get(source, 'さらに')}、"
        
        if insights:
            paragraph += f"{insights[0]}。"
            if len(insights) > 1:
                paragraph += f"また、{insights[1]}。"
        
        paragraph += f"{winner_relation.get(winner_source, 'これは')}先述の分析と相互に補完し合い、より包括的な理解を可能にします。"
        
        return paragraph
    
    def _create_integrated_conclusion(self, source1: DataSourceType, source2: DataSourceType, 
                                    content1: str, content2: str) -> str:
        """統合的な結論を作成"""
        # 両方のコンテンツから数値データを抽出
        import re
        numbers1 = re.findall(r'\d+[％%]|\d+議席|\d+万?回', content1)
        numbers2 = re.findall(r'\d+[％%]|\d+議席|\d+万?回', content2)
        
        conclusion = "以上の分析から、参政党の躍進は"
        
        source_perspectives = {
            DataSourceType.PDF: "政策面での戦略的アプローチ",
            DataSourceType.WEB_SEARCH: "メディア戦略の巧みさ",
            DataSourceType.VIDEO_DATA: "SNSでの効果的な拡散"
        }
        
        factors = []
        if source1 in source_perspectives:
            factors.append(source_perspectives[source1])
        if source2 in source_perspectives:
            factors.append(source_perspectives[source2])
        
        if factors:
            conclusion += "、".join(factors) + "が"
        
        # 数値データがあれば含める
        if numbers1 or numbers2:
            sample_numbers = (numbers1[:1] + numbers2[:1])[:2]
            if sample_numbers:
                conclusion += f"（{', '.join(sample_numbers)}という具体的データが示すように）"
        
        conclusion += "複合的に作用した結果と言えるでしょう。単一の要因ではなく、これらの要素が相互に影響し合い、シナジー効果を生み出したことが真の成功要因です。"
        
        return conclusion
    
    def _is_duplicate_paragraph(self, new_para: str, existing_paras: List[str]) -> bool:
        """段落が既存の段落と重複しているかチェック"""
        for existing_para in existing_paras:
            similarity = self._calculate_similarity(new_para, existing_para)
            if similarity > 0.7:  # 70%以上類似していたら重複とみなす
                return True
        return False

    def _adjust_content_length_improved(self, content: str, target_length: int = 500) -> str:
        """コンテンツの長さを調整（重複を避ける改善版）"""
        current_length = len(content)
        
        # 目標範囲内ならそのまま返す
        if 400 <= current_length <= 600:
            return content
        
        if current_length < 400:
            # 短すぎる場合は内容を補強（ただし重複を避ける）
            additional_text = self._generate_additional_insight(content)
            if additional_text and additional_text not in content:
                content += f"\\n\\n{additional_text}"
        
        elif current_length > 600:
            # 長すぎる場合は段落を調整
            paragraphs = content.split('\\n\\n')
            
            # 重要度スコアを計算
            paragraph_scores = []
            for i, para in enumerate(paragraphs):
                score = 0
                # 最初と最後の段落は重要
                if i == 0 or i == len(paragraphs) - 1:
                    score += 10
                # データを含む段落は重要
                if any(char.isdigit() for char in para):
                    score += 5
                # キーワードを含む段落は重要
                important_keywords = ['要因', '背景', '結論', '分析', '統合']
                for keyword in important_keywords:
                    if keyword in para:
                        score += 2
                
                paragraph_scores.append((score, i, para))
            
            # スコアでソートして上位を選択
            paragraph_scores.sort(key=lambda x: x[0], reverse=True)
            
            # 必要な段落数を計算
            target_paragraphs = 3
            selected_indices = sorted([x[1] for x in paragraph_scores[:target_paragraphs]])
            
            # 選択された段落を元の順序で結合
            shortened = []
            for idx in selected_indices:
                para = paragraphs[idx]
                # 各段落も長すぎる場合は短縮
                if len(para) > 200:
                    sentences = para.split('。')
                    # 重要な文を選択（最初の2文と数値を含む文）
                    important_sentences = sentences[:2]
                    for sent in sentences[2:]:
                        if any(char.isdigit() for char in sent):
                            important_sentences.append(sent)
                            break
                    para = '。'.join(important_sentences) + '。'
                shortened.append(para)
            
            content = '\\n\\n'.join(shortened)
        
        return content

    def _generate_additional_insight(self, content: str) -> str:
        """追加の洞察を生成（既存内容と重複しない）"""
        # コンテンツに含まれていない新しい視点を追加
        if '多角的' not in content and '総合的' not in content:
            return "このような多角的な分析により、現象の本質的な理解が可能となります。"
        elif 'シナジー' not in content and '相乗効果' not in content:
            return "各要素が相乗効果を生み出し、予想を超える影響力を発揮したと考えられます。"
        elif '今後' not in content and '将来' not in content:
            return "今後の政治動向を予測する上でも、これらの要因は重要な示唆を与えています。"
        else:
            return ""
    
    def _add_accumulated_knowledge(self, content: str, source: DataSourceType) -> str:
        """蓄積された知識を追加"""
        accumulated = self.integrated_knowledge.get(source.value, [])
        if not accumulated:
            return content
        
        # 最新の蓄積知識を選択
        latest_knowledge = accumulated[-1] if accumulated else None
        if latest_knowledge:
            insight = latest_knowledge.get('content', [])
            if insight and isinstance(insight, list) and insight:
                # コンテンツの中盤に知識を挿入
                paragraphs = content.split('\n\n')
                insertion_point = len(paragraphs) // 2
                additional = f"関連する観点として、{insight[0]}という視点も重要です。"
                paragraphs.insert(insertion_point, additional)
                return '\n\n'.join(paragraphs)
        
        return content

# ====================
# Persuasion Optimizer
# ====================
class PersuasionOptimizer:
    """Core optimization algorithm for generating persuasive responses"""

    def __init__(self, api_config: APIConfig, tournament_config: TournamentConfig = None):
        self.api_config = api_config
        self.config = tournament_config or TournamentConfig()
        self.models = ['gpt-4.1'] if api_config.openai_client else ['fallback']
        self.community_aligned_model = 'gpt-4.1-finetuned'
        self.pdf_cache = {}
        self.json_cache = {}
        self.search_cache = {}
        self.text_cleaner = PDFTextCleaner()
        self.csv_analyzer = TwitterCSVAnalyzer()
        self.video_analyzer = VideoDataAnalyzer()
        self.tournament_selector = TournamentSelector(persuasion_optimizer=self, config=self.config)
        self.executor = ThreadPoolExecutor(max_workers=10) if self.config.parallel_processing else None
    
    def get_current_datetime_context(self) -> str:
        """現在の日時情報をLLM用のコンテキストとして生成"""
        now = datetime.now()
        weekdays = ['月曜日', '火曜日', '水曜日', '木曜日', '金曜日', '土曜日', '日曜日']
        weekday = weekdays[now.weekday()]
        
        return f"""
【重要：現在の日時情報】
現在日時: {now.strftime('%Y年%m月%d日')} {weekday} {now.strftime('%H時%M分')}
今日は{now.year}年{now.month}月{now.day}日です。
"""

    def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """Web検索を実行して関連情報を取得"""
        cache_key = f"{query}_{max_results}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        if not self.api_config.tavily_client:
            logger.warning("Tavily API key not configured. Skipping web search.")
            return []
        
        try:
            response = self.api_config.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=["luup.co.jp", "asahi.com", "nikkei.com", "nhk.or.jp", 
                            "mainichi.jp", "yomiuri.co.jp", "itmedia.co.jp", 
                            "techcrunch.com", "response.jp"],
                include_raw_content=True
            )
            
            results = []
            for result in response.get('results', []):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'published_date': result.get('published_date', ''),
                    'score': result.get('score', 0)
                })
            
            self.search_cache[cache_key] = results
            logger.info(f"Web search for '{query}' returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
    
    def analyze_pdf_report(self, pdf_path: str) -> Dict[str, Any]:
        """Luup分析レポートPDFを詳細分析"""
        pdf_content = self.load_pdf_content(pdf_path)
        
        if not pdf_content:
            return {'error': 'Failed to load PDF'}
        
        # クリーナーを使用して重要情報を抽出
        cleaner = PDFTextCleaner()
        critical_users = cleaner.extract_critical_users(pdf_content)
        risk_clusters = cleaner.extract_risk_clusters(pdf_content)
        
        # クラスター分析
        cluster_analysis = {
            'identified_clusters': [],
            'critical_risks': [],
            'key_influencers': critical_users,
            'timeline_events': []
        }
        
        # 最大リスククラスターの特定
        if '4_マンションの避難器具設置と消防法違反' in pdf_content:
            cluster_analysis['critical_risks'].append({
                'type': 'fire_safety_violation',
                'severity': 'critical',
                'description': '他のクラスターと比べて圧倒的に世間からの議論を呼んでいる',
                'burst_timing': '10月',
                'trigger': 'awakku氏の投稿、ねとらぼニュース掲載'
            })
        
        # インフルエンサー分析
        if 'ちんにい' in pdf_content or '@chinniisan' in pdf_content:
            cluster_analysis['key_influencers'].append({
                'account': '@chinniisan',
                'alias': 'ちんにい',
                'type': 'anti_luup_leader',
                'activity': '#反Luup教の中心人物',
                'pagerank': '岡井さんより上位',
                'strategy_needed': '岡井さんの発言を引用RTして活動を増幅'
            })
        
        # タイムライン分析
        import re
        dates = re.findall(r'(\d{1,2})月', pdf_content)
        for date in set(dates):
            if date == '10':
                cluster_analysis['timeline_events'].append({
                    'month': '10月',
                    'event': '消防法違反クラスターのバースト発生',
                    'severity': 'critical'
                })
            elif date == '9' or date == '8':
                cluster_analysis['timeline_events'].append({
                    'month': f'{date}月',
                    'event': '駐輪問題の議論（影響は限定的）',
                    'severity': 'low'
                })
        
        return cluster_analysis
    
    def generate_crisis_response_strategy(self, pdf_analysis: Dict[str, Any]) -> str:
        """PDFレポートの分析に基づく危機対応戦略を生成"""
        strategy = """
        【Luup危機対応戦略 - PDFレポート分析に基づく】
        
        1. 最優先対応事項：
        - 消防法違反クラスター（マンション避難器具問題）への即座の対応
        - 10月のバースト事例を教訓とした予防的措置
        
        2. インフルエンサー対策：
        - @chinniisan（ちんにい）等の#反Luup教アカウントの監視
        - 岡井CEOの発言が引用RTされるリスクを考慮した慎重な発信
        - 権威批判（東大卒、天下り）を誘発しないコミュニケーション
        
        3. ステークホルダー別対応：
        - 一般市民：ポート設置への本質的不満への対処
        - メディア：ねとらぼ等のニュースメディアとの関係構築
        - 批判的ユーザー：建設的対話の機会創出
        
        4. タイミング戦略：
        - バースト発生の予兆を早期検知
        - 自転車赤切符ニュース等の関連報道時の迅速対応
        
        5. コンテンツ戦略：
        - ポジティブキャンペーン（銭湯、CM）の継続
        - 安全対策の可視化と積極的発信
        """
        
        return strategy

    def format_csv_analysis(self, analysis: Dict[str, Any]) -> str:
        """CSV分析結果を文字列にフォーマット"""
        if not analysis or 'error' in analysis:
            return ""
        
        formatted = "\n\n【X（Twitter）データ分析結果】\n"
        formatted += f"分析対象投稿数: {analysis['total_posts']:,}件\n"
        formatted += f"期間: {analysis['date_range']['start']} から {analysis['date_range']['end']}\n"
        
        # センチメント分析
        if 'sentiment_analysis' in analysis:
            sentiment = analysis['sentiment_analysis']
            formatted += "\n■ センチメント分析:\n"
            for sent_type, data in sentiment.get('by_sentiment', {}).items():
                formatted += f"  {sent_type}: {data['count']}件 ({data['percentage']:.1f}%)\n"
                formatted += f"    平均いいね: {data['avg_likes']:.1f}, 平均リポスト: {data['avg_reposts']:.1f}\n"
        
        # インフルエンサー分析
        if 'top_influencers' in analysis:
            formatted += "\n■ 影響力の高いアカウント:\n"
            for i, influencer in enumerate(analysis['top_influencers'][:3], 1):
                formatted += f"  {i}. {influencer['author_name']} (@{influencer['author_handle']})\n"
                formatted += f"     投稿数: {influencer['post_count']}, 総いいね: {influencer['total_likes']:,}\n"
                if 'top_post' in influencer:
                    post_text = influencer['top_post']['text'][:100] + '...' if len(influencer['top_post']['text']) > 100 else influencer['top_post']['text']
                    formatted += f"     代表的投稿: 「{post_text}」\n"
        
        # キートピック
        if 'key_topics' in analysis:
            formatted += "\n■ 主要トピック:\n"
            for topic in analysis['key_topics'][:5]:
                formatted += f"  - {topic['keyword']} (出現回数: {topic['frequency']})\n"
        
        return formatted

    def format_search_results(self, search_results: List[Dict]) -> str:
        """検索結果を文字列にフォーマット"""
        if not search_results:
            return ""
        
        formatted = "\n\n【Web検索結果】\n"
        for i, result in enumerate(search_results[:3], 1):
            formatted += f"\n{i}. {result['title']}\n"
            formatted += f"   出典: {result['url']}\n"
            if result.get('published_date'):
                formatted += f"   日付: {result['published_date']}\n"
            formatted += f"   内容: {result['content'][:200]}...\n"
        
        return formatted

    def load_pdf_content(self, pdf_path: str, summarize: bool = True) -> str:
        """日本語PDFファイルの内容を読み込んでテキストとして返す"""
        if pdf_path in self.pdf_cache:
            return self.pdf_cache[pdf_path]
        
        text = ""
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could get FontBBox")
                warnings.filterwarnings("ignore", category=UserWarning)
                
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    
                    if text.strip():
                        logger.info(f"Successfully extracted text using pdfplumber from {pdf_path}")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        if not text.strip():
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                if text.strip():
                    logger.info(f"Successfully extracted text using PyPDF2 from {pdf_path}")
            except Exception as e:
                logger.error(f"PyPDF2 extraction also failed: {e}")
                return ""
        
        if text.strip():
            text = self.text_cleaner.clean_text(text)
            self.pdf_cache[pdf_path] = text
            return self.pdf_cache[pdf_path]
        else:
            logger.error(f"No text could be extracted from {pdf_path}")
            return ""
    
    def get_relevant_pdf_sections(self, pdf_content: str, query: str, max_sections: int = 3) -> List[str]:
        """PDFコンテンツから関連するセクションを抽出"""
        if not pdf_content or not query:
            return []
        
        sentence_endings = r'[。！？]'
        sentences = re.split(sentence_endings, pdf_content)
        
        sections = []
        for i in range(0, len(sentences), 3):
            section = '。'.join(sentences[i:i+3])
            if section.strip() and len(section) > 30:
                sections.append(section + '。')
        
        query_keywords = []
        
        kanji_kata_pattern = r'[一-龥ァ-ヶー]{3,}'
        query_keywords.extend(re.findall(kanji_kata_pattern, query))
        
        hiragana_pattern = r'[ぁ-ん]{4,}'
        query_keywords.extend(re.findall(hiragana_pattern, query))
        
        political_keywords = [
            '選挙', '政党', '議席', '投票', '政権', '与党', '野党',
            '自民党', '公明党', '参政党', 'SNS', 'カルト', '宗教'
        ]
        
        for keyword in political_keywords:
            if keyword in query:
                query_keywords.append(keyword)
        
        query_keywords = list(set(query_keywords))
        
        scored_sections = []
        for section in sections:
            score = 0
            
            for keyword in query_keywords:
                score += section.count(keyword) * 2
                
                if len(keyword) >= 3:
                    for i in range(len(keyword) - 2):
                        partial = keyword[i:i+3]
                        score += section.count(partial) * 0.5
            
            length_bonus = min(len(section) / 100, 2.0)
            score += length_bonus
            
            if re.search(r'\d+[％%]|\d+議席|\d+票', section):
                score += 1.0
            
            if score > 0:
                scored_sections.append((score, section))
        
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        
        result_sections = []
        for i, (score, section) in enumerate(scored_sections[:max_sections]):
            result_sections.append(f"【関連箇所{i+1}】\n{section}")
        
        return result_sections
    
    def analyze_json_data(self, json_data: Any, query: str) -> Dict[str, Any]:
        """YouTube/TikTok動画データを詳細分析（強化版）"""
        # VideoDataAnalyzerを使用
        if not hasattr(self, 'video_analyzer'):
            self.video_analyzer = VideoDataAnalyzer()
        
        # JSONデータを一時ファイルに保存
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data if isinstance(json_data, list) else [json_data], f)
            temp_path = f.name
        
        try:
            # 包括的な分析を実行
            analysis = self.video_analyzer.generate_comprehensive_analysis(temp_path, query)
            return analysis
        finally:
            # 一時ファイルを削除
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def generate_responses(self, 
                        post: RedditPost, 
                        treatment: TreatmentCondition,
                        user_profile: Optional[UserProfile] = None,
                        num_candidates: int = 128,
                        pdf_references: List[str] = None,
                        json_data: Optional[Dict] = None) -> List[PersuasiveResponse]:
        """並列処理で複数のエージェントを生成（改善案3: パフォーマンスの最適化）"""
        
        # データソースタイプのリスト
        data_sources = [DataSourceType.PDF, DataSourceType.WEB_SEARCH, DataSourceType.VIDEO_DATA]
        
        # 各データソースあたりのエージェント数を計算
        agents_per_source = num_candidates // len(data_sources)
        remaining_agents = num_candidates % len(data_sources)
        
        logger.info(f"Generating {num_candidates} agents with single model (GPT-4) - Parallel: {self.config.parallel_processing}")
        
        if self.config.parallel_processing and self.executor:
            # 並列処理で生成
            futures = []
            
            for source_idx, data_source in enumerate(data_sources):
                source_agents = agents_per_source
                if source_idx < remaining_agents:
                    source_agents += 1
                
                for i in range(source_agents):
                    future = self.executor.submit(
                        self._generate_source_specific_response,
                        post, treatment, self.models[0], data_source,
                        user_profile, pdf_references, json_data
                    )
                    futures.append(future)
            
            # 結果を収集
            responses = []
            for future in as_completed(futures):
                try:
                    response = future.result(timeout=30)
                    if response:
                        responses.append(response)
                except Exception as e:
                    logger.error(f"Failed to generate agent: {str(e)}")
                    # フォールバック応答を追加
                    responses.append(self._create_fallback_response(post, treatment, user_profile, pdf_references))
            
        else:
            # 従来の逐次処理
            responses = []
            for source_idx, data_source in enumerate(data_sources):
                source_agents = agents_per_source
                if source_idx < remaining_agents:
                    source_agents += 1
                
                logger.info(f"Generating {source_agents} agents for {data_source.value}")
                
                for i in range(source_agents):
                    try:
                        response = self._generate_source_specific_response(
                            post, treatment, self.models[0], data_source,
                            user_profile, pdf_references, json_data
                        )
                        
                        if response:
                            responses.append(response)
                        
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Failed to generate agent for {data_source.value}: {str(e)}")
                        responses.append(self._create_fallback_response(post, treatment, user_profile, pdf_references, data_source))
        
        logger.info(f"Total {len(responses)} agents generated")
        return responses
    
    def _generate_source_specific_response(self, 
                                         post: RedditPost,
                                         treatment: TreatmentCondition,
                                         model: str,
                                         data_source: DataSourceType,
                                         user_profile: Optional[UserProfile] = None,
                                         pdf_references: List[str] = None,
                                         json_data: Optional[Dict] = None) -> PersuasiveResponse:
        """特定のデータソースのみを使用して応答を生成"""
        
        datetime_context = self.get_current_datetime_context()
        source_context = ""
        source_specific_content = {}
        
        if data_source == DataSourceType.PDF:
            if pdf_references:
                for i, pdf_path in enumerate(pdf_references[:3]):
                    pdf_content = self.load_pdf_content(pdf_path, summarize=True)
                    if pdf_content:
                        relevant_sections = self.get_relevant_pdf_sections(pdf_content, post.title + " " + post.body, max_sections=2)
                        if relevant_sections:
                            source_context += f"\n\n参考資料{i+1}（{Path(pdf_path).name}）:\n"
                            source_context += "\n".join(relevant_sections)
                            source_specific_content[f'pdf_{i}'] = {
                                'name': Path(pdf_path).name,
                                'sections': relevant_sections
                            }
        
        elif data_source == DataSourceType.WEB_SEARCH:
            search_queries = [
                "Luup 岡井　評判",
                "Luup 電動キックボード 炎上",
                "Luup 電動キックボード 事故 マナー 問題",
                "シェアライド サービス 評判 改善 戦略",
                "マイクロモビリティ 規制 安全対策"
            ]
            
            all_search_results = []
            for query in search_queries[:2]:
                results = self.search_web(query, max_results=3)
                all_search_results.extend(results)
                source_specific_content[f'search_{query}'] = results
            
            unique_results = {}
            for result in all_search_results:
                url = result['url']
                if url not in unique_results or result['score'] > unique_results[url]['score']:
                    unique_results[url] = result
            
            source_context = self.format_search_results(list(unique_results.values())[:5])
        
        elif data_source == DataSourceType.VIDEO_DATA:
            if json_data:
                analysis = self.analyze_json_data(json_data, post.title + " " + post.body)
                if analysis:
                    source_context = self._format_video_analysis(analysis)
                    source_specific_content['video_analysis'] = analysis
        
        elif data_source == DataSourceType.CSV:
            if hasattr(self, 'csv_analyzer'):
                # CSVファイルのパスを取得（環境に応じて設定）
                csv_files = glob.glob(os.path.join(PROJECT_ROOT, "data", "csv", "*.csv"))
                if not csv_files:
                    # デフォルトのCSVファイルパス
                    csv_files = glob.glob(os.path.join(PROJECT_ROOT, "*.csv"))
                
                if csv_files:
                    # 最新のCSVファイルを使用
                    csv_path = max(csv_files, key=os.path.getmtime)
                    analysis = self.csv_analyzer.generate_analysis_summary(csv_path, post.title + " " + post.body)
                    if analysis and 'error' not in analysis:
                        source_context = self.format_csv_analysis(analysis)
                        source_specific_content['csv_analysis'] = analysis

        source_instructions = {
            DataSourceType.PDF: "政策文書やマニフェストから得られる公式な情報に基づいて",
            DataSourceType.WEB_SEARCH: "最新のニュース報道や専門家の分析から得られる情報に基づいて",
            DataSourceType.VIDEO_DATA: "YouTube/TikTokの動画トレンドと視聴者の反応から得られる情報に基づいて",
            DataSourceType.CSV: "X（Twitter）の投稿データとユーザーの反応から得られる情報に基づいて"  # 新規追加
        }
        
        prompt = f"""
{datetime_context}

あなたは企業評判管理とPR戦略の専門家で、{source_instructions[data_source]}分析を行います。
現在は{datetime.now().strftime('%Y年%m月%d日')}です。

【分析対象企業】
株式会社Luup - 電動キックボードを中心としたシェアライドサービス

【重要な背景情報（PDFレポートより）】
1. 最大のリスク：消防法違反（マンション避難器具設置問題）- 10月にバースト発生
2. 要注意アカウント：@chinniisan（ちんにい）- #反Luup教の中心人物、岡井CEOより高いPagerank
3. 主要クラスター：信号無視、ポート問題、権威批判（天下り・東大卒）
4. バースト傾向：一般ユーザーの不満が蓄積し、特定の事件で爆発的に拡散

【現状の課題】
- 交通事故の増加やユーザーマナーの悪さがSNS上で指摘
- SNS上での炎上により企業イメージが悪化
- #反Luup教コミュニティの組織的な批判活動
- ポート設置場所への本質的な不満の蓄積

【分析の目的】
{post.title}
{post.body}

【利用可能なデータ】
{source_context}

【重要な指示】
1. PDFレポートで特定された最大リスク（消防法違反）を必ず考慮
2. インフルエンサー対策（特に@chinniisan）の重要性を認識
3. 岡井CEOの発言が引用RTされるリスクを踏まえた提案
4. バースト発生のタイミング（10月の事例）を参考にした予防策
5. 権威批判（東大卒、天下り）を誘発しないコミュニケーション戦略
6. ポート設置への本質的不満への対処法
7. 1800-2000文字程度で具体的かつ実行可能な提案

株式会社Luupが注視すべきポイントと、SNS・メディア発信時の注意点について、
PDFレポートの知見を踏まえた専門的な洞察を提供してください。
"""
        
        response_content = self._call_llm_api_with_retry(prompt, model, max_tokens=1000)
        
        return PersuasiveResponse(
            content=response_content,
            treatment_condition=treatment,
            persuasion_score=0.0,
            model_used=f"{model}_{data_source.value}",
            generation_params={
                'temperature': 0.7,
                'max_tokens': 1000,
                'data_source': data_source.value,
                'source_context_length': len(source_context)
            },
            data_source_type=data_source,
            source_specific_content=source_specific_content,
            tournament_round=0
        )
    
    def _format_video_analysis(self, analysis: Dict[str, Any]) -> str:
        """動画分析結果をフォーマット（詳細版）"""
        if not analysis or 'error' in analysis:
            return ""
        
        formatted = "\n\n【YouTube/TikTok動画分析（詳細版）】\n"
        
        # サマリー
        if 'summary' in analysis:
            summary = analysis['summary']
            formatted += f"━━━━━━━━━━━━━━━━━━━━━━\n"
            formatted += f"総動画数: {summary['total_videos']:,}件\n"
            formatted += f"総視聴回数: {summary['total_views']:,}回\n"
            formatted += f"ユニークチャンネル: {summary['unique_channels']}件\n"
        
        # センチメント分析
        if 'sentiment_analysis' in analysis:
            sentiment = analysis['sentiment_analysis']
            formatted += f"\n■ センチメント分析:\n"
            
            if 'percentages' in sentiment:
                for sent_type, percentage in sentiment['percentages'].items():
                    if percentage > 0:
                        formatted += f"  {sent_type}: {percentage:.1f}%"
                        
                        # エンゲージメント統計を追加
                        if 'engagement_stats' in sentiment and sent_type in sentiment['engagement_stats']:
                            avg_views = sentiment['engagement_stats'][sent_type].get('avg_views', 0)
                            formatted += f" (平均{avg_views:,.0f}回視聴)\n"
                        else:
                            formatted += "\n"
            
            # トップ動画の例
            if 'top_videos_by_sentiment' in sentiment:
                for sent_type in ['positive', 'negative']:
                    if sent_type in sentiment['top_videos_by_sentiment']:
                        top_videos = sentiment['top_videos_by_sentiment'][sent_type]
                        if top_videos:
                            formatted += f"\n  {sent_type}動画の例:\n"
                            for video in top_videos[:2]:
                                formatted += f"    ・「{video['title'][:40]}...」\n"
                                formatted += f"      ({video['views']:,}回視聴)\n"
        
        # インフルエンサー分析
        if 'influencer_analysis' in analysis:
            influencers = analysis['influencer_analysis']
            formatted += f"\n■ 影響力の高いチャンネル:\n"
            
            for i, channel in enumerate(influencers[:3], 1):
                formatted += f"  {i}. {channel['channel_name']}\n"
                formatted += f"     フォロワー: {channel['follower_count']:,}人\n"
                formatted += f"     平均視聴回数: {channel['avg_views']:,.0f}回\n"
                
                if channel.get('top_video'):
                    video = channel['top_video']
                    formatted += f"     人気動画:「{video['title'][:30]}...」\n"
                    formatted += f"     ({video['views']:,}回視聴)\n"
        
        # トレンド分析
        if 'temporal_trends' in analysis:
            trends = analysis['temporal_trends']
            
            if 'peak_day' in trends and trends['peak_day']:
                formatted += f"\n■ 時系列トレンド:\n"
                formatted += f"  ピーク日: {trends['peak_day']['date']}\n"
                formatted += f"  (視聴回数: {trends['peak_day']['total_views']:,}回)\n"
                
                if 'trend_direction' in trends:
                    direction_text = {
                        'increasing': '上昇傾向',
                        'decreasing': '下降傾向',
                        'insufficient_data': 'データ不足'
                    }
                    formatted += f"  トレンド: {direction_text.get(trends['trend_direction'], '不明')}\n"
        
        # トピック分析
        if 'topic_analysis' in analysis:
            topics = analysis['topic_analysis']
            
            # マッチしたキーワード
            if 'matched_keywords' in topics and topics['matched_keywords']:
                formatted += f"\n■ 主要キーワード:\n"
                for kw in topics['matched_keywords'][:5]:
                    formatted += f"  ・{kw['keyword']}: {kw['video_count']}件 "
                    formatted += f"(総{kw['total_views']:,}回視聴)\n"
            
            # 政治分析
            if 'political_analysis' in topics:
                political = topics['political_analysis']
                
                if political.get('parties_mentioned'):
                    formatted += f"\n■ 言及された政党:\n"
                    sorted_parties = sorted(political['parties_mentioned'].items(), 
                                        key=lambda x: x[1], reverse=True)
                    for party, count in sorted_parties[:3]:
                        formatted += f"  ・{party}: {count}回\n"
                
                if political.get('politicians_mentioned'):
                    formatted += f"\n■ 言及された政治家:\n"
                    sorted_politicians = sorted(political['politicians_mentioned'].items(),
                                            key=lambda x: x[1], reverse=True)
                    for politician, count in sorted_politicians[:3]:
                        formatted += f"  ・{politician}: {count}回\n"
            
            # トレンディングトピック
            if 'trending_topics' in topics and topics['trending_topics']:
                formatted += f"\n■ トレンドトピック:\n"
                for topic in topics['trending_topics'][:3]:
                    formatted += f"  ・{topic}\n"
        
        # エンゲージメント指標
        if 'engagement_metrics' in analysis:
            metrics = analysis['engagement_metrics']
            
            if 'performance_distribution' in metrics:
                formatted += f"\n■ パフォーマンス分布:\n"
                perf_dist = metrics['performance_distribution']
                
                for tier in ['viral', 'high', 'medium']:
                    if tier in perf_dist and perf_dist[tier]['count'] > 0:
                        tier_names = {
                            'viral': 'バイラル（10万回以上）',
                            'high': '高パフォーマンス（5-10万回）',
                            'medium': '中パフォーマンス（1-5万回）'
                        }
                        formatted += f"  {tier_names[tier]}: {perf_dist[tier]['count']}件 "
                        formatted += f"({perf_dist[tier]['percentage']:.1f}%)\n"
        
        formatted += "━━━━━━━━━━━━━━━━━━━━━━\n"
        
        return formatted
    
    def _call_llm_api_with_retry(self, prompt: str, model: str, max_tokens: int = 2000) -> str:
        """リトライ機構付きAPI呼び出し（改善案6: エラーハンドリングの強化）"""
        for attempt in range(self.config.max_retries):
            try:
                return self._call_llm_api(prompt, model, max_tokens)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"All retries failed for {model}: {e}")
                    return self._generate_fallback_response(prompt)
                
                wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Retry {attempt + 1}/{self.config.max_retries} after {wait_time}s")
                time.sleep(wait_time)
        
        return self._generate_fallback_response(prompt)
    
    def _call_llm_api(self, prompt: str, model: str, max_tokens: int = 2000) -> str:
        """Call the appropriate LLM API"""
        datetime_context = self.get_current_datetime_context()
        
        natural_writing_instruction = f"""
{datetime_context}

以下の点に注意して、日本語で応答してください：
1. 現在の日時を正しく認識し、時系列を正確に把握する（今日は{datetime.now().strftime('%Y年%m月%d日')}です）
2. 専門用語を適切に使用し、具体と抽象のバランスのとれた、読者の知的好奇心を刺激する文章を書きなさい
3. 段落は適切に分け、読みやすい長さにする
4. 提供されたデータを適切に利用し、数字や用語などデータから得られた情報を用いて具体性のある文章を書きなさい
5. 「です・ます」調で一貫性のある文体を使う
6. 箇条書きや番号付けは避け、流れるような文章にする
7. 同じ表現の繰り返しを避ける
8. 必ず文章を完結させ、途中で切れないようにする
"""
        enhanced_prompt = natural_writing_instruction + prompt
        
        if model == 'gpt-4.1' or model == 'gpt-4.1-finetuned':
            if self.api_config.openai_client:
                response = self.api_config.openai_client.chat.completions.create(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system", "content": f"あなたは日本の最大の新聞社（クオリティペーパー）のエース編集委員です。現在は{datetime.now().strftime('%Y年%m月%d日')}です。"},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    temperature=0.8,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                return self._post_process_response(content)
        
        return self._generate_fallback_response(prompt)
    
    def _post_process_response(self, content: str) -> str:
        """応答を後処理して自然さを向上させる"""
        import re
        
        content = re.sub(r'\*\*([^*]+)\*\*', r'「\1」', content)
        content = re.sub(r'###\s*', '', content)
        content = re.sub(r'##\s*', '', content)
        content = re.sub(r'#\s*', '', content)
        
        sentences = content.split('。')
        unique_sentences = []
        seen_patterns = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            pattern = sentence[:20] if len(sentence) > 20 else sentence
            
            if pattern not in seen_patterns:
                unique_sentences.append(sentence)
                seen_patterns.add(pattern)
        
        content = '。'.join(unique_sentences)
        
        if content and not content.endswith(('。', '！', '？', '」')):
            content += '。'
        
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'([。！？])\n([^「\n])', r'\1\n\n\2', content)
        
        connectives = ['また', 'さらに', 'そして', 'しかし', 'ただし', 'なお', 'つまり']
        for conn in connectives:
            content = re.sub(f'。{conn}、', f'。\n\n{conn}、', content)
        
        content = re.sub(r'^[-・]\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\d+\.\s*', '', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response if API calls fail"""
        return (
            "ご指摘の視点は大変興味深く、よく考えられた内容だと思います。"
            "参政党の躍進について、確かにSNSやショート動画の影響は無視できない要因です。"
            "しかし、より深い分析のためには、有権者の意識変化や既存政党への不満、"
            "そして新しいメディア環境がどのように政治参加の形を変えているかを"
            "総合的に検討する必要があるでしょう。"
            "この問題についてさらに議論を深めていければ幸いです。"
        )
    
    def _create_fallback_response(self, post: RedditPost, treatment: TreatmentCondition,
                                 user_profile: Optional[UserProfile], 
                                 pdf_references: Optional[List[str]],
                                 data_source: Optional[DataSourceType] = None) -> PersuasiveResponse:
        """エラー時のフォールバック応答を作成"""
        return PersuasiveResponse(
            content=self._generate_fallback_response(""),
            treatment_condition=treatment,
            persuasion_score=0.0,
            model_used=f"gpt-4.1_fallback_{data_source.value if data_source else 'unknown'}",
            generation_params={'error': 'fallback'},
            user_profile=user_profile,
            pdf_context=str(pdf_references) if pdf_references else None,
            data_source_type=data_source,
            tournament_round=0
        )

# ====================
# Persuasion Experiment
# ====================
class PersuasionExperiment:
    """Complete experimental pipeline"""
    
    def __init__(self, api_config: APIConfig, tournament_config: TournamentConfig = None):
        self.profiler = UserProfiler()
        self.config = tournament_config or TournamentConfig()
        self.optimizer = PersuasionOptimizer(api_config, self.config)
        self.selector = self.optimizer.tournament_selector

    def run_experiment(self, 
                    target_post: RedditPost,
                    user_history: List[RedditPost],
                    treatment: TreatmentCondition,
                    use_full_candidates: bool = False,
                    pdf_references: List[str] = None,
                    json_data: Optional[Dict] = None,
                    enable_learning: bool = True,
                    target_length: int = 500,
                    max_tokens: Optional[int] = None,
                    use_single_model: bool = False,
                    model_name: Optional[str] = None,
                    force_data_source: Optional[str] = None) -> Dict:
        """Run complete persuasion optimization pipeline"""
        
        if not self._is_answerable(target_post):
            return {'status': 'filtered', 'reason': 'Post requires future knowledge'}
        
        topic = self._classify_topic(target_post)
        readability = textstat.flesch_reading_ease(target_post.body)
        
        user_profile = None
        if treatment == TreatmentCondition.PERSONALIZATION:
            user_profile = self.profiler.analyze_posting_history(user_history)
        else:
            user_profile = self.profiler.create_political_researcher_profile()
        
        # 単一モデルモードの場合（1つのデータソースのみを使用）
        if use_single_model:
            logger.info(f"Using single model mode with single data source")
            
            # force_data_sourceが指定されている場合はそれを使用
            if force_data_source:
                data_source_map = {
                    'pdf': DataSourceType.PDF,
                    'web_search': DataSourceType.WEB_SEARCH,
                    'video_data': DataSourceType.VIDEO_DATA
                }
                selected_source = data_source_map.get(force_data_source, DataSourceType.WEB_SEARCH)
                logger.info(f"Using forced data source: {selected_source.value}")
            else:
                # データソースタイプをランダムに1つ選択
                available_sources = []
                if pdf_references:
                    available_sources.append(DataSourceType.PDF)
                if self.optimizer.api_config.tavily_client:
                    available_sources.append(DataSourceType.WEB_SEARCH)
                if json_data:
                    available_sources.append(DataSourceType.VIDEO_DATA)
                
                if not available_sources:
                    # データソースが利用できない場合はデフォルトでWeb検索を試みる
                    selected_source = DataSourceType.WEB_SEARCH
                    logger.warning("No data sources available, defaulting to web search")
                else:
                    selected_source = random.choice(available_sources)
                    logger.info(f"Randomly selected data source: {selected_source.value}")
            
            # 選択されたデータソースのみを使用して応答を生成
            single_response = self.optimizer._generate_source_specific_response(
                target_post,
                treatment,
                model_name or 'gpt-4.1',
                selected_source,
                user_profile,
                pdf_references,
                json_data
            )
            
            # スコアを計算
            single_response.detailed_score = self.selector._score_response_with_details(single_response)
            single_response.persuasion_score = single_response.detailed_score.total
            
            return {
                'status': 'success',
                'response': single_response,
                'user_profile': user_profile,
                'topic': topic,
                'readability': readability,
                'delay_minutes': self._calculate_posting_delay(),
                'num_candidates': 1,
                'treatment': treatment.value,
                'pdf_references': pdf_references,
                'tournament_log': [{
                    'round': 'single',
                    'type': 'single_model_single_source',
                    'model': model_name or 'gpt-4.1',
                    'data_source': selected_source.value,
                    'score': single_response.persuasion_score,
                    'score_details': single_response.detailed_score.breakdown if single_response.detailed_score else {}
                }],
                'learning_enabled': False,
                'single_model_mode': True,
                'selected_data_source': selected_source.value
            }
        
        # 通常のトーナメントモード（複数エージェント）
        num_candidates = 128 if use_full_candidates else 4
        logger.info(f"Generating {num_candidates} candidate responses...")
        
        candidates = self.optimizer.generate_responses(
            target_post, 
            treatment, 
            user_profile,
            num_candidates=num_candidates,
            pdf_references=pdf_references,
            json_data=json_data
        )
        
        time.sleep(1)

        winning_response, tournament_log = self.selector.run_tournament_with_learning_tracking(
            candidates,
            enable_learning=enable_learning,
            user_question=f"{target_post.title} {target_post.body}"  # 追加
        )

        delay_minutes = self._calculate_posting_delay()

        return {
            'status': 'success',
            'response': winning_response,
            'user_profile': user_profile,
            'topic': topic,
            'readability': readability,
            'delay_minutes': delay_minutes,
            'num_candidates': len(candidates),
            'treatment': treatment.value,
            'pdf_references': pdf_references,
            'tournament_log': tournament_log,
            'learning_enabled': enable_learning,
            'single_model_mode': False
        }
    
    def _is_answerable(self, post: RedditPost) -> bool:
        """Check if post can be answered with current knowledge"""
        analysis_keywords = ['分析', '考察', '議論', 'analysis', 'discussion', 'CMV']
        if any(keyword in post.title or keyword in post.body for keyword in analysis_keywords):
            return True
        
        far_future_indicators = ['2030', '2040', '2050', 'distant future', 'long-term prediction']
        text_lower = (post.title + ' ' + post.body).lower()
        
        return not any(indicator in text_lower for indicator in far_future_indicators)
    
    def _classify_topic(self, post: RedditPost) -> str:
        """Classify post topic"""
        text_lower = (post.title + ' ' + post.body).lower()
        
        topic_keywords = {
            'politics': ['government', 'election', 'policy', 'political', 'vote',
                        '政治', '選挙', '政党', '議会', '投票', '参院選', '与党', '野党'],
            'technology': ['ai', 'computer', 'software', 'internet', 'tech',
                        'AI', 'コンピュータ', 'ソフトウェア', 'インターネット', 'テクノロジー'],
            'society': ['people', 'culture', 'social', 'community', 'human',
                    '社会', '文化', 'コミュニティ', '人間', 'SNS'],
            'economics': ['money', 'economy', 'market', 'business', 'finance',
                        '経済', '市場', 'ビジネス', '金融', '財政'],
            'education': ['school', 'teacher', 'student', 'education', 'learn',
                        '学校', '教師', '学生', '教育', '学習'],
            'health': ['health', 'medical', 'doctor', 'disease', 'treatment',
                    '健康', '医療', '医師', '病気', '治療']
        }
        
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower or keyword in post.title + post.body)
            topic_scores[topic] = score
        
        return max(topic_scores, key=topic_scores.get) if any(topic_scores.values()) else 'general'
    
    def _calculate_posting_delay(self) -> int:
        """Calculate random posting delay"""
        delay = np.random.normal(15, 30)
        return int(np.clip(delay, 10, 180))

# ====================
# FastAPI Application
# ====================
app = FastAPI(
    title="知識蒸留 API Server - 企業評判分析版 v4.0",
    description="株式会社Luupの企業イメージ改善とSNS戦略分析のための多角的データ分析システム",
    version="4.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
tournament_config = TournamentConfig()
api_config = APIConfig()
experiment = PersuasionExperiment(api_config, tournament_config)
uploaded_files_data = {}

# ====================
# Request/Response Models
# ====================
class GenerateRequest(BaseModel):
    title: str
    body: str
    treatment: Optional[str] = Field(default="generic")
    processing_type: Optional[str] = Field(default=None)
    num_candidates: Optional[int] = Field(default=4)
    enable_learning: Optional[bool] = Field(default=True)
    enable_web_search: Optional[bool] = Field(default=True)
    json_data: Optional[Any] = Field(default=None)
    pdf_references: Optional[List[str]] = Field(default_factory=list)
    user_profile: Optional[Dict] = Field(default=None)
    user_history: Optional[List[Dict]] = Field(default_factory=list)
    target_length: Optional[int] = Field(default=500, description="目標文字数")
    max_tokens: Optional[int] = Field(default=None, description="最大トークン数")
    use_single_model: Optional[bool] = Field(default=False, description="単一モデル・単一データソースモード")
    model_name: Optional[str] = Field(default=None, description="使用するモデル名")
    parallel_processing: Optional[bool] = Field(default=True, description="並列処理を有効化")
    force_data_source: Optional[str] = Field(default=None, description="強制的に使用するデータソース（pdf/web_search/video_data）")
    csv_references: Optional[List[str]] = Field(default_factory=list, description="CSVファイルのパスリスト")
    
    @validator('json_data', pre=True)
    def parse_json_data(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                logger.warning(f"Failed to parse json_data as JSON string")
                return v
        return v
    
    @validator('force_data_source', pre=True)
    def validate_data_source(cls, v):
        if v is not None:
            valid_sources = ['pdf', 'web_search', 'video_data']
            if v not in valid_sources:
                raise ValueError(f"force_data_source must be one of {valid_sources}")
        return v
    
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

class GenerateResponse(BaseModel):
    response: str
    model_used: str
    persuasion_score: float
    score_details: Optional[Dict[str, float]] = None
    tournament_log: List[Dict[str, Any]]
    processing_time: float
    total_candidates: int
    learning_applied: bool = False
    json_analysis_included: bool = False
    web_search_included: bool = False
    topic: Optional[str] = None
    readability: Optional[float] = None
    user_profile: Optional[Dict] = None
    cache_hit_rate: Optional[float] = None
    single_model_mode: bool = False
    selected_data_source: Optional[str] = None

# ====================
# Endpoints
# ====================
@app.get("/")
async def root():
    return {
        "message": "知識蒸留 API Server - 単一モデル複数エージェント版 v3.1",
        "version": "3.1.0",
        "modes": {
            "tournament_mode": {
                "description": "複数エージェントがトーナメントで競い合い、学習を通じて知識を統合",
                "use_single_model": False,
                "data_sources": "各エージェントが1つのデータソースを専門的に扱う"
            },
            "single_model_mode": {
                "description": "単一モデルが1つのデータソースのみを使用して応答を生成",
                "use_single_model": True,
                "data_sources": "PDF、Web検索、動画データ、CSVデータから1つを選択（ランダムまたは指定）",
                "force_data_source_options": ["pdf", "web_search", "video_data", "csv"]
            }
        },
        "improvements": [
            "アーキテクチャの一貫性改善",
            "スコアリング関数の公平性向上",
            "並列処理によるパフォーマンス最適化",
            "学習キャッシュによる効率改善",
            "詳細なスコア情報による透明性向上",
            "リトライ機構によるエラーハンドリング強化",
            "メモリ効率の改善",
            "設定の外部化",
            "単一モデルモードでの単一データソース使用"
        ],
        "config": {
            "agents_per_source": tournament_config.agents_per_source,
            "learning_enabled": tournament_config.learning_enabled,
            "parallel_processing": tournament_config.parallel_processing,
            "cache_enabled": tournament_config.cache_enabled,
            "max_retries": tournament_config.max_retries
        }
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """メイン生成エンドポイント"""
    try:
        start_time = datetime.now()
        
        # 設定を更新
        if request.parallel_processing is not None:
            tournament_config.parallel_processing = request.parallel_processing
        
        logger.info(f"=== Generate Request Received (v3) ===")
        logger.info(f"Title: {request.title}")
        logger.info(f"Num candidates: {request.num_candidates}")
        logger.info(f"Parallel processing: {tournament_config.parallel_processing}")
        logger.info(f"Single model mode: {request.use_single_model}")
        if request.force_data_source:
            logger.info(f"Forced data source: {request.force_data_source}")
        
        # RedditPost オブジェクトを作成
        target_post = RedditPost(
            post_id=hashlib.md5(request.title.encode()).hexdigest()[:8],
            subreddit="changemyview",
            title=request.title,
            body=request.body,
            author="api_user",
            timestamp=datetime.now(),
            score=100
        )
        
        # ユーザー履歴を変換
        user_history = []
        if request.user_history:
            for hist in request.user_history:
                try:
                    user_history.append(RedditPost(
                        post_id=hist.get("post_id", f"hist_{len(user_history)}"),
                        subreddit=hist.get("subreddit", "unknown"),
                        title=hist.get("title", ""),
                        body=hist.get("body", ""),
                        author=hist.get("author", "api_user"),
                        timestamp=datetime.now(),
                        score=hist.get("score", 0)
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse user history item: {e}")
        
        # Treatment condition を Enum に変換
        treatment_map = {
            "generic": TreatmentCondition.GENERIC,
            "汎用": TreatmentCondition.GENERIC,
            "personalization": TreatmentCondition.PERSONALIZATION,
            "パーソナライゼーション": TreatmentCondition.PERSONALIZATION,
            "community_aligned": TreatmentCondition.COMMUNITY_ALIGNED,
            "コミュニティ最適化": TreatmentCondition.COMMUNITY_ALIGNED
        }
        treatment = treatment_map.get(request.treatment.lower(), TreatmentCondition.GENERIC)
        
        # JSONデータを設定
        json_data = request.json_data or uploaded_files_data.get("current")
        
        # 実験を実行
        result = experiment.run_experiment(
            target_post=target_post,
            user_history=user_history,
            treatment=treatment,
            use_full_candidates=(request.num_candidates > 10),
            pdf_references=request.pdf_references,
            json_data=json_data,
            enable_learning=request.enable_learning,
            target_length=request.target_length,
            max_tokens=request.max_tokens,
            use_single_model=request.use_single_model,
            model_name=request.model_name,
            force_data_source=request.force_data_source
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if result['status'] == 'success':
            winning_response = result['response']
            
            # ユーザープロフィールを辞書形式に変換
            user_profile_dict = None
            if result.get('user_profile'):
                profile = result['user_profile']
                user_profile_dict = {
                    "username": profile.username,
                    "gender": profile.gender.value,
                    "age_range": list(profile.age_range),
                    "location": profile.location,
                    "political_orientation": profile.political_orientation.value,
                    "interests": profile.interests,
                    "writing_style": profile.writing_style
                }
            
            # 詳細スコア情報を取得
            score_details = None
            if winning_response.detailed_score:
                score_details = winning_response.detailed_score.breakdown
            
            # キャッシュヒット率を取得
            cache_hit_rate = None
            if experiment.selector.learning_cache:
                cache_hit_rate = experiment.selector.learning_cache.get_hit_rate()
            
            return GenerateResponse(
                response=winning_response.content,
                model_used=winning_response.model_used,
                persuasion_score=float(winning_response.persuasion_score),
                score_details=score_details,
                tournament_log=result.get('tournament_log', []),
                processing_time=float(processing_time),
                total_candidates=result.get('num_candidates', 0),
                learning_applied=result.get('learning_enabled', False),
                json_analysis_included=bool(json_data),
                web_search_included=request.enable_web_search,
                topic=result.get('topic'),
                readability=float(result.get('readability', 0.0)),
                user_profile=user_profile_dict,
                cache_hit_rate=cache_hit_rate,
                single_model_mode=result.get('single_model_mode', False),
                selected_data_source=result.get('selected_data_source')
            )
        else:
            raise HTTPException(status_code=400, detail=result.get('reason', 'Unknown error'))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def api_status():
    """APIのステータスと利用可能な機能を確認"""
    return {
        "status": "operational",
        "version": "3.1.0",
        "architecture": "single_model_multiple_agents_v3.1",
        "modes": {
            "tournament": {
                "enabled": True,
                "description": "複数エージェントによるトーナメント方式",
                "agents_per_source": tournament_config.agents_per_source
            },
            "single_model": {
                "enabled": True,
                "description": "単一モデル・単一データソース方式",
                "available_sources": ["pdf", "web_search", "video_data"]
            }
        },
        "features": {
            "llm_models": {
                "openai": api_config.openai_client is not None,
                "anthropic": api_config.anthropic_client is not None,
                "gemini": api_config.gemini_model is not None,
                "openrouter": api_config.openrouter_api_key is not None
            },
            "web_search": api_config.tavily_client is not None,
            "pdf_processing": True,
            "json_analysis": True,
            "user_profiling": True,
            "sentiment_analysis": True,
            "tournament_selection": True,
            "cross_source_learning": True,
            "multi_agent_system": True,
            "parallel_processing": tournament_config.parallel_processing,
            "learning_cache": tournament_config.cache_enabled,
            "memory_compression": tournament_config.memory_compression
        },
        "performance_metrics": {
            "cache_hit_rate": experiment.selector.learning_cache.get_hit_rate() if experiment.selector.learning_cache else None,
            "max_concurrent_agents": 10 if tournament_config.parallel_processing else 1,
            "retry_config": {
                "max_retries": tournament_config.max_retries,
                "retry_delay": tournament_config.retry_delay
            }
        }
    }

@app.get("/api/available-pdfs")
async def get_available_pdfs():
    """利用可能なPDFファイルのリストを返す"""
    try:
        pdf_files = []
        full_paths = []
        
        # PDFディレクトリが存在するか確認
        if os.path.exists(PDF_DIR):
            # PDFファイルを検索
            for file in os.listdir(PDF_DIR):
                if file.endswith('.pdf'):
                    pdf_files.append(file)
                    full_paths.append(os.path.join(PDF_DIR, file))
            
            # サブディレクトリも検索
            for root, dirs, files in os.walk(PDF_DIR):
                for file in files:
                    if file.endswith('.pdf') and file not in pdf_files:
                        pdf_files.append(file)
                        full_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {PDF_DIR}")
        
        return {
            "pdfs": pdf_files,
            "full_paths": full_paths,
            "directory": PDF_DIR,
            "count": len(pdf_files)
        }
        
    except Exception as e:
        logger.error(f"Error fetching PDF list: {str(e)}")
        return {
            "pdfs": [],
            "full_paths": [],
            "directory": PDF_DIR,
            "count": 0,
            "error": str(e)
        }

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """PDFファイルをアップロード"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # ファイルを保存
        file_path = os.path.join(PDF_DIR, file.filename)
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(PDF_DIR, exist_ok=True)
        
        # ファイルを保存
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"PDF uploaded successfully: {file_path}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": file_path,
            "size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/available-csvs")
async def get_available_csvs():
    """利用可能なCSVファイルのリストを返す"""
    try:
        # デバッグ情報を追加
        logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
        
        csv_dir = os.path.join(PROJECT_ROOT, "data", "csv")
        logger.info(f"CSV directory path: {csv_dir}")
        logger.info(f"CSV directory exists: {os.path.exists(csv_dir)}")
        
        csv_files = []
        full_paths = []
        
        # CSVディレクトリが存在するか確認
        if os.path.exists(csv_dir):
            logger.info(f"Listing files in {csv_dir}")
            files = os.listdir(csv_dir)
            logger.info(f"All files in directory: {files}")
            
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(file)
                    full_path = os.path.join(csv_dir, file)
                    full_paths.append(full_path)
                    logger.info(f"Found CSV: {file}")
        else:
            logger.warning(f"CSV directory does not exist: {csv_dir}")
        
        # プロジェクトルートも検索
        root_files = os.listdir(PROJECT_ROOT)
        for file in root_files:
            if file.endswith('.csv') and file not in csv_files:
                csv_files.append(file)
                full_paths.append(os.path.join(PROJECT_ROOT, file))
                logger.info(f"Found CSV in root: {file}")
        
        logger.info(f"Total CSV files found: {len(csv_files)}")
        
        return {
            "csvs": csv_files,
            "full_paths": full_paths,
            "count": len(csv_files),
            "debug": {
                "project_root": PROJECT_ROOT,
                "csv_dir": csv_dir,
                "csv_dir_exists": os.path.exists(csv_dir)
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching CSV list: {str(e)}", exc_info=True)
        return {
            "csvs": [],
            "full_paths": [],
            "count": 0,
            "error": str(e),
            "debug": {
                "project_root": PROJECT_ROOT if 'PROJECT_ROOT' in locals() else "undefined",
                "error_type": type(e).__name__
            }
        }
    
@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """CSVファイルをアップロード"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # CSVディレクトリを作成
        csv_dir = os.path.join(PROJECT_ROOT, "data", "csv")
        os.makedirs(csv_dir, exist_ok=True)
        
        # ファイルを保存
        file_path = os.path.join(csv_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"CSV uploaded successfully: {file_path}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": file_path,
            "size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Error uploading CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/analyze-reputation-report")
async def analyze_reputation_report(file: UploadFile = File(...)):
    """Luup評判分析レポートPDFを詳細分析"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # ファイルを一時保存
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # PDFレポートを分析
        analysis = experiment.optimizer.analyze_pdf_report(tmp_path)
        
        # 危機対応戦略を生成
        strategy = experiment.optimizer.generate_crisis_response_strategy(analysis)
        
        # 一時ファイルを削除
        import os
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "analysis": analysis,
            "recommended_strategy": strategy,
            "risk_summary": {
                "critical_risks": len([r for r in analysis.get('critical_risks', []) if r.get('severity') == 'critical']),
                "key_influencers": len(analysis.get('key_influencers', [])),
                "timeline_events": len(analysis.get('timeline_events', []))
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing reputation report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏆 知識蒸留 API Server - 単一モデル複数エージェント版 v3.1 🏆")
    print("="*60)
    print("\n主要機能:")
    print("【トーナメントモード】")
    print("✓ 単一モデル（GPT-4）による複数エージェント生成")
    print("✓ 各エージェントが異なるデータソースを専門的に扱う")
    print("✓ 改善された学習メカニズム（エージェントの制約を維持）")
    print("✓ 公平なスコアリング（ラウンドに応じた評価）")
    print("")
    print("【単一モデルモード】")
    print("✓ 単一モデルが1つのデータソースのみを使用")
    print("✓ データソースの選択：ランダムまたは指定可能")
    print("✓ 利用可能なデータソース：PDF、Web検索、動画データ")
    print("")
    print("【パフォーマンス機能】")
    print("✓ 並列処理によるパフォーマンス向上")
    print("✓ 学習キャッシュによる効率化")
    print("✓ 詳細なスコア情報による透明性")
    print("✓ リトライ機構によるロバスト性")
    print("✓ メモリ効率の最適化")
    print("✓ 設定の外部化")
    print("\nサーバー起動中...")
    print("http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)