#!/usr/bin/env python3
"""
知識蒸留APIサーバー - 単一モデル複数エージェント版 v3.1
各エージェントが異なるデータソースを専門的に扱い、トーナメントで知識を統合
改善版：パフォーマンス最適化、学習効率改善、評価透明性向上
v3.1: 単一モデルモードでも単一データソースのみを使用
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

class TreatmentCondition(Enum):
    GENERIC = "generic"
    PERSONALIZATION = "personalization"
    COMMUNITY_ALIGNED = "community_aligned"

class DataSourceType(Enum):
    PDF = "pdf"
    WEB_SEARCH = "web_search"
    VIDEO_DATA = "video_data"

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
    gender: Gender
    age_range: Tuple[int, int]
    ethnicity: Optional[str]
    location: Optional[str]
    political_orientation: PoliticalOrientation
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
            r'ファクト情報[なし|あり]',
            r'^\d+$',
            r'✔',
            r'\[.*?\]「.*?」',
            r'についての具体的説明',
        ]
        
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

# ====================
# User Profiler
# ====================
class UserProfiler:
    """Analyzes user's posting history to infer demographic attributes"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
        self.gender_indicators = {
            Gender.MALE: ['俺', '僕', 'my wife', 'my girlfriend', 'as a man'],
            Gender.FEMALE: ['私', 'あたし', 'my husband', 'my boyfriend', 'as a woman'],
            Gender.NON_BINARY: ['they/them', 'non-binary', 'enby'],
            Gender.UNKNOWN: []
        }
        
        self.political_indicators = {
            PoliticalOrientation.LIBERAL: ['progressive', 'social justice', 'equality', '平等', 'リベラル'],
            PoliticalOrientation.CONSERVATIVE: ['traditional', 'free market', 'liberty', '保守', '伝統'],
            PoliticalOrientation.MODERATE: ['center', 'both sides', 'middle ground', '中道', '穏健'],
            PoliticalOrientation.PROGRESSIVE: ['revolution', 'systemic change', 'radical', '革新', '変革'],
            PoliticalOrientation.LIBERTARIAN: ['individual freedom', 'minimal government', '自由主義'],
            PoliticalOrientation.UNKNOWN: []
        }
        
        self.location_patterns = [
            r"I live in ([A-Za-z\s]+)",
            r"from ([A-Za-z\s]+)",
            r"([東京|大阪|名古屋|福岡|札幌|京都|神戸|横浜|川崎])在住"
        ]
    
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
        
        return UserProfile(
            username=posts[0].author if posts else "unknown",
            gender=gender,
            age_range=age_range,
            ethnicity=None,
            location=location,
            political_orientation=political,
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
            gender=Gender.UNKNOWN,
            age_range=(35, 45),
            ethnicity=None,
            location="東京都",
            political_orientation=PoliticalOrientation.MODERATE,
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
        """詳細なスコアリング - 長文対応版"""
        breakdown = {
            'base_quality': 0.0,
            'data_richness': 0.0,
            'source_coverage': 0.0,
            'integration_quality': 0.0,
            'learning_bonus': 0.0,
            'depth_bonus': 0.0,
            'knowledge_synthesis': 0.0,
            'length_appropriateness': 0.0  # 新規追加
        }
        
        content = response.content
        
        # 文字数評価（6000文字を目標）
        char_count = len(content)
        if 5000 <= char_count <= 7000:
            breakdown['length_appropriateness'] = 2.0
        else:
            breakdown['length_appropriateness'] = max(0, 2.0 - abs(char_count - 6000) / 3000)
        
        # 既存のスコアリングロジック（省略）...
        
        # 具体的データの豊富さを重視
        import re
        concrete_count = 0
        concrete_count += len(re.findall(r'「[^」]+」', content))  # タイトル
        concrete_count += len(re.findall(r'『[^』]+』', content))  # 記事名
        concrete_count += len(re.findall(r'\d{1,3}(?:,\d{3})*[回円票％%]', content))  # 数値
        
        breakdown['data_richness'] = min(5.0, concrete_count * 0.2)
        
        # 重み付けを更新
        weights = {
            'base_quality': 0.5,
            'data_richness': 3.0,  # 大幅に増加
            'source_coverage': 2.0,
            'integration_quality': 2.0,
            'depth_bonus': 1.5,
            'knowledge_synthesis': 2.5,
            'learning_bonus': 1.5,
            'length_appropriateness': 1.0
        }
        
        weighted_total = sum(
            breakdown[key] * weights.get(key, 1.0)
            for key in breakdown
        )
        
        return DetailedScore(
            total=weighted_total,
            breakdown=breakdown,
            winning_factors=self._identify_winning_factors(breakdown, concrete_count),
            missing_elements=self._identify_missing_elements(breakdown, concrete_count),
            tournament_round=response.tournament_round
        )

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
        
        return insights

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
            DataSourceType.VIDEO_DATA: "SNS動画トレンド"
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
        self.text_cleaner = PDFTextCleaner()
        self.json_cache = {}
        self.search_cache = {}
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
                include_domains=["asahi.com", "nikkei.com", "nhk.or.jp", "mainichi.jp", 
                               "yomiuri.co.jp", "sankei.com", "jiji.com", "reuters.com"],
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
        """YouTube/TikTok動画データを詳細分析"""
        if not json_data:
            return {}
        
        # データの正規化
        videos = []
        if isinstance(json_data, list):
            videos = json_data
        elif isinstance(json_data, dict):
            videos = json_data.get("videos", json_data.get("data", []))
        
        if not videos:
            return {}
        
        # 基本統計
        total_views = 0
        total_likes = 0
        
        # 動画タイトル分析用
        all_titles = []
        channel_info = {}
        keyword_to_videos = {}
        party_mentions = {
            '自民党': {'count': 0, 'videos': [], 'total_views': 0},
            '参政党': {'count': 0, 'videos': [], 'total_views': 0},
            '立憲民主党': {'count': 0, 'videos': [], 'total_views': 0},
            '国民民主党': {'count': 0, 'videos': [], 'total_views': 0},
            '公明党': {'count': 0, 'videos': [], 'total_views': 0},
        }
        
        for video in videos:
            if isinstance(video, dict):
                # 基本データ収集
                view_count = video.get("view_count", video.get("views", 0))
                like_count = video.get("likes", 0)
                
                if isinstance(view_count, (int, float)):
                    total_views += view_count
                if isinstance(like_count, (int, float)):
                    total_likes += like_count
                
                # タイトル分析
                title = video.get("title", "")
                all_titles.append(title)
                
                # チャンネル情報収集
                channel = video.get("channel", "unknown")
                if channel not in channel_info:
                    channel_info[channel] = {
                        'videos': [],
                        'total_views': 0,
                        'follower_count': video.get("channel_follower_count", 0)
                    }
                
                channel_info[channel]['videos'].append({
                    'title': title,
                    'view_count': view_count,
                    'id': video.get("id", "")
                })
                channel_info[channel]['total_views'] += view_count
        
        # 分析結果をまとめる
        analysis = {
            "total_videos": len(videos),
            "total_views": total_views,
            "total_likes": total_likes,
            "avg_engagement": total_likes / max(total_views, 1) if total_views > 0 else 0,
            "party_analysis": party_mentions,
            "top_viewed_videos": sorted(
                [v for v in videos if isinstance(v, dict)],
                key=lambda x: x.get('view_count', 0),
                reverse=True
            )[:10]
        }
        
        return analysis
    
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
                "参政党 躍進 理由 分析",
                "SNS 政治 影響 日本"
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
        
        source_instructions = {
            DataSourceType.PDF: "政策文書やマニフェストから得られる公式な情報に基づいて",
            DataSourceType.WEB_SEARCH: "最新のニュース報道や専門家の分析から得られる情報に基づいて",
            DataSourceType.VIDEO_DATA: "YouTube/TikTokの動画トレンドと視聴者の反応から得られる情報に基づいて"
        }
        
        prompt = f"""
{datetime_context}

あなたは日本の政治分析専門家で、{source_instructions[data_source]}分析を行います。
現在は{datetime.now().strftime('%Y年%m月%d日')}です。

投稿内容:
タイトル: {post.title}
本文: {post.body}

【利用可能なデータ】
{source_context}

【重要な指示】
1. あなたは{data_source.value}のデータソースのみにアクセスできます
2. 他のデータソースの情報は一切持っていません
3. 自分が持つデータソースの情報を最大限活用して分析してください
4. 400-600文字程度で応答してください
5. データソースの種類を明示してください（例：「PDFデータによると」「Web検索の結果」「動画分析では」）

あなたの専門的な視点から、説得力のある応答を作成してください。
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
        """動画分析結果をフォーマット"""
        formatted = "\n\n【YouTube/TikTok動画分析】\n"
        formatted += f"- 総動画数: {analysis['total_videos']:,}件\n"
        formatted += f"- 総視聴回数: {analysis['total_views']:,}回\n"
        
        if analysis.get("party_analysis"):
            formatted += "\n【政党別動画分析】\n"
            for party, data in analysis["party_analysis"].items():
                if data['count'] > 0:
                    formatted += f"\n{party}関連:\n"
                    formatted += f"  - 動画数: {data['count']}件（総視聴回数: {data['total_views']:,}回）\n"
        
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
    title="知識蒸留 API Server - 単一モデル複数エージェント版 v3.1",
    description="改善版：単一モデルモードでも単一データソース使用、パフォーマンス最適化、学習効率改善、評価透明性向上",
    version="3.1.0"
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
                "data_sources": "PDF、Web検索、動画データから1つを選択（ランダムまたは指定）",
                "force_data_source_options": ["pdf", "web_search", "video_data"]
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