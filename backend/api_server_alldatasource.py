#!/usr/bin/env python3
"""
知識蒸留APIサーバー - 完全統合版
paste_fixed.pyとpaste.pyのすべての機能を実装
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

# プロジェクトのパスを設定（loggerの定義後に移動）
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message="Could get FontBBox")
warnings.filterwarnings("ignore", category=UserWarning)

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

# ====================
# Data Classes
# ====================
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

        # 要約用の設定を追加（ログの前に定義）
        self.summarization_model = "o3-2025-04-16"
        self.max_prompt_length = 50000

        # Log API status
        logger.info("API Keys Status:")
        logger.info(f"OpenAI: {'✓' if self.openai_client else '✗'}")
        logger.info(f"Anthropic: {'✓' if self.anthropic_client else '✗'}")
        logger.info(f"Gemini: {'✓' if self.gemini_model else '✗'}")
        logger.info(f"OpenRouter: {'✓' if self.openrouter_api_key else '✗'}")
        logger.info(f"Tavily: {'✓' if self.tavily_client else '✗'}")
        logger.info(f"Summarization: ✓ (using {self.summarization_model})")

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
        
        self.researcher_indicators = {
            'academic': ['研究', '論文', '学会', '博士', '教授', '准教授', '講師'],
            'analytical': ['分析', 'データ', '統計', '調査', '考察', '検証'],
            'political_science': ['政治学', '選挙', '政党', '議会', '投票行動', '世論']
        }
        
        self.election_keywords = {
            'ruling_party': ['自民党', '公明党', '与党', '政権党'],
            'opposition': ['野党', '立憲民主党', '日本維新の会', '国民民主党'],
            'new_parties': ['参政党', '新党', '政治団体'],
            'issues': ['カルト', 'SNS', 'ショート動画', 'TikTok', 'YouTube Shorts']
        }
    
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
    """Implements single-elimination tournament for response selection"""
    
    def __init__(self, persuasion_optimizer=None):
        self.sia = SentimentIntensityAnalyzer()
        self.persuasion_optimizer = persuasion_optimizer
    
    def _refine_response_with_llm(self, content: str, original_data: Dict[str, Any]) -> str:
        """LLMを使用して応答文を洗練させる（構造化維持版）"""
        
        # データから固有名と統計情報を抽出
        data_points = []
        video_examples = []
        
        if 'pdf_context' in original_data:
            # PDFから政党名、数値データを抽出
            import re
            party_names = re.findall(r'(自民党|公明党|参政党|国民民主党|立憲民主党|日本維新の会)', str(original_data.get('pdf_context', '')))
            data_points.extend(list(set(party_names)))
        
        if 'video_analysis' in original_data or 'json_analysis' in original_data:
            # 動画分析データから具体例を抽出
            analysis = original_data.get('video_analysis') or original_data.get('json_analysis', {})
            
            if analysis and isinstance(analysis, dict):
                # 総視聴回数
                if 'total_views' in analysis:
                    data_points.append(f"総視聴回数{analysis['total_views']:,}回")
                
                # トップ動画の例
                if 'top_viewed_videos' in analysis:
                    for video in analysis['top_viewed_videos'][:3]:
                        if isinstance(video, dict):
                            video_examples.append({
                                'title': video.get('title', ''),
                                'channel': video.get('channel', ''),
                                'views': video.get('view_count', 0),
                                'followers': video.get('channel_follower_count', 0)
                            })
                
                # トレンドキーワード
                if 'trending_topics' in analysis:
                    for topic, info in list(analysis['trending_topics'].items())[:3]:
                        if info['count'] > 0:
                            data_points.append(f"「{topic}」関連動画{info['count']}件")
        
        # 動画例を文章化
        video_examples_text = ""
        if video_examples:
            video_examples_text = "\n具体的に引用すべき動画例:\n"
            for v in video_examples[:3]:
                video_examples_text += f"- 「{v['title'][:50]}...」（{v['channel']}、{v['views']:,}回視聴）\n"
        
        refinement_prompt = f"""
    以下の文章を、日本の新聞記事として自然で読みやすい文章に改善してください。
    特に、以下の3段落構造を必ず維持してください：

    1. 第1段落：動画データの整理（具体的な動画タイトルやチャンネル名を含む）
    2. 第2段落：動画データ、PDF、Web検索結果の統合的考察
    3. 第3段落：結論と新たな視点の提示

    【改善が必要な文章】
    {content}

    【必ず組み込むべきデータ】
    {', '.join(data_points[:10]) if data_points else 'なし'}

    {video_examples_text}

    【改善の指示】
    1. 3段落構造を必ず維持する（各段落は2-3文以上）
    2. 第1段落では動画の具体例を必ず含める
    3. 第2段落では3つの情報源（動画、PDF、Web）の統合を明確にする
    4. 動画タイトルは「」で囲んで正確に引用する
    5. チャンネル名と視聴回数を併記する
    6. 段落間の接続を自然にする
    7. 600-800文字程度にまとめる

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
        
    def _extract_structured_data(self, content: str, source: DataSourceType, 
                                source_specific_content: Optional[Dict]) -> Dict[str, Any]:
        """コンテンツから構造化データを抽出"""
        structured_data = {
            'source_type': source.value,
            'key_points': [],
            'specific_data': {},
            'insights': []
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
                
                # トップ動画の情報を抽出
                if 'top_viewed_videos' in video_analysis:
                    for video in video_analysis['top_viewed_videos'][:3]:
                        if isinstance(video, dict):
                            structured_data['specific_data']['top_videos'].append({
                                'title': video.get('title', ''),
                                'channel': video.get('channel', ''),
                                'views': video.get('view_count', 0)
                            })
                
                # 動画トレンドから洞察を抽出
                structured_data['insights'].append("SNSでの拡散力が既存メディアを凌駕")
        
        elif source == DataSourceType.PDF and source_specific_content:
            # PDFデータから政策情報を抽出
            for key, value in source_specific_content.items():
                if 'pdf' in key and isinstance(value, dict):
                    sections = value.get('sections', [])
                    for section in sections:
                        # 政党名と政策を抽出
                        import re
                        parties = re.findall(r'(自民党|公明党|参政党|国民民主党|立憲民主党)', section)
                        policies = re.findall(r'(政策|公約|改革|支援|規制)', section)
                        
                        if parties:
                            structured_data['specific_data']['parties'] = list(set(parties))
                        if policies:
                            structured_data['specific_data']['policy_keywords'] = list(set(policies))
            
            structured_data['insights'].append("既存政党の政策では捉えきれない層の存在")
        
        elif source == DataSourceType.WEB_SEARCH and source_specific_content:
            # Web検索結果から専門家の分析を抽出
            for key, value in source_specific_content.items():
                if 'search' in key and isinstance(value, list):
                    for result in value[:3]:
                        if isinstance(result, dict):
                            structured_data['specific_data'].setdefault('articles', []).append({
                                'title': result.get('title', ''),
                                'source': result.get('url', '').split('/')[2] if result.get('url') else '',
                                'key_point': result.get('content', '')[:100]
                            })
            
            structured_data['insights'].append("専門家による多角的な要因分析")
        
        return structured_data
    
    def _create_pdf_based_response(self, winner: PersuasiveResponse, 
                                loser_data: Dict[str, Any], loser_source: DataSourceType) -> str:
        """PDFデータソースを基にした3段落構造の応答を生成"""
        
        # PDFデータから具体的な内容を抽出
        pdf_content = winner.content
        pdf_specific = winner.source_specific_content or {}
        
        # 第1段落：PDFデータの分析（自分のデータソース）
        paragraph1 = self._extract_first_paragraph(pdf_content)
        if not paragraph1:
            paragraph1 = "PDFデータによると、2025年参議院選挙において主要政党が掲げた公約は、経済対策や子育て支援、政治資金の透明化など、従来型の政策テーマが中心でした。"
        
        # 第2段落：統合的考察（敗者の洞察を統合）
        paragraph2 = "政策文書の分析から見えてくるのは、既存政党が提供する政策だけでは説明できない有権者層の存在です。"
        
        if loser_source == DataSourceType.VIDEO_DATA and loser_data.get('specific_data'):
            views = loser_data['specific_data'].get('total_views', 0)
            if views > 0:
                paragraph2 += f"動画データが示す{views:,}回という膨大な視聴回数は、従来の政治報道では捉えきれない情報拡散の実態を物語っています。"
            else:
                paragraph2 += "SNS動画による情報拡散は、従来の政治報道では捉えきれない層に到達していることが示唆されます。"
        elif loser_source == DataSourceType.WEB_SEARCH:
            paragraph2 += "メディア分析が指摘する「既存の枠組みへの不信感」は、政策文書に現れない深層的な要因として重要です。"
        
        paragraph2 += "このような多面的な視点から見ると、参政党の躍進は単なる選挙戦術の成功ではなく、社会的な変化の表れと解釈できます。"
        
        # 第3段落：結論
        paragraph3 = "以上の分析から、参政党の躍進はSNS戦略だけでなく、既存政党の政策が捉えきれない有権者の不満や期待、"
        paragraph3 += "そして新しいメディア環境における情報伝達の変化が複合的に作用した結果と言えるでしょう。"
        paragraph3 += "あなたの考える「SNSやショート動画の戦略」という単一要因では、この現象の本質は説明できません。"
        
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
            paragraph1 += "動画データの分析によると、SNSでの政治コンテンツが unprecedented な拡散力を示しています。"
        
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
    
    def _extract_first_paragraph(self, content: str) -> str:
        """コンテンツから最初の段落を抽出"""
        paragraphs = content.split('\n\n')
        if paragraphs:
            return paragraphs[0].strip()
        
        # 改行がない場合は最初の2文を抽出
        sentences = content.split('。')
        if len(sentences) >= 2:
            return '。'.join(sentences[:2]) + '。'
        
        return content

    def _split_into_natural_paragraphs(self, content: str) -> List[str]:
        """文章を自然な段落に分割（改善版）"""
        # 改行で分割
        parts = content.split('\\n\\n')
        
        paragraphs = []
        current_paragraph = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            sentences = part.split('。')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # 現在の段落に追加
                if current_paragraph:
                    current_paragraph += f"{sentence}。"
                else:
                    current_paragraph = f"{sentence}。"
                
                # 段落の長さが適切になったら新しい段落を開始
                if len(current_paragraph) > 150 or self._is_paragraph_end(sentence):
                    paragraphs.append(current_paragraph)
                    current_paragraph = ""
        
        # 最後の段落を追加
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        # 短すぎる段落を結合
        final_paragraphs = []
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]
            
            # 1文だけの段落は次と結合
            sentence_count = para.count('。')
            if sentence_count <= 1 and i + 1 < len(paragraphs):
                next_para = paragraphs[i + 1]
                connector = self._select_natural_connector(para, next_para)
                final_paragraphs.append(f"{para}{connector}{next_para}")
                i += 2
            else:
                final_paragraphs.append(para)
                i += 1
        
        return final_paragraphs

    def _is_paragraph_end(self, sentence: str) -> bool:
        """段落の終わりかどうかを判定"""
        # 結論的な表現
        conclusion_markers = ['である', 'でした', 'います', 'でしょう', 'と考えられます', 'と言えます']
        return any(marker in sentence for marker in conclusion_markers)

    def _select_natural_connector(self, para1: str, para2: str) -> str:
        """2つの段落を自然につなぐ接続詞を選択"""
        # para1の最後とpara2の最初の内容を分析
        
        if 'データ' in para2 or '数字' in para2 or '%' in para2:
            return "実際のデータを見ると、"
        elif '一方' in para2 or 'しかし' in para2:
            return ""  # すでに接続詞がある
        elif '要因' in para2 or '理由' in para2:
            return "この背景には、"
        elif '分析' in para2 or '考察' in para2:
            return "詳しく分析すると、"
        else:
            connectors = ["", "また、", "さらに言えば、", "これに関連して、"]
            return random.choice(connectors)

    def _ensure_paragraph_flow(self, content: str) -> str:
        """段落間の流れを確認し改善"""
        paragraphs = content.split('\\n\\n')
        improved_paragraphs = []
        
        for i, para in enumerate(paragraphs):
            if i > 0:
                # 前の段落との関係を確認
                prev_para = paragraphs[i-1]
                
                # 同じような始まり方を避ける
                if para.startswith('また、') and prev_para.count('また') > 0:
                    para = para.replace('また、', 'さらに、', 1)
                elif para.startswith('さらに、') and prev_para.count('さらに') > 0:
                    para = para.replace('さらに、', 'これに加えて、', 1)
            
            improved_paragraphs.append(para)
        
        return '\\n\\n'.join(improved_paragraphs)

    def run_tournament(self, responses: List[PersuasiveResponse]) -> PersuasiveResponse:
        """Run single-elimination tournament"""
        for response in responses:
            response.persuasion_score = self._score_response(response)
        
        contestants = responses.copy()
        random.shuffle(contestants)
        
        while len(contestants) > 1:
            next_round = []
            
            for i in range(0, len(contestants), 2):
                if i + 1 < len(contestants):
                    winner = self._compare_responses(contestants[i], contestants[i + 1])
                    next_round.append(winner)
                else:
                    next_round.append(contestants[i])
            
            contestants = next_round
        
        return contestants[0]
    
    def _calculate_score_improvement(self, original_score: float, new_score: float) -> Dict[str, float]:
        """スコアの改善を計算"""
        improvement = new_score - original_score
        improvement_percentage = (improvement / original_score * 100) if original_score > 0 else 0
        
        return {
            'original_score': original_score,
            'new_score': new_score,
            'improvement': improvement,
            'improvement_percentage': improvement_percentage,
            'improved': improvement > 0
        }
    
    def _log_learning_failure(self, winner_model: str, loser_model: str, original_score: float, 
                            best_attempt_score: float, reason: str = ""):
        """学習失敗時の詳細ログ"""
        logger.warning(f"""
    Learning Failed:
    - Winner: {winner_model} (Original Score: {original_score:.2f})
    - Loser: {loser_model}
    - Best Attempt Score: {best_attempt_score:.2f}
    - Score Difference: {best_attempt_score - original_score:.2f}
    - Reason: {reason or 'Score not improved'}
    - Action: Keeping original response
    """)

    def run_tournament_with_learning_tracking(self, responses: List[PersuasiveResponse], 
                                            enable_learning: bool = True) -> Tuple[PersuasiveResponse, List[Dict]]:
        """学習機能付きトーナメントを実行し、各ラウンドの結果を記録（スコア保持版）"""
        tournament_log = []
        
        # スコアを初期化
        for response in responses:
            response.persuasion_score = self._score_response(response)
        
        initial_scores = [(r.model_used, r.persuasion_score) for r in responses]
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
            
            for i in range(0, len(contestants), 2):
                if i + 1 < len(contestants):
                    contestant1 = contestants[i]
                    contestant2 = contestants[i + 1]
                    
                    if enable_learning:
                        winner = self._compare_responses_with_learning(contestant1, contestant2)
                    else:
                        winner = self._compare_responses(contestant1, contestant2)
                    
                    round_results.append({
                        'match': f"{contestant1.model_used} vs {contestant2.model_used}",
                        'scores': f"{contestant1.persuasion_score:.2f} vs {contestant2.persuasion_score:.2f}",
                        'winner': winner.model_used,
                        'winner_improved_score': f"{winner.persuasion_score:.2f}",
                        'learning_applied': enable_learning and hasattr(winner, 'learning_history') and len(winner.learning_history) > 0
                    })
                    
                    next_round.append(winner)
                else:
                    next_round.append(contestants[i])
                    round_results.append({
                        'match': f"{contestants[i].model_used} (bye)",
                        'scores': f"{contestants[i].persuasion_score:.2f}",
                        'winner': contestants[i].model_used,
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
        
        # 最終的な勝者の応答を洗練（スコア保持版）
        if enable_learning and hasattr(self, '_refine_response_with_llm'):
            try:
                logger.info("Attempting final refinement with LLM...")
                
                # 元のコンテンツとスコアを保存
                original_final_content = final_winner.content
                original_final_score = final_winner.persuasion_score
                
                # 洗練用のデータを準備
                original_data = {
                    'json_analysis': getattr(final_winner, 'json_analysis', {}),
                    'pdf_context': getattr(final_winner, 'pdf_context', ''),
                }
                
                # LLMによる洗練を試みる
                refined_content = self._refine_response_with_llm(
                    final_winner.content,
                    original_data
                )
                
                # 洗練された応答のスコアを計算
                refined_response = PersuasiveResponse(
                    content=refined_content,
                    treatment_condition=final_winner.treatment_condition,
                    persuasion_score=0.0,
                    model_used=final_winner.model_used,
                    generation_params=final_winner.generation_params,
                    user_profile=final_winner.user_profile,
                    pdf_context=final_winner.pdf_context
                )
                
                refined_score = self._score_response(refined_response)
                
                logger.info(f"Final refinement score: {refined_score:.2f} (original: {original_final_score:.2f})")
                
                # スコアが改善された場合のみ洗練されたコンテンツを使用
                if refined_score > original_final_score:
                    final_winner.content = refined_content
                    final_winner.persuasion_score = refined_score
                    
                    tournament_log.append({
                        'round': 'final_refinement',
                        'type': 'llm_refinement',
                        'original_score': original_final_score,
                        'refined_score': refined_score,
                        'improvement': refined_score - original_final_score,
                        'status': 'success',
                        'original_length': len(original_final_content),
                        'refined_length': len(refined_content)
                    })
                    
                    logger.info(f"Final refinement successful: Score improved by {refined_score - original_final_score:.2f}")
                else:
                    # スコアが改善されなかった場合は元のコンテンツを保持
                    logger.warning(f"Final refinement failed to improve score, keeping original content")
                    
                    tournament_log.append({
                        'round': 'final_refinement',
                        'type': 'llm_refinement',
                        'original_score': original_final_score,
                        'refined_score': refined_score,
                        'improvement': refined_score - original_final_score,
                        'status': 'failed_no_improvement',
                        'action': 'kept_original'
                    })
                    
                    # 元のコンテンツとスコアを保持
                    final_winner.content = original_final_content
                    final_winner.persuasion_score = original_final_score
                
            except Exception as e:
                logger.error(f"Failed to refine final response: {e}")
                tournament_log.append({
                    'round': 'final_refinement',
                    'type': 'llm_refinement',
                    'status': 'error',
                    'error': str(e)
                })
                # エラーの場合も元のコンテンツを保持
        
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
                'total_score_improvement': sum(h.get('improved_score', 0) - h.get('original_score', 0) for h in successful_learnings)
            })
        
        return final_winner, tournament_log
    
    def _score_response(self, response: PersuasiveResponse) -> float:
        """政治分析投稿に適したスコアリング"""
        score = 0.0
        content = response.content
        
        char_count = len(content)
        if 400 <= char_count <= 600:
            score += 1.0
        else:
            score += max(0, 1.0 - abs(char_count - 500) / 500)
        
        academic_terms = ['分析', '考察', '要因', '背景', '傾向', '影響', '構造', '動向']
        academic_score = sum(1 for term in academic_terms if term in content)
        score += min(1.0, academic_score / 4)
        
        data_keywords = ['データ', '統計', '調査', '%', '割合', '票', '議席']
        data_score = sum(1 for keyword in data_keywords if keyword in content)
        score += min(1.0, data_score / 3)
        
        reference_patterns = ['によると', '参考資料', '出典', 'より引用', '参照']
        ref_score = sum(1 for pattern in reference_patterns if pattern in content)
        score += min(1.0, ref_score / 2)
        
        extreme_words = ['絶対', '完全に', '間違いなく', '明らかに誤り']
        neutrality_penalty = sum(0.2 for word in extreme_words if word in content)
        score -= neutrality_penalty
        
        connectives = ['しかし', 'また', '一方で', 'さらに', 'つまり', 'したがって']
        structure_score = sum(1 for conn in connectives if conn in content)
        score += min(1.0, structure_score / 3)
        
        if response.generation_params.get('pdf_references'):
            score += 0.5
        
        return max(0, score)
    
    def _compare_responses(self, resp1: PersuasiveResponse, resp2: PersuasiveResponse) -> PersuasiveResponse:
        """Compare two responses and return winner"""
        if resp1.persuasion_score > resp2.persuasion_score:
            return resp1
        elif resp2.persuasion_score > resp1.persuasion_score:
            return resp2
        
        if resp1.treatment_condition == TreatmentCondition.PERSONALIZATION:
            return resp1
        elif resp2.treatment_condition == TreatmentCondition.PERSONALIZATION:
            return resp2
        
        return random.choice([resp1, resp2])
    
    def _compare_responses_with_learning(self, resp1: PersuasiveResponse, resp2: PersuasiveResponse) -> PersuasiveResponse:
        """強化された知識統合を伴う応答比較（修正版）"""
        
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
        
        # 元のスコアとコンテンツを保存
        original_score = basic_winner.persuasion_score
        original_content = basic_winner.content
        
        # 勝者と敗者のデータソースタイプを確認
        winner_source = basic_winner.data_source_type
        loser_source = loser.data_source_type
        
        logger.info(f"Knowledge integration: {basic_winner.model_used} ({winner_source.value if winner_source else 'unknown'}) learning from {loser.model_used} ({loser_source.value if loser_source else 'unknown'})")
        
        # 異なるデータソースの場合、知識を統合
        if winner_source and loser_source and winner_source != loser_source:
            # データソースに応じた統合戦略を選択
            improved_content = self._integrate_with_source_constraints(
                basic_winner, loser, winner_source, loser_source
            )
            
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
                source_specific_content=self._merge_source_contents(
                    basic_winner.source_specific_content,
                    loser.source_specific_content,
                    winner_source,
                    loser_source
                ),
                tournament_round=basic_winner.tournament_round
            )
            
            # 学習履歴を記録
            improved_winner.learning_history = getattr(basic_winner, 'learning_history', []).copy()
            
            # 改善後のスコアを計算
            improved_winner.detailed_score = self._score_response_with_details(improved_winner)
            improved_winner.persuasion_score = improved_winner.detailed_score.total
            
            logger.info(f"Score after knowledge integration: {improved_winner.persuasion_score:.2f} (original: {original_score:.2f})")
            
            # 学習履歴を追加
            improved_winner.learning_history.append({
                'learned_from': f"{loser.model_used} ({loser_source.value})",
                'original_score': original_score,
                'improved_score': improved_winner.persuasion_score,
                'opponent_score': loser.persuasion_score,
                'cross_source_learning': True,
                'source_types': {'winner': winner_source.value, 'loser': loser_source.value},
                'status': 'success',
                'integration_type': 'constrained'
            })
            
            return improved_winner
        
        return basic_winner
    
    def _score_response_with_details(self, response: PersuasiveResponse) -> DetailedScore:
        """詳細なスコアリング - マルチソース統合を最重視"""
        breakdown = {
            'base_quality': 0.0,
            'data_richness': 0.0,
            'source_coverage': 0.0,
            'integration_quality': 0.0,
            'learning_bonus': 0.0,
            'depth_bonus': 0.0,
            'knowledge_synthesis': 0.0  # 新規追加：知識統合の評価
        }
        
        content = response.content
        
        # 基本的な文字数評価
        char_count = len(content)
        if 400 <= char_count <= 600:
            breakdown['base_quality'] += 1.0
        else:
            breakdown['base_quality'] += max(0, 1.0 - abs(char_count - 500) / 500)
        
        # 学術的な表現の評価
        academic_terms = ['分析', '考察', '要因', '背景', '傾向', '影響', '構造', '動向']
        academic_score = sum(1 for term in academic_terms if term in content)
        breakdown['base_quality'] += min(1.0, academic_score / 4)
        
        # データキーワードの評価
        data_keywords = ['データ', '統計', '調査', '%', '割合', '票', '議席', '視聴回数']
        data_score = sum(1 for keyword in data_keywords if keyword in content)
        breakdown['data_richness'] = min(1.5, data_score / 2)
        
        # マルチソース統合の評価（全ラウンドで重視）
        pdf_mentioned = self._check_pdf_content(content)
        web_mentioned = self._check_web_content(content)
        video_mentioned = self._check_video_content(content)
        
        source_count = sum([pdf_mentioned, web_mentioned, video_mentioned])
        
        # 各データソースの深度をチェック
        pdf_depth = self._evaluate_source_depth(content, 'pdf')
        web_depth = self._evaluate_source_depth(content, 'web')
        video_depth = self._evaluate_source_depth(content, 'video')
        
        # ソースカバレッジスコア（3つ全て含む場合に最高点）
        breakdown['source_coverage'] = source_count * 1.0
        if source_count == 3:
            breakdown['source_coverage'] += 2.0  # ボーナス
        
        # 各ソースの深度を評価
        breakdown['depth_bonus'] = (pdf_depth + web_depth + video_depth) / 3
        
        # 統合的な分析を示す表現の評価
        integration_keywords = [
            '統合', '総合的', '多角的', '横断的', '複合的', '包括的',
            '一方で', 'これに対し', 'さらに', '加えて', '組み合わせ'
        ]
        integration_score = sum(1 for keyword in integration_keywords if keyword in content)
        breakdown['integration_quality'] = min(2.0, integration_score * 0.5)
        
        # 知識統合の評価（新規追加）
        synthesis_indicators = [
            'PDFデータ.*Web検索',
            'Web検索.*動画',
            'PDF.*動画',
            '3つのデータソース',
            '複数の観点',
            'それぞれのデータ'
        ]
        import re
        synthesis_score = sum(1 for pattern in synthesis_indicators 
                             if re.search(pattern, content))
        breakdown['knowledge_synthesis'] = min(3.0, synthesis_score)
        
        # 学習履歴による加点
        if hasattr(response, 'learning_history') and response.learning_history:
            learning_bonus = len(response.learning_history) * 0.5
            # クロスソース学習の場合は追加ボーナス
            cross_source_learnings = [h for h in response.learning_history 
                                     if h.get('cross_source_learning')]
            learning_bonus += len(cross_source_learnings) * 1.0
            breakdown['learning_bonus'] = min(3.0, learning_bonus)
        
        # 重み付けを適用（マルチソース統合を最重視）
        weights = {
            'base_quality': 0.5,
            'data_richness': 0.8,
            'source_coverage': 2.0,  # 最重要
            'integration_quality': 1.5,
            'depth_bonus': 1.0,
            'knowledge_synthesis': 2.0,  # 最重要
            'learning_bonus': 1.0
        }
        
        weighted_total = sum(
            breakdown[key] * weights.get(key, 1.0)
            for key in breakdown
        )
        
        # 強みと弱みの特定
        winning_factors = []
        missing_elements = []
        
        if breakdown['source_coverage'] >= 3.0:
            winning_factors.append("全データソースの完全統合")
        if breakdown['knowledge_synthesis'] >= 2.0:
            winning_factors.append("優れた知識統合")
        if breakdown['data_richness'] >= 1.0:
            winning_factors.append("豊富なデータ引用")
        
        if breakdown['source_coverage'] < 2.0:
            missing_elements.append("データソースの不足")
        if breakdown['knowledge_synthesis'] < 1.0:
            missing_elements.append("知識統合の欠如")
        
        return DetailedScore(
            total=weighted_total,
            breakdown=breakdown,
            winning_factors=winning_factors,
            missing_elements=missing_elements,
            tournament_round=response.tournament_round
        )

    def _extract_response_elements(self, response: PersuasiveResponse) -> Dict[str, List[str]]:
        """応答から重要な要素を抽出"""
        content = response.content
        elements = {
            'data_points': [],
            'key_insights': [],
            'references': [],
            'narrative_elements': [],
            'structural_elements': []
        }
        
        # データポイントの抽出（数値、統計）- 改善版
        import re
        data_patterns = [
            r'\d{2,}[,\d]*[件回票%％]',  # 2桁以上の数値のみ
            r'\d+\.\d+[倍%％]',  # 小数点を含む数値
            r'約?\d{2,}[,\d]*万?',  # 2桁以上の万単位
            r'\d+対\d+',
        ]
        for pattern in data_patterns:
            matches = re.findall(pattern, content)
            # 無意味なデータを除外
            filtered_matches = [m for m in matches if '0%' not in m and '00' not in m and len(m) > 2]
            elements['data_points'].extend(filtered_matches)
        
        # 重要な洞察の抽出（キーワードベース）
        insight_keywords = ['特筆すべき', '重要な', '特徴的', '注目すべき', 'ポイント', '要因']
        sentences = content.split('。')
        for sentence in sentences:
            if any(keyword in sentence for keyword in insight_keywords):
                elements['key_insights'].append(sentence.strip())
        
        # 参照・引用の抽出
        reference_patterns = ['によると', '参考資料', '出典', 'データ', '分析']
        for sentence in sentences:
            if any(pattern in sentence for pattern in reference_patterns):
                elements['references'].append(sentence.strip())
        
        # ナラティブ要素の抽出
        narrative_keywords = ['ストーリー', 'ナラティブ', '物語', '構造', 'システム']
        for sentence in sentences:
            if any(keyword in sentence for keyword in narrative_keywords):
                elements['narrative_elements'].append(sentence.strip())
        
        # 構造的要素（接続詞、論理展開）
        structural_markers = ['まず', '次に', 'さらに', 'また', '一方で', 'しかし', 'つまり', 'したがって']
        for sentence in sentences:
            if any(marker in sentence for marker in structural_markers):
                elements['structural_elements'].append(sentence.strip())
        
        return elements

    def _identify_valuable_elements(self, winner_elements: Dict, loser_elements: Dict) -> Dict[str, List[str]]:
        """敗者から勝者にない価値ある要素を特定"""
        valuable = {
            'unique_data': [],
            'unique_insights': [],
            'complementary_analysis': []
        }
        
        # 勝者にないユニークなデータポイント
        winner_data_set = set(winner_elements['data_points'])
        for data in loser_elements['data_points']:
            if data not in winner_data_set:
                valuable['unique_data'].append(data)
        
        # 勝者にない洞察
        winner_insights_text = ' '.join(winner_elements['key_insights'])
        for insight in loser_elements['key_insights']:
            # 意味的に重複していないかチェック（簡易版）
            if not any(word in winner_insights_text for word in insight.split()[:3]):
                valuable['unique_insights'].append(insight)
        
        # 補完的な分析視点
        winner_full_text = ' '.join([
            ' '.join(winner_elements.get(key, [])) 
            for key in winner_elements
        ])
        
        for element in loser_elements['narrative_elements'] + loser_elements['references']:
            if element and len(element) > 20:  # 意味のある長さの要素のみ
                # 勝者にない視点かチェック
                key_words = [word for word in element.split() if len(word) > 3][:3]
                if not all(word in winner_full_text for word in key_words):
                    valuable['complementary_analysis'].append(element)
        
        return valuable
    
    def _integrate_elements(self, winner: PersuasiveResponse, valuable_elements: Dict) -> str:
        """価値ある要素を勝者の構造に統合（自然な文章を維持）"""
        original_content = winner.content
        
        # まず後処理を適用して基準となる文章を整える
        if hasattr(self.persuasion_optimizer, '_post_process_response'):
            original_content = self.persuasion_optimizer._post_process_response(original_content)
        
        # 段落単位で分割（。で終わる文のグループ）
        paragraphs = self._split_into_paragraphs(original_content)
        
        # 統合戦略：段落構造を維持しながら要素を追加
        enhanced_paragraphs = []
        data_inserted = False
        insight_inserted = False
        
        for i, paragraph in enumerate(paragraphs):
            enhanced_paragraph = paragraph
            
            # データポイントの自然な統合（修正版）
            if not data_inserted and valuable_elements['unique_data'] and any(word in paragraph for word in ['動画', 'データ', '分析']):
                data_points = valuable_elements['unique_data'][:2]
                if data_points:
                    # データポイントの内容を確認して、意味のある文章を生成
                    valid_data = []
                    for data in data_points:
                        # 数値データの妥当性を確認
                        if data and len(str(data)) > 2 and '0%' not in str(data) and '00' not in str(data):
                            valid_data.append(data)
                    
                    if valid_data:
                        # より自然な文章として統合
                        if len(valid_data) == 1:
                            data_sentence = f"データによると、{valid_data[0]}という興味深い傾向が見られます"
                        else:
                            data_sentence = f"データ分析から、{valid_data[0]}や{valid_data[1]}といった傾向が明らかになっています"
                        enhanced_paragraph = f"{paragraph}。{data_sentence}"
                        data_inserted = True
                
            enhanced_paragraphs.append(enhanced_paragraph)
            
            # 洞察の自然な統合（段落の間に挿入）
            if not insight_inserted and i == len(paragraphs) // 2 and valuable_elements['unique_insights']:
                insight = valuable_elements['unique_insights'][0]
                # 接続詞を使って自然につなげる
                transition_phrases = [
                    "ここで重要なのは、",
                    "さらに興味深いことに、",
                    "別の視点から見ると、",
                    "加えて注目すべきは、"
                ]
                transition = random.choice(transition_phrases)
                insight_paragraph = f"{transition}{insight}"
                enhanced_paragraphs.append(insight_paragraph)
                insight_inserted = True
        
        # 補完的分析を最後に追加（ただし結論は保持）
        if valuable_elements['complementary_analysis'] and len(enhanced_paragraphs) > 1:
            # 最後の段落（結論）を保存
            conclusion = enhanced_paragraphs[-1]
            
            # 補完的分析を追加
            analysis = valuable_elements['complementary_analysis'][0]
            transition_to_conclusion = [
                "このような観点を踏まえると、",
                "以上の分析から言えることは、",
                "結論として、"
            ]
            transition = random.choice(transition_to_conclusion)
            
            # 分析を挿入し、結論につなげる
            enhanced_paragraphs.insert(-1, f"{analysis}。{transition}{conclusion}")
            enhanced_paragraphs.pop()  # 元の結論を削除
        
        # 段落を結合
        enhanced_content = '\n\n'.join(enhanced_paragraphs)
        
        # 最終的な後処理
        if hasattr(self.persuasion_optimizer, '_post_process_response'):
            enhanced_content = self.persuasion_optimizer._post_process_response(enhanced_content)
        
        # 文字数調整（自然な切れ目で）
        original_length = len(original_content)
        if len(enhanced_content) > original_length * 1.3:
            enhanced_content = self._natural_truncate(enhanced_content, int(original_length * 1.3))
        
        return enhanced_content
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """文章を自然な段落に分割"""
        # まず改行で分割
        parts = content.split('\n\n')
        
        paragraphs = []
        for part in parts:
            part = part.strip()
            if part:
                # 長すぎる段落はさらに分割
                if len(part) > 300:
                    sentences = part.split('。')
                    current_paragraph = ""
                    for sentence in sentences:
                        if sentence.strip():
                            if len(current_paragraph) + len(sentence) > 200:
                                if current_paragraph:
                                    paragraphs.append(current_paragraph + '。')
                                current_paragraph = sentence
                            else:
                                current_paragraph += sentence + '。'
                    if current_paragraph:
                        paragraphs.append(current_paragraph)
                else:
                    paragraphs.append(part)
        
        return paragraphs
    
    def _natural_truncate(self, content: str, max_length: int) -> str:
        """自然な位置で文章を切り詰める"""
        if len(content) <= max_length:
            return content
        
        # 段落単位で切り詰め
        paragraphs = self._split_into_paragraphs(content)
        truncated = []
        current_length = 0
        
        for paragraph in paragraphs:
            if current_length + len(paragraph) + 2 <= max_length:  # +2 for \n\n
                truncated.append(paragraph)
                current_length += len(paragraph) + 2
            else:
                # 最後の段落は文単位で調整
                sentences = paragraph.split('。')
                for sentence in sentences:
                    if current_length + len(sentence) + 1 <= max_length:
                        if truncated:
                            truncated[-1] += sentence + '。'
                        else:
                            truncated.append(sentence + '。')
                        current_length += len(sentence) + 1
                    else:
                        break
                break
        
        result = '\n\n'.join(truncated)
        
        # 文末が適切でない場合は修正
        if result and not result.endswith(('。', '！', '？', '」')):
            # 最後の不完全な文を削除
            last_period = result.rfind('。')
            if last_period > 0:
                result = result[:last_period + 1]
        
        return result
    
    def _alternative_integration(self, winner: PersuasiveResponse, loser: PersuasiveResponse) -> str:
        """代替の統合方法：より構造的なアプローチ"""
        winner_content = winner.content
        loser_content = loser.content
        
        # 両者の強みを分析
        winner_strengths = self._analyze_strengths(winner_content)
        loser_strengths = self._analyze_strengths(loser_content)
        
        # 構造的な統合
        integrated_parts = []
        
        # 1. 導入部（勝者の導入が優れている場合はそれを使用）
        intro_winner = self._extract_introduction(winner_content)
        intro_loser = self._extract_introduction(loser_content)
        
        if len(intro_winner) > len(intro_loser):
            integrated_parts.append(intro_winner)
        else:
            integrated_parts.append(intro_loser)
        
        # 2. 本論（両者の優れた分析を組み合わせる）
        winner_analysis = self._extract_main_analysis(winner_content)
        loser_analysis = self._extract_main_analysis(loser_content)
        
        # データの豊富さで判断
        if winner_content.count('％') + winner_content.count('%') > loser_content.count('％') + loser_content.count('%'):
            integrated_parts.append(winner_analysis)
            # 敗者から補完的なデータを追加
            loser_data = re.findall(r'\d+[,\d]*[件回票%％]', loser_analysis)
            if loser_data:
                integrated_parts.append(f"なお、{loser_data[0]}という観点も重要である")
        else:
            integrated_parts.append(loser_analysis)
        
        # 3. 結論（より説得力のある方を選択）
        conclusion_winner = self._extract_conclusion(winner_content)
        conclusion_loser = self._extract_conclusion(loser_content)
        
        if '示唆' in conclusion_winner or '重要' in conclusion_winner:
            integrated_parts.append(conclusion_winner)
        elif '示唆' in conclusion_loser or '重要' in conclusion_loser:
            integrated_parts.append(conclusion_loser)
        else:
            integrated_parts.append(conclusion_winner)
        
        return '。'.join(integrated_parts)
    
    def _analyze_strengths(self, content: str) -> Dict[str, float]:
        """コンテンツの強みを分析"""
        strengths = {
            'data_richness': len(re.findall(r'\d+[,\d]*[件回票%％]', content)),
            'analytical_depth': content.count('分析') + content.count('要因') + content.count('背景'),
            'structural_clarity': content.count('まず') + content.count('次に') + content.count('さらに'),
            'persuasiveness': content.count('重要') + content.count('特筆') + content.count('注目')
        }
        return strengths
    
    def _extract_introduction(self, content: str) -> str:
        """導入部を抽出"""
        sentences = content.split('。')
        if len(sentences) > 0:
            return sentences[0]
        return ""
    
    def _extract_main_analysis(self, content: str) -> str:
        """本論部分を抽出"""
        sentences = content.split('。')
        if len(sentences) > 2:
            return '。'.join(sentences[1:-1])
        return content
    
    def _extract_conclusion(self, content: str) -> str:
        """結論部を抽出"""
        sentences = content.split('。')
        if len(sentences) > 0:
            return sentences[-1]
        return ""
    
    def _smart_truncate(self, content: str, max_length: int) -> str:
        """スマートな文字数調整"""
        if len(content) <= max_length:
            return content
        
        sentences = content.split('。')
        truncated = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) + 1 <= max_length:
                truncated.append(sentence)
                current_length += len(sentence) + 1
            else:
                break
        
        return '。'.join(truncated)

    def extract_good_points_sync(self, winner: PersuasiveResponse, loser: PersuasiveResponse) -> str:
        """敗者の応答から良い点を抽出し、勝者の応答に統合する"""
        extraction_prompt = f"""
あなたは政治分析の専門家です。以下の2つの分析を比較し、敗者の分析から勝者の分析に取り入れるべき良い点を見つけてください。

勝者（{winner.model_used}）の分析:
{winner.content}

敗者（{loser.model_used}）の分析:
{loser.content}

タスク:
1. 敗者の分析から、勝者の分析に含まれていない以下の価値ある要素を特定してください：
- 具体的なデータや統計
- 独自の視点や洞察
- 重要な歴史的文脈
- 見落とされている要因
- より明確な説明方法

2. これらの要素を勝者の分析に自然に統合し、より包括的で説得力のある分析を作成してください。

3. 統合する際は：
- 元の勝者の論理構造を維持する
- 400-600文字程度にまとめる

改善された分析のみを出力してください。
"""
        
        try:
            if self.persuasion_optimizer and hasattr(self.persuasion_optimizer, 'api_config'):
                api_config = self.persuasion_optimizer.api_config
                
                if 'gpt' in winner.model_used.lower() and api_config.openai_client:
                    response = api_config.openai_client.chat.completions.create(
                        model="gpt-4.1-2025-04-14",
                        messages=[{"role": "user", "content": extraction_prompt}],
                        temperature=0.7,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content
                
                elif 'claude' in winner.model_used.lower() and api_config.anthropic_client:
                    response = api_config.anthropic_client.messages.create(
                        model="claude-sonnet-4-20250514",
                        messages=[{"role": "user", "content": extraction_prompt}],
                        max_tokens=2000,
                        temperature=0.7
                    )
                    return response.content[0].text
            
            return winner.content
            
        except Exception as e:
            logger.error(f"Error in extracting good points: {str(e)}")
            return winner.content

# ====================
# Persuasion Optimizer
# ====================
class PersuasionOptimizer:
    """Core optimization algorithm for generating persuasive responses"""

    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        # モデルリストを拡張
        self.models = []
        if api_config.openai_client:
            self.models.append('gpt-4.1')
        if api_config.anthropic_client:
            self.models.append('claude-sonnet-4')
        if api_config.gemini_model:
            self.models.append('gemini-2.5-pro')
        if api_config.openrouter_api_key:
            self.models.append('llama-3.3-70b')  # OpenRouter経由のLlama
        
        # モデルが一つもない場合のフォールバック
        if not self.models:
            logger.warning("No LLM APIs configured. Using fallback mode.")
            self.models = ['fallback']
        
        self.community_aligned_model = 'gpt-4.1-finetuned'
        self.pdf_cache = {}
        self.text_cleaner = PDFTextCleaner()
        self.json_cache = {}
        self.search_cache = {}
        self.tournament_selector = TournamentSelector(persuasion_optimizer=self)
        
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
    
    def summarize_pdf_content(self, pdf_content: str, pdf_name: str, max_length: int = 3000) -> str:
        """PDFコンテンツをLLMで要約し、重要な情報を保持"""
        if len(pdf_content) <= max_length:
            return pdf_content
        
        logger.info(f"Summarizing PDF content: {pdf_name} (original: {len(pdf_content)} chars)")
        
        summarize_prompt = f"""
    以下のPDFコンテンツを要約してください。
    重要な指示：
    1. 固有名詞（政党名、人名、地名など）は必ず残す
    2. 具体的な数値データ（％、票数、金額など）は必ず残す
    3. 重要な政策や主張の具体例を3-5個選んで残す
    4. 要約は{max_length}文字以内にまとめる
    5. 箇条書きや番号付けは使わず、自然な文章で要約する

    PDFファイル名: {pdf_name}
    元のコンテンツ:
    {pdf_content[:10000]}  # 最初の10000文字のみをLLMに送る

    要約:"""
        
        try:
            # 軽量なモデルで要約
            if self.api_config.openai_client:
                response = self.api_config.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",  # より高速で安価なモデル
                    messages=[
                        {"role": "system", "content": "あなたは政治文書の要約専門家です。"},
                        {"role": "user", "content": summarize_prompt}
                    ],
                    temperature=0.3,  # より決定的な出力
                    max_tokens=max_length // 2  # 日本語は約2文字/トークン
                )
                summarized = response.choices[0].message.content
                logger.info(f"PDF summarized: {len(summarized)} chars")
                return summarized
            else:
                # フォールバック：単純な切り詰め
                return pdf_content[:max_length]
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return pdf_content[:max_length]

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
            
            # 要約処理を追加
            if summarize and len(text) > 5000:
                text = self.summarize_pdf_content(text, Path(pdf_path).name, max_length=3000)
            
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
        """YouTube/TikTok動画データを詳細分析（動画タイトル・内容分析強化版）"""
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
        
        # タイトルから抽出する重要キーワード
        important_keywords = [
            '悲報', '速報', '解体', 'デモ', '交渉失敗', '落選', '土下座',
            '戦犯', '橋下徹', '神谷宗幣', '石破総理', '税金', 'トランプ'
        ]
        keyword_appearances = {kw: [] for kw in important_keywords}
        
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
                
                # 政党別分析
                for party, data in party_mentions.items():
                    if party in title or any(kw in video.get("matched_keywords", []) for kw in ['自民', '参政', '立憲', '国民', '公明']):
                        data['count'] += 1
                        data['total_views'] += view_count
                        data['videos'].append({
                            'title': title,
                            'channel': channel,
                            'view_count': view_count,
                            'channel_followers': video.get("channel_follower_count", 0)
                        })
                
                # 重要キーワード分析
                for keyword in important_keywords:
                    if keyword in title:
                        keyword_appearances[keyword].append({
                            'title': title,
                            'channel': channel,
                            'view_count': view_count
                        })
                
                # マッチしたキーワードごとの動画収集
                for keyword in video.get("matched_keywords", []):
                    if keyword not in keyword_to_videos:
                        keyword_to_videos[keyword] = []
                    keyword_to_videos[keyword].append(video)
        
        # チャンネル情報をソート（視聴回数順）
        top_channels = sorted(
            [(ch, info) for ch, info in channel_info.items()],
            key=lambda x: x[1]['total_views'],
            reverse=True
        )[:10]
        
        # タイトルから頻出フレーズを抽出
        title_phrases = self._extract_common_phrases(all_titles)
        
        # 分析結果をまとめる
        analysis = {
            "total_videos": len(videos),
            "total_views": total_views,
            "total_likes": total_likes,
            "avg_engagement": total_likes / max(total_views, 1) if total_views > 0 else 0,
            
            # 政党別分析
            "party_analysis": party_mentions,
            
            # チャンネル分析
            "top_channels_detailed": [
                {
                    'name': ch,
                    'follower_count': info['follower_count'],
                    'video_count': len(info['videos']),
                    'total_views': info['total_views'],
                    'top_video': max(info['videos'], key=lambda x: x['view_count']) if info['videos'] else None
                }
                for ch, info in top_channels
            ],
            
            # キーワード分析
            "keyword_frequency": {k: len(v) for k, v in keyword_to_videos.items()},
            
            # 重要キーワード分析
            "trending_topics": {
                k: {
                    'count': len(v),
                    'total_views': sum(vid['view_count'] for vid in v),
                    'top_video': max(v, key=lambda x: x['view_count']) if v else None
                }
                for k, v in keyword_appearances.items() if v
            },
            
            # タイトルフレーズ分析
            "common_title_phrases": title_phrases[:10],
            
            # 高視聴回数動画
            "top_viewed_videos": sorted(
                [v for v in videos if isinstance(v, dict)],
                key=lambda x: x.get('view_count', 0),
                reverse=True
            )[:10],
            
            # 参政党関連の詳細
            "sanseito_detailed": self._analyze_party_videos(videos, ['参政', '神谷'], 'sanseito'),
            
            # 自民党関連の詳細
            "jiminto_detailed": self._analyze_party_videos(videos, ['自民', '石破'], 'jiminto')
        }
        
        return analysis
    
    def _extract_common_phrases(self, titles: List[str]) -> List[Tuple[str, int]]:
        """タイトルから頻出フレーズを抽出"""
        import re
        from collections import Counter
        
        # 2-4語の連続したフレーズを抽出
        phrases = []
        for title in titles:
            # 記号で区切られた部分を抽出
            parts = re.split(r'[!！?？【】#\s]+', title)
            for part in parts:
                if len(part) >= 4:  # 4文字以上のフレーズ
                    phrases.append(part)
        
        # 頻度をカウント
        phrase_counter = Counter(phrases)
        return phrase_counter.most_common()


    def _analyze_party_videos(self, videos: List[Dict], keywords: List[str], party_name: str) -> Dict:
        """特定政党の動画を詳細分析"""
        party_videos = []
        
        for video in videos:
            if isinstance(video, dict):
                matched_keywords = video.get("matched_keywords", [])
                title = video.get("title", "")
                
                if any(k in matched_keywords for k in keywords) or any(k in title for k in keywords):
                    party_videos.append(video)
        
        if not party_videos:
            return {}
        
        # タイトルから内容傾向を分析
        positive_keywords = ['成功', '支持', '人気', '評価', '勝利']
        negative_keywords = ['悲報', '失敗', '批判', '問題', '落選', '解体']
        
        positive_count = sum(1 for v in party_videos if any(k in v.get('title', '') for k in positive_keywords))
        negative_count = sum(1 for v in party_videos if any(k in v.get('title', '') for k in negative_keywords))
        
        return {
            'count': len(party_videos),
            'total_views': sum(v.get('view_count', 0) for v in party_videos),
            'avg_views': sum(v.get('view_count', 0) for v in party_videos) / len(party_videos) if party_videos else 0,
            'sentiment_analysis': {
                'positive_videos': positive_count,
                'negative_videos': negative_count,
                'neutral_videos': len(party_videos) - positive_count - negative_count
            },
            'top_channels': Counter([v.get('channel', '') for v in party_videos]).most_common(5),
            'sample_titles': [v.get('title', '') for v in sorted(party_videos, key=lambda x: x.get('view_count', 0), reverse=True)[:5]]
        }

    def generate_single_response(self, 
                                post: RedditPost, 
                                treatment: TreatmentCondition,
                                model: str,
                                user_profile: Optional[UserProfile] = None,
                                pdf_references: List[str] = None,
                                json_data: Optional[Dict] = None) -> PersuasiveResponse:
        """単一のLLMモデルから応答を生成（トーナメントなし）"""
        logger.info(f"Generating single response from {model}")
        
        try:
            if treatment == TreatmentCondition.GENERIC:
                response = self._generate_generic_response(
                    post, model, pdf_references, json_data,
                    enable_web_search=True,
                    target_length=700
                )
            elif treatment == TreatmentCondition.PERSONALIZATION:
                response = self._generate_personalized_response(
                    post, user_profile, model, pdf_references
                )
            elif treatment == TreatmentCondition.COMMUNITY_ALIGNED:
                response = self._generate_community_aligned_response(
                    post, pdf_references
                )
            
            if response:
                response.user_profile = user_profile
                response.pdf_context = str(pdf_references) if pdf_references else None
            
            return response
            
        except Exception as e:  # インデントを修正
            logger.error(f"Failed to generate response: {str(e)}")
            fallback_response = PersuasiveResponse(
                content=self._generate_fallback_response(""),
                treatment_condition=treatment,
                persuasion_score=0.0,
                model_used=f"{model}_fallback",
                generation_params={'error': str(e)},
                user_profile=user_profile,
                pdf_context=str(pdf_references) if pdf_references else None
            )
            return fallback_response

    def generate_responses(self, 
                        post: RedditPost, 
                        treatment: TreatmentCondition,
                        user_profile: Optional[UserProfile] = None,
                        num_candidates: int = 32,
                        pdf_references: List[str] = None,
                        json_data: Optional[Dict] = None) -> List[PersuasiveResponse]:
        """各LLMから均等に候補を生成"""
        responses = []
        
        candidates_per_model = num_candidates // len(self.models)
        remaining_candidates = num_candidates % len(self.models)
        
        logger.info(f"Generating {num_candidates} candidates")
        
        for model_idx, model in enumerate(self.models):
            model_candidates = candidates_per_model
            
            if model_idx < remaining_candidates:
                model_candidates += 1
                
            logger.info(f"Generating {model_candidates} candidates from {model}")
            
            for i in range(model_candidates):
                try:
                    response = None
                    
                    if treatment == TreatmentCondition.GENERIC:
                        response = self._generate_generic_response(
                            post, model, pdf_references, json_data,
                            enable_web_search=True,
                            target_length=700
                        )
                    elif treatment == TreatmentCondition.PERSONALIZATION:
                        response = self._generate_personalized_response(
                            post, user_profile, model, pdf_references
                        )
                    elif treatment == TreatmentCondition.COMMUNITY_ALIGNED:
                        response = self._generate_community_aligned_response(
                            post, pdf_references
                        )
                    
                    if response:
                        response.user_profile = user_profile
                        response.pdf_context = str(pdf_references) if pdf_references else None
                        
                    responses.append(response)
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Failed to generate candidate: {str(e)}")
                    fallback_response = PersuasiveResponse(
                        content=self._generate_fallback_response(""),
                        treatment_condition=treatment,
                        persuasion_score=0.0,
                        model_used=f"{model}_fallback",
                        generation_params={'error': str(e)},
                        user_profile=user_profile,
                        pdf_context=str(pdf_references) if pdf_references else None
                    )
                    responses.append(fallback_response)
        
        logger.info(f"Total {len(responses)} candidates generated")
        return responses
    
    def summarize_json_analysis(self, analysis: Dict[str, Any], max_items: int = 10) -> Dict[str, Any]:
        """JSON分析結果を要約し、重要な情報のみを保持"""
        summarized = {
            "total_videos": analysis.get("total_videos", 0),
            "total_views": analysis.get("total_views", 0),
            "avg_engagement": analysis.get("avg_engagement", 0)
        }
        
        # 参政党分析の要約
        if analysis.get("sanseito_analysis"):
            sanseito = analysis["sanseito_analysis"]
            summarized["sanseito_analysis"] = {
                "count": sanseito.get("count", 0),
                "total_views": sanseito.get("total_views", 0),
                "avg_views_per_video": sanseito.get("total_views", 0) / max(sanseito.get("count", 1), 1),
                "top_videos": sanseito.get("videos", [])[:3]  # トップ3のみ
            }
        
        # 自民党分析の要約
        if analysis.get("jiminto_analysis"):
            jiminto = analysis["jiminto_analysis"]
            summarized["jiminto_analysis"] = {
                "count": jiminto.get("count", 0),
                "total_views": jiminto.get("total_views", 0)
            }
        
        # キーワード頻度の要約（上位5つのみ）
        if analysis.get("keyword_frequency"):
            top_keywords = dict(sorted(
                analysis["keyword_frequency"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
            summarized["top_keywords"] = top_keywords
        
        # 関連動画の要約（上位5つのみ）
        if analysis.get("relevant_videos"):
            summarized["top_relevant_videos"] = analysis["relevant_videos"][:5]
        
        # トップチャンネルの要約（上位5つのみ）
        if analysis.get("top_channels"):
            summarized["top_channels"] = analysis["top_channels"][:5]
        
        return summarized

    def _analyze_video_content_trends(self, analysis: Dict) -> str:
        """動画内容の傾向を文章で分析"""
        trends = "\n【動画内容の傾向分析】\n"
        
        # 参政党の詳細分析
        if analysis.get("sanseito_detailed") and analysis["sanseito_detailed"].get("count", 0) > 0:
            sanseito = analysis["sanseito_detailed"]
            trends += f"\n参政党関連動画の傾向:\n"
            trends += f"- 全{sanseito['count']}件中、"
            
            sentiment = sanseito.get('sentiment_analysis', {})
            if sentiment.get('negative_videos', 0) > sentiment.get('positive_videos', 0):
                trends += f"批判的な内容が{sentiment['negative_videos']}件と多数を占める\n"
            else:
                trends += f"好意的な内容が{sentiment['positive_videos']}件含まれる\n"
            
            if sanseito.get('sample_titles'):
                trends += "- 代表的な動画タイトル:\n"
                for title in sanseito['sample_titles'][:3]:
                    trends += f"  「{title}」\n"
        
        # 自民党の詳細分析
        if analysis.get("jiminto_detailed") and analysis["jiminto_detailed"].get("count", 0) > 0:
            jiminto = analysis["jiminto_detailed"]
            trends += f"\n自民党関連動画の傾向:\n"
            trends += f"- 全{jiminto['count']}件、総視聴{jiminto['total_views']:,}回\n"
            
            if jiminto.get('top_channels'):
                trends += "- 主な発信チャンネル: "
                trends += ", ".join([f"{ch[0]}（{ch[1]}本）" for ch in jiminto['top_channels'][:3]])
                trends += "\n"
        
        # 頻出フレーズ分析
        if analysis.get("common_title_phrases"):
            trends += "\n頻出するフレーズ:\n"
            for phrase, count in analysis["common_title_phrases"][:5]:
                if count > 2:  # 3回以上出現
                    trends += f"- 「{phrase}」（{count}回）\n"
        
        return trends

    def _generate_generic_response(self, post: RedditPost, model: str, 
                                pdf_paths: List[str] = None, 
                                json_data: Optional[Dict] = None,
                                enable_web_search: bool = True,
                                target_length: int = 700) -> PersuasiveResponse:
        """PDFとWeb検索、動画情報を参照して構造化された応答を生成"""
        
        # 現在の日時コンテキストを取得
        datetime_context = self.get_current_datetime_context()

        # PDFコンテキスト（既存のコード）
        pdf_context = ""
        if pdf_paths:
            for i, pdf_path in enumerate(pdf_paths[:3]):
                pdf_content = self.load_pdf_content(pdf_path, summarize=True)
                if pdf_content:
                    relevant_sections = self.get_relevant_pdf_sections(pdf_content, post.title + " " + post.body, max_sections=2)
                    if relevant_sections:
                        pdf_context += f"\n\n参考資料{i+1}（{Path(pdf_path).name}）:\n"
                        pdf_context += "\n".join(relevant_sections)
        
        # JSONコンテキストの生成（拡張版）
        json_context = ""
        video_content_analysis = ""
        structured_video_summary = ""  # 新規追加：構造化された動画サマリー
        
        if json_data:
            analysis = self.analyze_json_data(json_data, post.title + " " + post.body)
            if analysis:
                json_context = f"\n\n【YouTube/TikTok動画分析】\n"
                json_context += f"- 総動画数: {analysis['total_videos']:,}件\n"
                json_context += f"- 総視聴回数: {analysis['total_views']:,}回\n"
                
                # 構造化された動画サマリーを作成
                structured_video_summary = self._create_structured_video_summary(analysis)
                
                # 政党別の詳細分析
                if analysis.get("party_analysis"):
                    json_context += "\n【政党別動画分析】\n"
                    for party, data in analysis["party_analysis"].items():
                        if data['count'] > 0:
                            json_context += f"\n{party}関連:\n"
                            json_context += f"  - 動画数: {data['count']}件（総視聴回数: {data['total_views']:,}回）\n"
                            if data['videos']:
                                top_video = max(data['videos'], key=lambda x: x['view_count'])
                                json_context += f"  - 最も視聴された動画: 「{top_video['title'][:50]}...」\n"
                                json_context += f"    （{top_video['channel']}、{top_video['view_count']:,}回視聴）\n"
                
                # トレンドトピック分析
                if analysis.get("trending_topics"):
                    json_context += "\n【注目のトレンド】\n"
                    for topic, info in sorted(analysis["trending_topics"].items(), 
                                            key=lambda x: x[1]['total_views'], 
                                            reverse=True)[:5]:
                        if info['count'] > 0:
                            json_context += f"- 「{topic}」: {info['count']}件の動画（計{info['total_views']:,}回視聴）\n"
                            if info['top_video']:
                                json_context += f"  代表例: 「{info['top_video']['title'][:40]}...」\n"
                
                # 影響力のあるチャンネル
                if analysis.get("top_channels_detailed"):
                    json_context += "\n【影響力のあるチャンネル】\n"
                    for ch_info in analysis["top_channels_detailed"][:3]:
                        json_context += f"- {ch_info['name']}（フォロワー{ch_info['follower_count']:,}人）\n"
                        json_context += f"  投稿{ch_info['video_count']}本、総視聴{ch_info['total_views']:,}回\n"
                        if ch_info['top_video']:
                            json_context += f"  人気動画: 「{ch_info['top_video']['title'][:30]}...」\n"
                
                # 動画内容の傾向分析
                video_content_analysis = self._analyze_video_content_trends(analysis)
        
        # Web検索結果（既存のコード）
        web_context = ""
        if enable_web_search:
            search_queries = [
                "参政党 躍進 理由 分析",
                "SNS 政治 影響 日本"
            ]
            
            all_search_results = []
            for query in search_queries[:2]:
                results = self.search_web(query, max_results=3)
                all_search_results.extend(results)
            
            unique_results = {}
            for result in all_search_results:
                url = result['url']
                if url not in unique_results or result['score'] > unique_results[url]['score']:
                    unique_results[url] = result
            
            web_context = self.format_search_results(list(unique_results.values())[:5])
        
        # 改善されたプロンプト（構造化された応答を要求）
        prompt = f"""
    {datetime_context}

    あなたは日本の最大の新聞社（クオリティペーパー）のエース編集委員です。
    現在は{datetime.now().strftime('%Y年%m月%d日')}です。この日付を基準に分析してください。

    以下の投稿について、提供されたデータを分析し、【必ず以下の構成】で応答を作成してください。

    投稿内容:
    タイトル: {post.title}
    本文: {post.body}

    {pdf_context}

    {json_context}

    {video_content_analysis}

    {web_context}

    【重要：応答の必須構成】
    応答は必ず以下の3つの段落で構成してください：

    1. 【動画データの整理】（第1段落）
    - YouTube/TikTokの動画データから読み取れる事実を整理
    - 具体的な動画タイトル、チャンネル名、視聴回数を含める
    - YouTube/TikTokの動画データから読み取れるトレンドワードの出現頻度
    - どのような論調の動画が多いかの傾向分析

    2. 【統合的な考察】（第2段落）
    - 第1段落で整理した動画の傾向
    - PDFから読み取れる政党の公約や政策
    - Web検索から得られた専門家の分析
    これら3つの情報源を統合して、参政党躍進の背景を多角的に分析

    3. 【結論】（第3段落）
    - 統合的な分析から導かれる洞察
    - CMVの投稿者の見解に対する建設的な反論や別の視点の提示

    {structured_video_summary}

    各段落は2-3文以上で構成し、{target_length - 100}〜{target_length + 100}文字程度で、
    データに基づいた具体的で説得力のある応答をお願いします。
    """
        
        response_content = self._call_llm_api(prompt, model, max_tokens=2000)
        
        # 応答が適切な構造を持っているか確認し、必要に応じて修正
        response_content = self._ensure_structured_response(response_content, analysis if 'analysis' in locals() else None)
        
        # 生成パラメータに動画分析情報を追加
        generation_params = {
            'temperature': 0.7,
            'max_tokens': 2000,
            'pdf_references': pdf_paths if pdf_paths else [],
            'web_search_enabled': enable_web_search,
            'search_results_count': len(unique_results) if enable_web_search and 'unique_results' in locals() else 0,
            'total_context_length': len(pdf_context) + len(json_context) + len(web_context),
            'video_data_included': bool(json_data),
            'videos_analyzed': analysis.get('total_videos', 0) if 'analysis' in locals() else 0,
            'structured_response': True  # 構造化された応答であることを記録
        }
        
        response = PersuasiveResponse(
            content=response_content,
            treatment_condition=TreatmentCondition.GENERIC,
            persuasion_score=0.0,
            model_used=model,
            generation_params=generation_params
        )
        
        # 動画分析データを応答に付加
        if 'analysis' in locals():
            response.video_analysis = analysis
        
        return response
    
    def _create_structured_video_summary(self, analysis: Dict) -> str:
        """動画分析から構造化されたサマリーを作成"""
        summary = "\n【動画データサマリー（第1段落用）】\n"
        
        # 全体統計
        summary += f"分析対象: {analysis['total_videos']}件の動画（総視聴回数{analysis['total_views']:,}回）\n"
        
        # トレンドワード
        if analysis.get("trending_topics"):
            top_trends = []
            for topic, info in sorted(analysis["trending_topics"].items(), 
                                    key=lambda x: x[1]['count'], 
                                    reverse=True)[:3]:
                if info['count'] > 0:
                    top_trends.append(f"「{topic}」{info['count']}件")
            if top_trends:
                summary += f"主要トレンド: {', '.join(top_trends)}\n"
        
        # 政党別の傾向
        if analysis.get("party_analysis"):
            party_summaries = []
            for party, data in analysis["party_analysis"].items():
                if data['count'] > 0:
                    party_summaries.append(f"{party}（{data['count']}件/{data['total_views']:,}回視聴）")
            if party_summaries:
                summary += f"政党別動画数: {', '.join(party_summaries)}\n"
        
        # 代表的な動画例
        if analysis.get("top_viewed_videos"):
            summary += "\n必ず言及すべき具体例:\n"
            for i, video in enumerate(analysis["top_viewed_videos"][:3], 1):
                summary += f"{i}. 「{video.get('title', '')}」（{video.get('channel', '')}、{video.get('view_count', 0):,}回視聴）\n"
        
        return summary


    def _ensure_structured_response(self, content: str, analysis: Optional[Dict] = None) -> str:
        """応答が適切な構造を持っているか確認し、必要に応じて修正"""
        
        # 段落を識別するためのパターン
        paragraphs = content.split('\n\n')
        
        # 3段落未満の場合、構造を強制
        if len(paragraphs) < 3:
            # 動画データの整理段落が含まれているか確認
            has_video_paragraph = any('動画' in p and ('視聴' in p or 'チャンネル' in p) for p in paragraphs)
            has_integration_paragraph = any('統合' in p or '考察' in p or 'PDF' in p for p in paragraphs)
            
            if not has_video_paragraph or not has_integration_paragraph:
                # 構造化された応答に再構成
                return self._restructure_response(content, analysis)
        
        return content


    def _restructure_response(self, content: str, analysis: Optional[Dict] = None) -> str:
        """応答を3段落構造に再構成"""
        sentences = content.split('。')
        
        # 動画関連の文を抽出
        video_sentences = []
        integration_sentences = []
        conclusion_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # 動画データに関する文
            if any(keyword in sentence for keyword in ['動画', '視聴', 'チャンネル', '悲報', '解体', 'YouTube', 'TikTok']):
                video_sentences.append(sentence.strip())
            # 統合的な考察に関する文
            elif any(keyword in sentence for keyword in ['統合', 'PDF', '公約', 'Web', '分析から', '踏まえ']):
                integration_sentences.append(sentence.strip())
            # その他は結論
            else:
                conclusion_sentences.append(sentence.strip())
        
        # 再構成
        restructured = ""
        
        # 第1段落：動画データの整理
        if video_sentences:
            restructured += "動画データを見ると、" + '。'.join(video_sentences[:3]) + '。'
        else:
            # 動画データがない場合は生成
            if analysis and analysis.get('total_videos'):
                restructured += f"YouTube/TikTokの動画データを分析すると、全{analysis['total_videos']}件の動画で総視聴回数{analysis['total_views']:,}回という大きな注目を集めています。"
        
        restructured += "\n\n"
        
        # 第2段落：統合的な考察
        if integration_sentences:
            restructured += "これらの動画の傾向とPDF資料、Web検索結果を統合すると、" + '。'.join(integration_sentences[:3]) + '。'
        else:
            restructured += "動画で示された世論の傾向と、各政党の公約資料、専門家の分析を統合的に考察すると、参政党の躍進には複数の要因が絡み合っていることが分かります。"
        
        restructured += "\n\n"
        
        # 第3段落：結論
        if conclusion_sentences:
            restructured += '。'.join(conclusion_sentences[:3]) + '。'
        else:
            restructured += "以上の分析から、参政党の躍進は単純なSNS効果だけでなく、既存政党への不満と新しいメディア環境が生み出した複合的な現象と考えるべきでしょう。"
        
        return restructured

    def _generate_personalized_response(self, 
                                    post: RedditPost, 
                                    profile: UserProfile,
                                    model: str,
                                    pdf_paths: List[str] = None) -> PersuasiveResponse:
        """Generate response using post content and user profile"""
        
        # 現在の日時コンテキストを取得
        datetime_context = self.get_current_datetime_context()

        pdf_context = ""
        if pdf_paths:
            for pdf_path in pdf_paths:
                pdf_content = self.load_pdf_content(pdf_path)
                if pdf_content:
                    relevant_sections = self.get_relevant_pdf_sections(pdf_content, post.title + " " + post.body)
                    pdf_context += f"\n\n参考資料（{Path(pdf_path).name}）:\n"
                    pdf_context += "\n".join(relevant_sections)
        
        profile_context = f"""
ユーザープロフィール：
- 性別: {profile.gender.value}
- 年齢: {profile.age_range[0]}-{profile.age_range[1]}歳
- 場所: {profile.location or '不明'}
- 政治的志向: {profile.political_orientation.value}
- 興味: {', '.join(profile.interests[:3])}
- 文章スタイル: Fleschスコア {profile.writing_style.get('flesch_reading_ease', 60):.1f}
"""
        
        prompt = f"""
{datetime_context}

{profile_context}

ユーザーの背景に合わせて以下の意見に反論してください：
タイトル: {post.title}
内容: {post.body}

{pdf_context}

説得力のある反論を作成してください：
1. 現在の日時を正しく認識し、事実に基づいて応答する
2. 彼らの価値観や経験に共鳴する
3. 教育レベルに適した言語を使用する
4. 彼らの興味から関連する例を参照する
5. 政治的見解に基づく潜在的な懸念に対処する

返信は350〜450文字程度で日本語で書いてください。
"""
        
        response_content = self._call_llm_api(prompt, model)
        
        return PersuasiveResponse(
            content=response_content,
            treatment_condition=TreatmentCondition.PERSONALIZATION,
            persuasion_score=0.0,
            model_used=model,
            generation_params={
                'temperature': 0.7,
                'max_tokens': 500,
                'personalization_features': asdict(profile)
            }
        )

    def _generate_community_aligned_response(self, post: RedditPost, pdf_paths: List[str] = None) -> PersuasiveResponse:
        """Generate response using fine-tuned model on successful comments"""
        
        # 現在の日時コンテキストを取得
        datetime_context = self.get_current_datetime_context()

        pdf_context = ""
        if pdf_paths:
            for pdf_path in pdf_paths:
                pdf_content = self.load_pdf_content(pdf_path)
                if pdf_content:
                    relevant_sections = self.get_relevant_pdf_sections(pdf_content, post.title + " " + post.body)
                    pdf_context += f"\n\n参考資料（{Path(pdf_path).name}）:\n"
                    pdf_context += "\n".join(relevant_sections)
        
        prompt = f"""
{datetime_context}

高評価を受けたr/ChangeMyViewコメントのスタイルで返信を生成してください。

成功したコメントの主要なパターン：
- 検証から始める：「あなたの視点は理解できます...」
- 修辞的な質問を使う：「〜について考えたことはありますか？」
- 具体的な例を提供する
- 考えさせる質問で終わる

反論する投稿：
タイトル: {post.title}
内容: {post.body}

{pdf_context}

返信は350〜450文字程度で日本語で書いてください。
"""
        
        response_content = self._call_llm_api(prompt, 'gpt-4.1')
        
        return PersuasiveResponse(
            content=response_content,
            treatment_condition=TreatmentCondition.COMMUNITY_ALIGNED,
            persuasion_score=0.0,
            model_used=self.community_aligned_model,
            generation_params={'temperature': 0.6, 'max_tokens': 1000}
        )

    def _call_llm_api(self, prompt: str, model: str, max_tokens: int = 2000) -> str:
        """Call the appropriate LLM API with improved naturalness"""
        
        # 現在の日時情報を含む指示を追加
        datetime_context = self.get_current_datetime_context()

        # プロンプトに自然な文章生成の指示を追加（f-stringに修正）
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
        
        try:
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
                        
            elif model == 'claude-sonnet-4':
                if self.api_config.anthropic_client:
                    response = self.api_config.anthropic_client.messages.create(
                        model="claude-sonnet-4-20250514",  # 実際のモデル名に修正
                        messages=[
                            {"role": "user", "content": f"あなたは日本の最大の新聞社（クオリティペーパー）のエース編集委員です。現在は{datetime.now().strftime('%Y年%m月%d日')}です。\n\n{enhanced_prompt}"}
                        ],
                        temperature=0.8,
                        max_tokens=max_tokens
                    )
                    content = response.content[0].text
                    return self._post_process_response(content)
                        
            elif model == 'gemini-2.5-pro':
                if self.api_config.gemini_model:
                    # Geminiの場合はシステムメッセージを含めたプロンプトを作成
                    gemini_prompt = f"あなたは日本の最大の新聞社（クオリティペーパー）のエース編集委員です。現在は{datetime.now().strftime('%Y年%m月%d日')}です。\n\n{enhanced_prompt}"
                    response = self.api_config.gemini_model.generate_content(
                        gemini_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.8,
                            max_output_tokens=max_tokens,
                        )
                    )
                    content = response.text
                    return self._post_process_response(content)
                        
            elif model == 'llama-3.3-70b':  # model名を修正
                if self.api_config.openrouter_api_key:
                    headers = {
                        "Authorization": f"Bearer {self.api_config.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "Knowledge Distillation App"
                    }
                    
                    data = {
                        "model": "meta-llama/llama-3.3-70b-instruct",
                        "messages": [
                            {"role": "system", "content": f"あなたは日本の最大の新聞社（クオリティペーパー）のエース編集委員です。現在は{datetime.now().strftime('%Y年%m月%d日')}です。"},
                            {"role": "user", "content": enhanced_prompt}
                        ],
                        "temperature": 0.8,
                        "max_tokens": max_tokens
                    }
                    
                    response = requests.post(
                        f"{self.api_config.openrouter_base_url}/chat/completions",
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        content = response.json()['choices'][0]['message']['content']
                        return self._post_process_response(content)
                    else:
                        logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                            
            elif model == 'fallback':
                # フォールバックモード
                return self._generate_fallback_response(prompt)
                            
        except Exception as e:
            logger.error(f"Error calling {model}: {str(e)}")
        
        return self._generate_fallback_response(prompt)

    def _post_process_response(self, content: str) -> str:
        """応答を後処理して自然さを向上させる"""
        import re
        
        # 1. マークダウン記号を除去または変換
        content = re.sub(r'\*\*([^*]+)\*\*', r'「\1」', content)  # **text** → 「text」
        content = re.sub(r'###\s*', '', content)  # ### を除去
        content = re.sub(r'##\s*', '', content)   # ## を除去
        content = re.sub(r'#\s*', '', content)    # # を除去
        
        # 2. 重複する文章を検出して除去
        sentences = content.split('。')
        unique_sentences = []
        seen_patterns = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 文章の最初の20文字をパターンとして使用
            pattern = sentence[:20] if len(sentence) > 20 else sentence
            
            # 似たような文章がすでにある場合はスキップ
            if pattern not in seen_patterns:
                unique_sentences.append(sentence)
                seen_patterns.add(pattern)
        
        content = '。'.join(unique_sentences)
        
        # 3. 不完全な文章を修正
        # 文末が適切でない場合は修正
        if content and not content.endswith(('。', '！', '？', '」')):
            content += '。'
        
        # 4. 過度な改行を修正
        content = re.sub(r'\n{3,}', '\n\n', content)  # 3つ以上の改行を2つに
        content = re.sub(r'([。！？])\n([^「\n])', r'\1\n\n\2', content)  # 段落間に適切な改行
        
        # 5. 文章の流れを改善
        # 接続詞の前後を調整
        connectives = ['また', 'さらに', 'そして', 'しかし', 'ただし', 'なお', 'つまり']
        for conn in connectives:
            content = re.sub(f'。{conn}、', f'。\n\n{conn}、', content)
        
        # 6. 箇条書き記号を自然な文章に変換
        content = re.sub(r'^[-・]\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\d+\.\s*', '', content, flags=re.MULTILINE)
        
        return content.strip()

    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response if API calls fail"""
        try:
            if self.api_config.gemini_model:
                response = self.api_config.gemini_model.generate_content(
                    prompt + "\n\n注: 限られたコンテキストでも思慮深い応答を生成してください。"
                )
                return response.text
        except:
            pass
        
        # 日本語のフォールバック応答
        return (
            "ご指摘の視点は大変興味深く、よく考えられた内容だと思います。"
            "参政党の躍進について、確かにSNSやショート動画の影響は無視できない要因です。"
            "しかし、より深い分析のためには、有権者の意識変化や既存政党への不満、"
            "そして新しいメディア環境がどのように政治参加の形を変えているかを"
            "総合的に検討する必要があるでしょう。"
            "この問題についてさらに議論を深めていければ幸いです。"
        )
    
    def _ensure_complete_response(self, content: str, model: str, max_tokens: int) -> str:
        """応答が完結しているか確認し、必要に応じて修正"""
        # 文末が適切でない場合の判定
        incomplete_endings = [
            '、', '。、', 'が、', 'は、', 'で、', 'と、', 'の、', 'に、',
            'を、', 'も、', 'や、', 'から、', 'まで、', 'より、'
        ]
        
        # 文章が途中で切れている可能性がある場合
        if (not content.endswith(('。', '！', '？', '」')) or 
            any(content.rstrip().endswith(ending) for ending in incomplete_endings)):
            
            # 完結させるための追加プロンプト
            completion_prompt = f"""
    以下の文章は途中で切れています。自然に完結させてください（50文字以内）：

    {content}

    完結部分のみを出力してください。
    """
            
            try:
                completion = self._call_llm_api(completion_prompt, model, max_tokens=2000)
                # 適切に結合
                if completion and not completion.startswith(('。', '！', '？')):
                    content = content.rstrip() + completion
            except:
                # エラーの場合は、最後の不完全な文を削除
                last_period = content.rfind('。')
                if last_period > 0:
                    content = content[:last_period + 1]
        
        return content

# ====================
# Persuasion Experiment
# ====================
class PersuasionExperiment:
    """Complete experimental pipeline"""
    
    def __init__(self, api_config: APIConfig):
        self.profiler = UserProfiler()
        self.optimizer = PersuasionOptimizer(api_config)
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
                    use_single_model: bool = False,  # 新規追加
                    model_name: Optional[str] = None) -> Dict:  # 新規追加
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
        
        # 単一モデルモードの場合
        if use_single_model and model_name:
            logger.info(f"Using single model mode with {model_name}")
            
            single_response = self.optimizer.generate_single_response(
                target_post,
                treatment,
                model_name,
                user_profile,
                pdf_references,
                json_data
            )
            
            # スコアを計算
            single_response.persuasion_score = self.selector._score_response(single_response)
            
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
                    'type': 'single_model',
                    'model': model_name,
                    'score': single_response.persuasion_score
                }],
                'learning_enabled': False,
                'single_model_mode': True
            }
        
        # 既存のトーナメントモードの処理
        num_candidates = 32 if use_full_candidates else 4
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
            enable_learning=enable_learning
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
    title="知識蒸留 API Server - 完全統合版",
    description="paste_fixed.pyとpaste.pyのすべての機能を実装",
    version="1.0.0"
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
api_config = APIConfig()
experiment = PersuasionExperiment(api_config)
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
    max_tokens: Optional[int] = Field(default=None, description="最大トークン数（Noneの場合は自動計算）")
    use_single_model: Optional[bool] = Field(default=False, description="単一モデルのみ使用")
    model_name: Optional[str] = Field(default=None, description="使用するモデル名（gpt-4.1, claude-sonnet-4）")
    
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
    
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

class GenerateResponse(BaseModel):
    response: str
    model_used: str
    persuasion_score: float
    tournament_log: List[Dict[str, Any]]
    processing_time: float
    total_candidates: int
    learning_applied: bool = False
    json_analysis_included: bool = False
    web_search_included: bool = False
    topic: Optional[str] = None
    readability: Optional[float] = None
    user_profile: Optional[Dict] = None

# ====================
# Endpoints
# ====================
@app.get("/")
async def root():
    return {
        "message": "知識蒸留 API Server - 完全統合版",
        "version": "1.0.0",
        "features": [
            "LLMエージェント間競争（トーナメント形式）",
            "学習機能付きトーナメント",
            "PDF処理（日本語対応）",
            "Web検索（Tavily API）",
            "YouTube Shorts/TikTokデータ分析",
            "ユーザープロファイリング",
            "センチメント分析",
            "3つの処理条件（Generic/Personalization/Community-aligned）"
        ],
        "llm_models": [
            "OpenAI GPT-4",
            "Anthropic Claude",
            "Google Gemini",
            "Meta Llama (via OpenRouter)"
        ]
    }

@app.post("/api/analyze-json")
async def analyze_json(file: UploadFile = File(...)):
    """JSONファイルをアップロードして分析"""
    try:
        content = await file.read()
        json_data = json.loads(content)
        
        uploaded_files_data[file.filename] = json_data
        
        # 基本分析
        analysis = experiment.optimizer.analyze_json_data(json_data, "")
        
        return {
            "status": "success",
            "filename": file.filename,
            "analysis_summary": {
                "total_videos": analysis.get("total_videos", 0),
                "total_views": analysis.get("total_views", 0),
                "avg_engagement": analysis.get("avg_engagement", 0),
                "sanseito_count": analysis.get("sanseito_analysis", {}).get("count", 0),
                "jiminto_count": analysis.get("jiminto_analysis", {}).get("count", 0),
                "keyword_frequency": analysis.get("keyword_frequency", {})
            }
        }
    except Exception as e:
        logger.error(f"JSON analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """PDFファイルをアップロード"""
    try:
        pdf_path = f"/tmp/{file.filename}"
        content = await file.read()
        
        with open(pdf_path, "wb") as f:
            f.write(content)
        
        # PDFから文字抽出をテスト
        pdf_content = experiment.optimizer.load_pdf_content(pdf_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": pdf_path,
            "extracted_chars": len(pdf_content),
            "preview": pdf_content[:500] + "..." if pdf_content else "No text extracted"
        }
    except Exception as e:
        logger.error(f"PDF upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """メイン生成エンドポイント - 完全な知識蒸留処理"""
    try:
        start_time = datetime.now()
        
        logger.info(f"=== Generate Request Received ===")
        logger.info(f"Title: {request.title}")
        logger.info(f"Body length: {len(request.body)} chars")
        logger.info(f"Treatment: {request.treatment}")
        logger.info(f"Num candidates: {request.num_candidates}")
        logger.info(f"Enable learning: {request.enable_learning}")
        logger.info(f"Enable web search: {request.enable_web_search}")
        logger.info(f"JSON data provided: {request.json_data is not None}")
        logger.info(f"PDF references: {request.pdf_references}")
        
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
            model_name=request.model_name
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
            
            return GenerateResponse(
                response=winning_response.content,
                model_used=winning_response.model_used,
                persuasion_score=float(winning_response.persuasion_score),
                tournament_log=result.get('tournament_log', []),
                processing_time=float(processing_time),
                total_candidates=result.get('num_candidates', 0),
                learning_applied=result.get('learning_enabled', False),
                json_analysis_included=bool(json_data),
                web_search_included=request.enable_web_search,
                topic=result.get('topic'),
                readability=float(result.get('readability', 0.0)),
                user_profile=user_profile_dict
            )
        else:
            raise HTTPException(status_code=400, detail=result.get('reason', 'Unknown error'))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# プロジェクトのパスを設定
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")

# PDFディレクトリが存在しない場合は作成
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR, exist_ok=True)
    logger.warning(f"Created PDF directory: {PDF_DIR}")

@app.get("/api/available-pdfs")
async def list_available_pdfs(directory: str = Query(default=None)):
    """利用可能なPDFファイルをリスト"""
    try:
        # directoryが指定されていない場合は、デフォルトのPDFディレクトリを使用
        if directory is None:
            directory = PDF_DIR
        
        # ディレクトリの存在確認
        if not os.path.exists(directory):
            logger.error(f"PDF directory not found: {directory}")
            return {
                "error": f"Directory not found: {directory}",
                "pdfs": [],
                "directory": directory,
                "count": 0,
                "expected_path": PDF_DIR,
                "current_working_dir": os.getcwd()
            }
        
        # PDFファイルを検索
        pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        return {
            "pdfs": [os.path.basename(f) for f in pdf_files],
            "full_paths": pdf_files,
            "directory": directory,
            "count": len(pdf_files),
            "exists": True
        }
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        return {
            "error": str(e),
            "pdfs": [],
            "directory": directory if 'directory' in locals() else PDF_DIR,
            "count": 0
        }

@app.get("/api/status")
async def api_status():
    """APIのステータスと利用可能な機能を確認"""
    return {
        "status": "operational",
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
            "learning_capability": True
        }
    }

@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """422エラーの詳細ログ"""
    logger.error(f"=== 422 Validation Error ===")
    
    try:
        body = await request.body()
        logger.error(f"Request body: {body.decode('utf-8')[:1000]}")
        
        try:
            json_body = json.loads(body)
            logger.error(f"Parsed JSON keys: {list(json_body.keys())}")
        except:
            logger.error("Failed to parse body as JSON")
    except:
        logger.error("Failed to read request body")
    
    if hasattr(exc, 'errors'):
        for error in exc.errors():
            logger.error(f"Validation error: {error}")
    
    return {
        "detail": exc.errors() if hasattr(exc, 'errors') else str(exc),
        "message": "Validation failed. Check the request format."
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏆 知識蒸留 API Server - 完全統合版 🏆")
    print("="*60)
    print("\n主要機能:")
    print("✓ 複数LLMエージェント間の競争（トーナメント形式）")
    print("✓ 学習機能（勝者が敗者から良い点を抽出）")
    print("✓ PDF処理（日本語対応）")
    print("✓ Web検索（Tavily API）")
    print("✓ YouTube Shorts/TikTokデータ分析")
    print("✓ ユーザープロファイリング")
    print("✓ センチメント分析")
    print("✓ 3つの処理条件対応")
    print("\nサーバー起動中...")
    print("http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)