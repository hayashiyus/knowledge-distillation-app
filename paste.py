# Import required libraries
import os
import re
import time
import glob
import json
import random
import logging
import hashlib
import asyncio
import pdfplumber
import unicodedata
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from tavily import TavilyClient
import logging

# LLM API imports
import openai
import anthropic
import google.generativeai as genai
import requests  # For OpenRouter

# For text analysis
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK downloads might be required manually")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load environment variables
load_dotenv()

# Configure API clients
class APIConfig:
    def __init__(self):
        # OpenAI
        self.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Anthropic
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        # Google Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        
        # OpenRouter configuration
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        # Tavily Search API
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        if self.tavily_api_key:
            self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        else:
            self.tavily_client = None

# Initialize API configuration
api_config = APIConfig()

# Verify API keys are loaded
print("API Keys Status:")
print(f"OpenAI: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
print(f"Anthropic: {'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗'}")
print(f"Google: {'✓' if os.getenv('GOOGLE_API_KEY') else '✗'}")
print(f"OpenRouter: {'✓' if os.getenv('OPENROUTER_API_KEY') else '✗'}")
print(f"Tavily (Web Search): {'✓' if os.getenv('TAVILY_API_KEY') else '✗'}")

class LLMManager:
    """LLM API管理クラス"""
    def __init__(self, openai_api_key: str = None, anthropic_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')

# Define enums for structured data
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

@dataclass
class PoliticalResearcherProfile:
    """政治学研究者の専門プロフィール"""
    research_areas: List[str]
    academic_position: str
    publications: List[str]
    analytical_style: str
    
@dataclass
class PoliticalPost:
    """政治分析投稿用の拡張データ構造"""
    post_id: str
    title: str
    body: str
    author: str
    timestamp: datetime
    references: List[str]  # 参照文献
    data_sources: List[str]  # データソース
    tags: List[str]  # 政治的タグ

class UserProfiler:
    """Analyzes user's posting history to infer demographic attributes"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
        # 政治学研究者向けのインジケーター
        self.researcher_indicators = {
            'academic': ['研究', '論文', '学会', '博士', '教授', '准教授', '講師'],
            'analytical': ['分析', 'データ', '統計', '調査', '考察', '検証'],
            'political_science': ['政治学', '選挙', '政党', '議会', '投票行動', '世論']
        }
        
        # 2025年参院選関連のキーワード
        self.election_keywords = {
            'ruling_party': ['自民党', '公明党', '与党', '政権党'],
            'opposition': ['野党', '立憲民主党', '日本維新の会', '国民民主党'],
            'new_parties': ['参政党', '新党', '政治団体'],
            'issues': ['カルト', 'SNS', 'ショート動画', 'TikTok', 'YouTube Shorts']
        }
        
    def analyze_posting_history(self, posts: List[RedditPost]) -> UserProfile:
        """Analyze user's last 100 posts to build profile"""
        
        # Combine all post text
        all_text = ' '.join([p.title + ' ' + p.body for p in posts])
        
        # Infer gender
        gender = self._infer_gender(all_text)
        
        # Infer age range
        age_range = self._infer_age_range(all_text)
        
        # Infer location
        location = self._infer_location(all_text)
        
        # Infer political orientation
        political = self._infer_political_orientation(all_text)
        
        # Extract interests from subreddits
        interests = self._extract_interests(posts)
        
        # Analyze writing style
        writing_style = self._analyze_writing_style(posts)
        
        return UserProfile(
            username=posts[0].author if posts else "unknown",
            gender=gender,
            age_range=age_range,
            ethnicity=None,  # Simplified for this implementation
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
        # Look for age mentions
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
        
        # Default to broad range
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
        
        # Calculate various style metrics
        style = {
            'avg_sentence_length': np.mean([len(s.split()) for s in all_text.split('.') if s.strip()]),
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
            gender=Gender.UNKNOWN,  # 研究者として中立性を保つ
            age_range=(35, 45),  # 中堅研究者想定
            ethnicity=None,
            location="東京都",
            political_orientation=PoliticalOrientation.MODERATE,  # 学術的中立性
            interests=['政治学', '選挙分析', 'SNS政治', '世論調査', 'メディア研究'],
            writing_style={
                'avg_sentence_length': 25.0,  # 学術的な長文
                'flesch_reading_ease': 40.0,  # 専門的な文章
                'sentiment_compound': 0.0,  # 中立的トーン
                'question_ratio': 0.05,  # 分析的な問いかけ
                'exclamation_ratio': 0.0  # 感嘆符は使わない
            }
        )
    
class PDFTextCleaner:
    """PDFから抽出したテキストをクリーンアップするクラス"""
    
    def __init__(self):
        # 除去すべきパターン
        self.noise_patterns = [
            r'←この[^→\n]*への',  # 矢印付き注釈
            r'ファクト情報[なし|あり]',  # ファクト情報の注釈
            r'^\d+$',  # 行頭の単独数字
            r'✔',  # チェックマーク
            r'\[.*?\]「.*?」',  # [番号]「タイトル」形式
            r'についての具体的説明',  # 説明注釈
        ]
        
    def clean_text(self, text: str) -> str:
        """PDFテキストをクリーンアップ"""
        # 1. 全角スペースを半角に統一
        text = text.replace('　', ' ')
        
        # 2. 改行で分割された単語を結合
        text = self._fix_split_words(text)
        
        # 3. ノイズパターンを除去
        for pattern in self.noise_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # 4. 短すぎる行（3文字以下）を前後の行と結合
        text = self._merge_short_lines(text)
        
        # 5. 連続する改行を2つまでに制限
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 6. 各段落の先頭・末尾の空白を除去
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # 7. 空行の連続を1つに
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # 8. ヘッダー・フッターを除去
        text = self._remove_headers_footers(text)
        
        # 9. 句読点を正規化
        text = self._normalize_punctuation(text)
        
        return text.strip()
    
    def _fix_split_words(self, text: str) -> str:
        """改行で分割された日本語単語を結合"""
        # ひらがな・カタカナ・漢字で終わり、次の行が同じ文字種で始まる場合は結合
        pattern = r'([ぁ-んァ-ヶー一-龥々])\n([ぁ-んァ-ヶー一-龥々])'
        text = re.sub(pattern, r'\1\2', text)
        
        # 助詞で終わる行は次の行と結合
        pattern = r'([がのにをはでとも])\n'
        text = re.sub(pattern, r'\1', text)
        
        return text
    
    def _merge_short_lines(self, text: str) -> str:
        """短い行を前後の行と結合"""
        lines = text.split('\n')
        merged_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 空行はそのまま保持
            if not line:
                merged_lines.append('')
                i += 1
                continue
            
            # 短い行（3文字以下）は次の行と結合
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
        """ヘッダーやフッターのパターンを除去"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # ページ番号パターン（例: "- 1 -", "1/10", "第1ページ"）
            if re.match(r'^[-\s]*\d+[-\s]*$', line.strip()):
                continue
            if re.match(r'^\d+/\d+$', line.strip()):
                continue
            if re.match(r'^第\d+ページ$', line.strip()):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _normalize_punctuation(self, text: str) -> str:
        """句読点の正規化"""
        # 全角句読点に統一
        text = text.replace('､', '、')
        text = text.replace('｡', '。')
        text = text.replace('･', '・')
        
        return text

class PersuasionOptimizer:
    """Core optimization algorithm for generating persuasive responses"""
    
    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        self.models = ['gpt-4.1', 'claude-3.7-sonnet']
        self.community_aligned_model = 'gpt-4.1-finetuned'
        self.pdf_cache = {}  # PDFコンテンツのキャッシュ
        self.text_cleaner = PDFTextCleaner()  # 日本語テキストクリーナーを追加
        self.json_cache = {}  # JSONコンテンツのキャッシュ
        self.logger = logging.getLogger(self.__class__.__name__)  # ロガーを追加
        self.search_cache = {}  # Web検索結果のキャッシュ
        self.tournament_selector = TournamentSelector(persuasion_optimizer=self)

    def optimize_text(self, 
                     original_text: str,
                     user_profile: UserProfile,
                     selection_method: str = 'tournament',
                     pdf_context: str = None,
                     enable_learning: bool = True) -> Tuple[str, List[Dict]]:
        """
        テキストを最適化（学習機能付きオプション）
        """
        # 各エージェントで応答を生成
        responses = []
        
        # モデルのリスト
        models = ['gpt-4.1-2025-04-14', 'claude-3-7-sonnet-20250219']
        
        # 各モデルで応答を生成
        for model in models:
            try:
                response = self._generate_response(
                    original_text,
                    user_profile,
                    model,
                    pdf_context
                )
                responses.append(response)
            except Exception as e:
                print(f"Error with {model}: {str(e)}")
                continue
        
        # 選択方法に応じて最適な応答を選択
        if selection_method == 'tournament':
            if enable_learning:
                # 学習機能付きトーナメント
                winner, tournament_log = self.tournament_selector.run_tournament_with_learning_tracking(
                    responses,
                    enable_learning=True
                )
            else:
                # 通常のトーナメント
                winner, tournament_log = self.tournament_selector.run_tournament_with_tracking(responses)
            
            return winner.content, tournament_log
        else:
            # 他の選択方法の実装
            pass
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """Web検索を実行して関連情報を取得"""
        
        # キャッシュチェック
        cache_key = f"{query}_{max_results}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        if not self.api_config.tavily_client:
            self.logger.warning("Tavily API key not configured. Skipping web search.")
            return []
        
        try:
            # Tavilyで検索実行
            response = self.api_config.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=["asahi.com", "nikkei.com", "nhk.or.jp", "mainichi.jp", 
                               "yomiuri.co.jp", "sankei.com", "jiji.com", "reuters.com"],
                include_raw_content=True
            )
            
            # 結果をパース
            results = []
            for result in response.get('results', []):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'published_date': result.get('published_date', ''),
                    'score': result.get('score', 0)
                })
            
            # キャッシュに保存
            self.search_cache[cache_key] = results
            self.logger.info(f"Web search for '{query}' returned {len(results)} results")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Web search error: {str(e)}")
            return []
    
    def format_search_results(self, search_results: List[Dict]) -> str:
        """検索結果を文字列にフォーマット"""
        
        if not search_results:
            return ""
        
        formatted = "\n\n【Web検索結果】\n"
        for i, result in enumerate(search_results[:3], 1):  # 上位3件のみ使用
            formatted += f"\n{i}. {result['title']}\n"
            formatted += f"   出典: {result['url']}\n"
            if result.get('published_date'):
                formatted += f"   日付: {result['published_date']}\n"
            formatted += f"   内容: {result['content'][:200]}...\n"
        
        return formatted
    
    def _generate_generic_response(self, post: RedditPost, model: str, 
                                pdf_paths: List[str] = None, 
                                json_data: Optional[Dict] = None,  # 追加
                                enable_web_search: bool = True) -> PersuasiveResponse:
        """PDFとWeb検索を参照して応答を生成"""
        
        # PDF内容の読み込み（既存のコード）
        pdf_context = ""
        if pdf_paths:
            for pdf_path in pdf_paths:
                pdf_content = self.load_pdf_content(pdf_path)
                if pdf_content:
                    relevant_sections = self.get_relevant_pdf_sections(pdf_content, post.title + " " + post.body)
                    pdf_context += f"\n\n参考資料（{Path(pdf_path).name}）:\n"
                    pdf_context += "\n".join(relevant_sections)

        # JSON分析を追加
        json_context = ""
        if json_data:
            analysis = self.analyze_json_data(json_data, post.title + " " + post.body)
            if analysis:
                json_context = f"\n\n【YouTube/TikTok動画分析】\n"
                json_context += f"- 総動画数: {analysis['total_videos']:,}件\n"
                json_context += f"- 総視聴回数: {analysis['total_views']:,}回\n"
                json_context += f"- 平均エンゲージメント率: {analysis['avg_engagement']:.2%}\n"
                
                if analysis.get("relevant_videos"):
                    json_context += "\n関連動画トップ5:\n"
                    for i, video in enumerate(analysis["relevant_videos"][:5], 1):
                        json_context += f"{i}. {video['title']}\n"
                        json_context += f"   視聴回数: {video['views']:,}回, いいね: {video['likes']:,}回\n"
                
                if analysis.get("top_hashtags"):
                    json_context += "\n人気ハッシュタグ:\n"
                    for tag, count in analysis["top_hashtags"][:10]:
                        json_context += f"- {tag}: {count}回使用\n"

        # Web検索の実行
        web_context = ""
        if enable_web_search:
            # 検索クエリを生成（投稿のキーワードを抽出）
            search_queries = [
                "参政党 躍進 理由 分析",
                "SNS 政治 影響 日本",
                "自民党 公明党 支持率 低下",
                "TikTok YouTube Shorts 政治 影響"
            ]
            
            all_search_results = []
            for query in search_queries[:2]:  # コスト削減のため2クエリに制限
                results = self.search_web(query)
                all_search_results.extend(results)
            
            # 重複を除去して上位結果を選択
            unique_results = {}
            for result in all_search_results:
                url = result['url']
                if url not in unique_results or result['score'] > unique_results[url]['score']:
                    unique_results[url] = result
            
            web_context = self.format_search_results(list(unique_results.values())[:5])
        
        # プロンプトを更新
        prompt = f"""
        あなたはプロのメディアジャーナリストとして、以下の意見に対してデータドリブンかつナラティブの観点を統合した独自性の高い視点から応答してください。
        
        投稿内容:
        タイトル: {post.title}
        本文: {post.body}
        
        {pdf_context}
        
        {json_context}
        
        {web_context}
        
        以下の点に留意して応答してください：
        1. 最新のニュースやデータを引用する場合は出典を明示
        2. 動画データがある場合は、そのトレンドや傾向を分析に含める
        3. 複数の視点からの考察
        
        応答は800〜1000文字程度で、参考資料がある場合は適切に引用してください。
        """
        
        response_content = self._call_llm_api(prompt, model)
        
        return PersuasiveResponse(
            content=response_content,
            treatment_condition=TreatmentCondition.GENERIC,
            persuasion_score=0.0,
            model_used=model,
            generation_params={
                'temperature': 0.7,
                'max_tokens': 800,
                'pdf_references': pdf_paths if pdf_paths else [],
                'web_search_enabled': enable_web_search,
                'search_results_count': len(unique_results) if enable_web_search else 0
            }
        )
        
    def load_pdf_content(self, pdf_path: str) -> str:
        """日本語PDFファイルの内容を読み込んでテキストとして返す（警告抑制版）"""
        
        # キャッシュチェック
        if pdf_path in self.pdf_cache:
            return self.pdf_cache[pdf_path]
        
        text = ""
        
        # まずpdfplumberで試す（警告を抑制）
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could get FontBBox")
                warnings.filterwarnings("ignore", category=UserWarning)
                
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    
                    if text.strip():
                        self.logger.info(f"Successfully extracted text using pdfplumber from {pdf_path}")
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
        
        # pdfplumberで失敗した場合はPyPDF2を試す
        if not text.strip():
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                if text.strip():
                    self.logger.info(f"Successfully extracted text using PyPDF2 from {pdf_path}")
            except Exception as e:
                self.logger.error(f"PyPDF2 extraction also failed: {e}")
                return ""
        
        # テキストが抽出できた場合はクリーンアップ
        if text.strip():
            text = self.text_cleaner.clean_text(text)
            
            # キャッシュに保存（最大20000文字に制限）
            self.pdf_cache[pdf_path] = text[:20000]
            
            return self.pdf_cache[pdf_path]
        else:
            self.logger.error(f"No text could be extracted from {pdf_path}")
            return ""


    def load_json_content(self, json_path: str) -> Dict:
        """JSONファイルの内容を読み込む"""
        
        # キャッシュチェック
        if json_path in self.json_cache:
            return self.json_cache[json_path]
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.json_cache[json_path] = data
                self.logger.info(f"Successfully loaded JSON from {json_path}")
                return data
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {json_path}: {e}")
            return {}
    
    def extract_tiktok_summaries(self, json_data: Dict, query: str) -> List[Dict]:
        """TikTok動画要約から関連情報を抽出"""
        
        relevant_videos = []
        videos = json_data.get('videos', [])
        
        # クエリキーワードを抽出
        query_keywords = set(query.lower().split())
        
        for video in videos:
            score = 0
            title = video.get('title', '').lower()
            summary = video.get('summary', '').lower()
            
            # キーワードマッチング
            for keyword in query_keywords:
                if keyword in title:
                    score += 2
                if keyword in summary:
                    score += 1
            
            # エンゲージメントスコア
            views = video.get('views', 0)
            likes = video.get('likes', 0)
            engagement_score = (views / 10000) + (likes / 1000)
            score += engagement_score
            
            if score > 0:
                video['relevance_score'] = score
                relevant_videos.append(video)
        
        # スコアでソートして上位を返す
        relevant_videos.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_videos[:5]
    
    def analyze_json_data(self, json_data: Dict, query: str) -> Dict[str, Any]:
        """YouTube/TikTok動画データを分析"""
        
        if not json_data or "videos" not in json_data:
            return {}
        
        videos = json_data["videos"]
        
        # 基本統計
        analysis = {
            "total_videos": len(videos),
            "total_views": sum(v.get("views", 0) for v in videos),
            "total_likes": sum(v.get("likes", 0) for v in videos),
            "avg_engagement": sum(v.get("likes", 0) / max(v.get("views", 1), 1) for v in videos) / max(len(videos), 1)
        }
        
        # キーワード関連動画を抽出
        query_keywords = set(query.lower().split())
        relevant_videos = []
        
        for video in videos:
            relevance_score = 0
            title = video.get("title", "").lower()
            summary = video.get("summary", "").lower()
            
            for keyword in query_keywords:
                if keyword in title:
                    relevance_score += 2
                if keyword in summary:
                    relevance_score += 1
            
            if relevance_score > 0:
                video["relevance_score"] = relevance_score
                relevant_videos.append(video)
        
        # 関連度でソート
        relevant_videos.sort(key=lambda x: x["relevance_score"], reverse=True)
        analysis["relevant_videos"] = relevant_videos[:10]
        
        # ハッシュタグ分析
        all_hashtags = [tag for v in videos for tag in v.get("hashtags", [])]
        hashtag_counts = Counter(all_hashtags)
        analysis["top_hashtags"] = hashtag_counts.most_common(20)
        
        # 時系列分析
        videos_by_date = {}
        for video in videos:
            date = video.get("posted_date", "unknown")
            if date not in videos_by_date:
                videos_by_date[date] = []
            videos_by_date[date].append(video)
        
        analysis["timeline"] = videos_by_date
        
        return analysis

    def get_relevant_pdf_sections(self, pdf_content: str, query: str, max_sections: int = 3) -> List[str]:
        """PDFコンテンツから関連するセクションを抽出（日本語対応改善版）"""
        
        if not pdf_content or not query:
            return []
        
        # 日本語の文章分割（。、！、？で分割）
        sentence_endings = r'[。！？]'
        sentences = re.split(sentence_endings, pdf_content)
        
        # 段落を形成（3文ごとにグループ化）
        sections = []
        for i in range(0, len(sentences), 3):
            section = '。'.join(sentences[i:i+3])
            if section.strip() and len(section) > 30:  # 30文字以上のセクションのみ
                sections.append(section + '。')
        
        # クエリから重要なキーワードを抽出（形態素解析の簡易版）
        # 3文字以上の単語を重要キーワードとして扱う
        query_keywords = []
        
        # 漢字・カタカナの連続を抽出
        kanji_kata_pattern = r'[一-龥ァ-ヶー]{3,}'
        query_keywords.extend(re.findall(kanji_kata_pattern, query))
        
        # ひらがなの重要語（4文字以上）
        hiragana_pattern = r'[ぁ-ん]{4,}'
        query_keywords.extend(re.findall(hiragana_pattern, query))
        
        # 政治関連の重要キーワード
        political_keywords = [
            '選挙', '政党', '議席', '投票', '政権', '与党', '野党',
            '自民党', '公明党', '参政党', 'SNS', 'カルト', '宗教'
        ]
        
        # クエリに含まれる政治キーワードを追加
        for keyword in political_keywords:
            if keyword in query:
                query_keywords.append(keyword)
        
        # 重複を除去
        query_keywords = list(set(query_keywords))
        
        # セクションのスコアリング
        scored_sections = []
        for section in sections:
            score = 0
            
            # キーワードマッチングスコア
            for keyword in query_keywords:
                # 完全一致
                score += section.count(keyword) * 2
                
                # 部分一致（キーワードが3文字以上の場合）
                if len(keyword) >= 3:
                    for i in range(len(keyword) - 2):
                        partial = keyword[i:i+3]
                        score += section.count(partial) * 0.5
            
            # セクションの長さによるボーナス（情報量）
            length_bonus = min(len(section) / 100, 2.0)
            score += length_bonus
            
            # 数字を含むセクションにボーナス（データの可能性）
            if re.search(r'\d+[％%]|\d+議席|\d+票', section):
                score += 1.0
            
            if score > 0:
                scored_sections.append((score, section))
        
        # スコアの高い順にソート
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        
        # 上位セクションを返す
        result_sections = []
        for i, (score, section) in enumerate(scored_sections[:max_sections]):
            # セクション番号を付けて返す
            result_sections.append(f"【関連箇所{i+1}】\n{section}")
        
        return result_sections

    def generate_responses(self, 
                        post: RedditPost, 
                        treatment: TreatmentCondition,
                        user_profile: Optional[UserProfile] = None,
                        num_candidates: int = 124,
                        pdf_references: List[str] = None,
                        json_data: Optional[Dict] = None) -> List[PersuasiveResponse]:
        """各LLMから均等に候補を生成してトーナメントに参加させる"""
        
        responses = []
        
        # 各モデルから生成する候補数を計算
        candidates_per_model = num_candidates // len(self.models)
        remaining_candidates = num_candidates % len(self.models)
        
        print(f"\nGenerating candidates:")
        print(f"- {candidates_per_model} candidates from each model")
        if remaining_candidates > 0:
            print(f"- {remaining_candidates} additional candidates from random models")
        
        # 各モデルから確実に候補を生成
        for model_idx, model in enumerate(self.models):
            model_candidates = candidates_per_model
            
            # 余りの候補を最初のモデルに割り当て
            if model_idx < remaining_candidates:
                model_candidates += 1
                
            print(f"\nGenerating {model_candidates} candidates from {model}...")
            
            for i in range(model_candidates):
                try:
                    response = None
                    
                    if treatment == TreatmentCondition.GENERIC:
                        response = self._generate_generic_response(
                            post, model, pdf_references, json_data
                        )
                    elif treatment == TreatmentCondition.PERSONALIZATION:
                        response = self._generate_personalized_response(
                            post, user_profile, model, pdf_references
                        )
                    elif treatment == TreatmentCondition.COMMUNITY_ALIGNED:
                        response = self._generate_community_aligned_response(
                            post, pdf_references
                        )
                    
                    # user_profileとpdf_contextを設定
                    if response:
                        response.user_profile = user_profile
                        response.pdf_context = str(pdf_references) if pdf_references else None
                        
                    responses.append(response)
                    print(f"  ✓ Candidate {i+1}/{model_candidates} generated")
                    
                    # API rate limit対策
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"  ✗ Failed to generate candidate {i+1}: {str(e)}")
                    # フォールバック応答を生成
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
        
        print(f"\n✓ Total {len(responses)} candidates generated")
        return responses
    
    def _generate_personalized_response(self, 
                                    post: RedditPost, 
                                    profile: UserProfile,
                                    model: str,
                                    pdf_paths: List[str] = None) -> PersuasiveResponse:  # pdf_pathsを追加
        """Generate response using post content and user profile"""
        
        # PDF内容の読み込み（_generate_generic_responseと同様の処理）
        pdf_context = ""
        if pdf_paths:
            for pdf_path in pdf_paths:
                pdf_content = self.load_pdf_content(pdf_path)
                if pdf_content:
                    relevant_sections = self.get_relevant_pdf_sections(pdf_content, post.title + " " + post.body)
                    pdf_context += f"\n\n参考資料（{Path(pdf_path).name}）:\n"
                    pdf_context += "\n".join(relevant_sections)
        
        # Build personalization context
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
        {profile_context}
        
        ユーザーの背景に合わせて以下の意見に反論してください：
        タイトル: {post.title}
        内容: {post.body}
        
        説得力のある反論を作成してください：
        1. 彼らの価値観や経験に共鳴する
        2. 教育レベルに適した言語を使用する
        3. 彼らの興味から関連する例を参照する
        4. 政治的見解に基づく潜在的な懸念に対処する
        
        返信は150〜250文字程度で日本語で書いてください。
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
        
        # PDF内容の読み込み（同様の処理）
        pdf_context = ""
        if pdf_paths:
            for pdf_path in pdf_paths:
                pdf_content = self.load_pdf_content(pdf_path)
                if pdf_content:
                    relevant_sections = self.get_relevant_pdf_sections(pdf_content, post.title + " " + post.body)
                    pdf_context += f"\n\n参考資料（{Path(pdf_path).name}）:\n"
                    pdf_context += "\n".join(relevant_sections)
        
        prompt = f"""
        高評価を受けたr/ChangeMyViewコメントのスタイルで返信を生成してください。
        
        成功したコメントの主要なパターン：
        - 検証から始める：「あなたの視点は理解できます...」
        - 修辞的な質問を使う：「〜について考えたことはありますか？」
        - 具体的な例を提供する
        - 考えさせる質問で終わる
        
        反論する投稿：
        タイトル: {post.title}
        内容: {post.body}
        
        返信は250〜350文字程度で日本語で書いてください。
        """
        
        # Use GPT-4.1 for community-aligned responses
        response_content = self._call_llm_api(prompt, 'gpt-4.1')
        
        return PersuasiveResponse(
            content=response_content,
            treatment_condition=TreatmentCondition.COMMUNITY_ALIGNED,
            persuasion_score=0.0,
            model_used=self.community_aligned_model,
            generation_params={'temperature': 0.6, 'max_tokens': 500}
        )
    
    def _call_llm_api(self, prompt: str, model: str) -> str:
        """Call the appropriate LLM API based on model name"""
        
        try:
            if model == 'gpt-4.1' or model == 'gpt-4.1-finetuned':
                # OpenAI API
                response = self.api_config.openai_client.chat.completions.create(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system", "content": "あなたは日本語で応答する説得力のあるアシスタントです。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
                
            elif model == 'claude-3.7-sonnet':
                # Anthropic API
                response = self.api_config.anthropic_client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    messages=[
                        {"role": "user", "content": f"日本語で応答してください。\n\n{prompt}"}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.content[0].text
                
            elif model == 'llama-3.3-8b':
                # OpenRouter API for Llama
                headers = {
                    "Authorization": f"Bearer {self.api_config.openrouter_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "meta-llama/llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                
                response = requests.post(
                    f"{self.api_config.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    raise Exception(f"OpenRouter API error: {response.status_code}")
                    
            else:
                # Fallback to Gemini
                response = self.api_config.gemini_model.generate_content(prompt)
                return response.text
                
        except Exception as e:
            print(f"Error calling {model}: {str(e)}")
            # Fallback response
            return self._generate_fallback_response(prompt)
            
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response if API calls fail"""
        
        # Try Gemini as fallback
        try:
            response = self.api_config.gemini_model.generate_content(
                prompt + "\n\nNote: Generate a thoughtful response even if you have limited context."
            )
            return response.text
        except:
            # Final fallback
            return (
                "あなたの視点を理解し、思慮深く提示していただいたことに感謝します。"
                "あなたの主張には価値がありますが、あなたの前提のいくつかに挑戦する可能性のある"
                "代替的な視点を検討されましたか？時には異なる角度から問題を検討することで、"
                "見落としていたかもしれないニュアンスが明らかになることがあります。"
                "このトピックのどの側面が最も説得力があると感じ、あなたに考えさせた反論はありますか？"
            )

class TournamentSelector:
    """Implements single-elimination tournament for response selection"""

    def __init__(self, persuasion_optimizer=None):  # persuasion_optimizerを追加
        self.sia = SentimentIntensityAnalyzer()
        self.persuasion_optimizer = persuasion_optimizer  # オプティマイザーへの参照を保存
    
    def run_tournament(self, responses: List[PersuasiveResponse]) -> PersuasiveResponse:
        """Run single-elimination tournament to select best response"""
        
        # Score all responses
        for response in responses:
            response.persuasion_score = self._score_response(response)
        
        # Shuffle for random bracket assignment
        contestants = responses.copy()
        random.shuffle(contestants)
        
        # Run tournament rounds
        while len(contestants) > 1:
            next_round = []
            
            for i in range(0, len(contestants), 2):
                if i + 1 < len(contestants):
                    winner = self._compare_responses(contestants[i], contestants[i + 1])
                    next_round.append(winner)
                else:
                    # Bye for odd contestant
                    next_round.append(contestants[i])
            
            contestants = next_round
        
        return contestants[0]

    def _score_response(self, response: PersuasiveResponse) -> float:
        """政治分析投稿に適したスコアリング"""
        
        score = 0.0
        content = response.content
        
        # 学術的な文章の長さ（400-600文字が最適）
        char_count = len(content)
        if 400 <= char_count <= 600:
            score += 1.0
        else:
            score += max(0, 1.0 - abs(char_count - 500) / 500)
        
        # 専門用語の使用
        academic_terms = ['分析', '考察', '要因', '背景', '傾向', '影響', '構造', '動向']
        academic_score = sum(1 for term in academic_terms if term in content)
        score += min(1.0, academic_score / 4)
        
        # データ・統計への言及
        data_keywords = ['データ', '統計', '調査', '%', '割合', '票', '議席']
        data_score = sum(1 for keyword in data_keywords if keyword in content)
        score += min(1.0, data_score / 3)
        
        # 参照・引用の存在
        reference_patterns = ['によると', '参考資料', '出典', 'より引用', '参照']
        ref_score = sum(1 for pattern in reference_patterns if pattern in content)
        score += min(1.0, ref_score / 2)
        
        # 政治的中立性（極端な表現を避ける）
        extreme_words = ['絶対', '完全に', '間違いなく', '明らかに誤り']
        neutrality_penalty = sum(0.2 for word in extreme_words if word in content)
        score -= neutrality_penalty
        
        # 構造的な議論（接続詞の使用）
        connectives = ['しかし', 'また', '一方で', 'さらに', 'つまり', 'したがって']
        structure_score = sum(1 for conn in connectives if conn in content)
        score += min(1.0, structure_score / 3)
        
        # PDF参照ボーナス
        if response.generation_params.get('pdf_references'):
            score += 0.5
        
        return max(0, score)  # 負のスコアを防ぐ
    
    def _compare_responses(self, resp1: PersuasiveResponse, resp2: PersuasiveResponse) -> PersuasiveResponse:
        """Compare two responses and return winner"""
        
        # Primary criterion: persuasion score
        if resp1.persuasion_score > resp2.persuasion_score:
            return resp1
        elif resp2.persuasion_score > resp1.persuasion_score:
            return resp2
        
        # Tiebreaker: prefer personalized responses
        if resp1.treatment_condition == TreatmentCondition.PERSONALIZATION:
            return resp1
        elif resp2.treatment_condition == TreatmentCondition.PERSONALIZATION:
            return resp2
        
        # Random tiebreaker
        return random.choice([resp1, resp2])

    def extract_good_points_sync(self, winner: PersuasiveResponse, loser: PersuasiveResponse) -> str:
        """
        敗者の応答から良い点を抽出し、勝者の応答に統合する（同期版）
        """
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
    - 学術的な文体を保つ
    - 400-600文字程度にまとめる
    - 政治的中立性を保つ

    改善された分析のみを出力してください。
    """

        try:
            # api_configを直接使用
            if self.persuasion_optimizer and hasattr(self.persuasion_optimizer, 'api_config'):
                api_config = self.persuasion_optimizer.api_config
                
                # 勝者と同じモデルを使用して改善
                if 'gpt' in winner.model_used.lower():
                    # OpenAI APIを使用
                    if api_config.openai_client:
                        model_name = "gpt-4-turbo-preview"
                        
                        response = api_config.openai_client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": extraction_prompt}],
                            temperature=0.7,
                            max_tokens=1000
                        )
                        return response.choices[0].message.content
                
                elif 'claude' in winner.model_used.lower():
                    # Anthropic APIを使用
                    if api_config.anthropic_client:
                        model_name = "claude-3-sonnet-20240229"
                        
                        response = api_config.anthropic_client.messages.create(
                            model=model_name,
                            messages=[{"role": "user", "content": extraction_prompt}],
                            max_tokens=1000,
                            temperature=0.7
                        )
                        return response.content[0].text
            
            # APIが利用できない場合は元の内容を返す
            return winner.content
            
        except Exception as e:
            print(f"Error in extracting good points: {str(e)}")
            return winner.content

    def _compare_responses_with_learning(self, resp1: PersuasiveResponse, resp2: PersuasiveResponse) -> PersuasiveResponse:
        """
        2つの応答を比較し、勝者が敗者から学習する
        """
        # 通常の比較で勝者を決定
        winner = self._compare_responses(resp1, resp2)
        loser = resp2 if winner == resp1 else resp1
        
        # 勝者が敗者から学習
        improved_content = self.extract_good_points_sync(winner, loser)
        
        # 改善された内容で新しいレスポンスを作成
        improved_winner = PersuasiveResponse(
            user_profile=winner.user_profile,
            treatment_condition=winner.treatment_condition,
            pdf_context=winner.pdf_context,
            content=improved_content,
            model_used=winner.model_used,
            generation_params=winner.generation_params,
            persuasion_score=winner.persuasion_score
        )
        
        # 学習履歴を記録
        improved_winner.learning_history = getattr(winner, 'learning_history', [])
        improved_winner.learning_history.append({
            'learned_from': loser.model_used,
            'original_score': winner.persuasion_score,
            'opponent_score': loser.persuasion_score
        })
        
        # 改善後のスコアを再計算
        improved_winner.persuasion_score = self._score_response(improved_winner)
        
        return improved_winner

    # 3. 既存の run_tournament_with_tracking メソッドを修正
    def run_tournament_with_learning_tracking(self, responses: List[PersuasiveResponse], enable_learning: bool = True) -> Tuple[PersuasiveResponse, List[Dict]]:
        """
        学習機能付きトーナメントを実行し、各ラウンドの結果を記録
        """
        tournament_log = []
        
        # 全応答をスコアリング
        for response in responses:
            response.persuasion_score = self._score_response(response)
        
        # 初期状態を記録
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
                    
                    # 学習機能の有無で処理を分岐
                    if enable_learning:
                        winner = self._compare_responses_with_learning(contestant1, contestant2)
                    else:
                        winner = self._compare_responses(contestant1, contestant2)
                    
                    round_results.append({
                        'match': f"{contestant1.model_used} vs {contestant2.model_used}",
                        'scores': f"{contestant1.persuasion_score:.2f} vs {contestant2.persuasion_score:.2f}",
                        'winner': winner.model_used,
                        'winner_improved_score': f"{winner.persuasion_score:.2f}",
                        'learning_applied': enable_learning and hasattr(winner, 'learning_history')
                    })
                    
                    next_round.append(winner)
                else:
                    # Bye
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
        
        # 最終勝者の学習履歴をログに追加
        final_winner = contestants[0]
        if hasattr(final_winner, 'learning_history') and final_winner.learning_history:
            tournament_log.append({
                'round': 'final',
                'type': 'learning_summary',
                'learning_history': final_winner.learning_history,
                'total_improvements': len(final_winner.learning_history)
            })
        
        return final_winner, tournament_log
    
    async def run_tournament_with_learning(self, responses: List[PersuasiveResponse]) -> Tuple[PersuasiveResponse, List[Dict]]:
        """
        学習機能付きトーナメントを実行し、各ラウンドの結果を記録
        """
        tournament_log = []
        
        # 全応答をスコアリング
        for response in responses:
            response.persuasion_score = self._score_response(response)
        
        # 初期状態を記録
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
                    
                    # 学習機能付き比較を実行
                    winner = await self._compare_responses_with_learning(contestant1, contestant2)
                    
                    round_results.append({
                        'match': f"{contestant1.model_used} vs {contestant2.model_used}",
                        'scores': f"{contestant1.persuasion_score:.2f} vs {contestant2.persuasion_score:.2f}",
                        'winner': winner.model_used,
                        'winner_improved_score': f"{winner.persuasion_score:.2f}",
                        'learning_applied': hasattr(winner, 'learning_history') and len(winner.learning_history) > 0
                    })
                    
                    next_round.append(winner)
                else:
                    # Bye
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
        
        # 最終勝者の学習履歴をログに追加
        final_winner = contestants[0]
        if hasattr(final_winner, 'learning_history'):
            tournament_log.append({
                'round': 'final',
                'type': 'learning_summary',
                'learning_history': final_winner.learning_history
            })
        
        return final_winner, tournament_log

class PersuasionExperiment:
    """Complete experimental pipeline from the research paper"""
    
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
                    enable_learning: bool = True) -> Dict:  # enable_learningパラメータを追加
        """Run complete persuasion optimization pipeline with learning"""
        
        # Step 1: Filter check
        if not self._is_answerable(target_post):
            return {'status': 'filtered', 'reason': 'Post requires future knowledge'}
        
        # Step 2: Calculate stratification factors
        topic = self._classify_topic(target_post)
        readability = textstat.flesch_reading_ease(target_post.body)
        
        # Step 3: Profile user (for personalization)
        user_profile = None
        if treatment == TreatmentCondition.PERSONALIZATION:
            user_profile = self.profiler.analyze_posting_history(user_history)
        else:
            # デフォルトプロファイルを使用
            user_profile = self.profiler.create_political_researcher_profile()
        
        # Step 4: Generate candidate responses
        num_candidates = 16 if use_full_candidates else 4
        print(f"\nGenerating {num_candidates} candidate responses...")
        
        candidates = self.optimizer.generate_responses(
            target_post, 
            treatment, 
            user_profile,
            num_candidates=num_candidates,
            pdf_references=pdf_references
        )
        
        # Add delay to respect API rate limits
        time.sleep(1)

        # Step 5: Run tournament selection with learning
        print(f"\nRunning tournament with learning enabled: {enable_learning}")
        
        winning_response, tournament_log = self.selector.run_tournament_with_learning_tracking(
            candidates,
            enable_learning=enable_learning
        )

        # Step 6: Calculate posting delay
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
            'learning_enabled': enable_learning
        }
    
    def _is_answerable(self, post: RedditPost) -> bool:
        """Check if post can be answered with current knowledge"""
        
        # 分析や議論は常に可能
        analysis_keywords = ['分析', '考察', '議論', 'analysis', 'discussion', 'CMV']
        if any(keyword in post.title or keyword in post.body for keyword in analysis_keywords):
            return True
        
        # 遠い未来の予測のみをフィルタリング
        far_future_indicators = ['2030', '2040', '2050', 'distant future', 'long-term prediction']
        text_lower = (post.title + ' ' + post.body).lower()
        
        return not any(indicator in text_lower for indicator in far_future_indicators)
    
    def _classify_topic(self, post: RedditPost) -> str:
        """Classify post topic (日本語対応版)"""
        
        # Simple keyword-based classification
        text_lower = (post.title + ' ' + post.body).lower()
        
        # 日本語キーワードも追加
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
        """Calculate random posting delay (10-180 minutes, normal distribution)"""
        
        # Normal distribution centered at 15 minutes, truncated between 10 and 180
        delay = np.random.normal(15, 30)
        return int(np.clip(delay, 10, 180))

class RedditSimulator:
    """Simulates a Reddit user with posting history"""
    
    def __init__(self):
        self.subreddits = [
            'changemyview', 'politics', 'technology', 'science', 
            'philosophy', 'economics', 'education', 'worldnews',
            'books', 'movies', 'gaming', 'fitness'
        ]
        
        self.post_templates = [
            "CMV: {controversial_statement}",
            "I believe {opinion} and here's why",
            "My view on {topic}: {stance}",
            "Why {question}? My thoughts inside",
            "Discussion: {topic} and its implications"
        ]
        
    def generate_user_history(self, username: str, num_posts: int = 100) -> List[RedditPost]:
        """Generate realistic posting history for a user"""
        
        posts = []
        base_date = datetime.now() - timedelta(days=365)
        
        # Define user personality traits
        political_lean = random.choice(['liberal', 'conservative', 'moderate'])
        age = random.randint(18, 65)
        interests = random.sample(self.subreddits, k=5)
        
        for i in range(num_posts):
            # Generate post metadata
            days_ago = random.randint(0, 365)
            timestamp = base_date + timedelta(days=days_ago)
            subreddit = random.choice(interests + self.subreddits)  # Bias towards interests
            
            # Generate content based on personality
            if subreddit == 'politics':
                title, body = self._generate_political_post(political_lean)
            elif subreddit == 'technology':
                title, body = self._generate_tech_post(age)
            else:
                title, body = self._generate_general_post(subreddit)
            
            post = RedditPost(
                post_id=f"post_{i}_{hashlib.md5(title.encode()).hexdigest()[:8]}",
                subreddit=subreddit,
                title=title,
                body=body,
                author=username,
                timestamp=timestamp,
                score=random.randint(1, 500)
            )
            posts.append(post)
        
        return sorted(posts, key=lambda x: x.timestamp, reverse=True)
    
    def _generate_political_post(self, lean: str) -> Tuple[str, str]:
        if lean == 'liberal':
            topics = [
                ("国民皆保険制度は基本的人権であるべきだ", 
                 "日本の医療制度について考えています。友人が医療費で苦しんでいるのを見て...")
            ]
        elif lean == 'conservative':
            topics = [
                ("自由市場の解決策は政府のプログラムよりも効果的だ",
                 "歴史を振り返ると、民間企業は一貫して政府の取り組みを上回っています...")
            ]
        else:
            topics = [
                ("経済政策については両党とも正しい点がある",
                 "税制と支出について両方の視点を理解しようとしています...")
            ]
        
        return random.choice(topics)
    
    def _generate_tech_post(self, age: int) -> Tuple[str, str]:
        if age < 30:
            return ("AIは私たちの働き方を革命的に変えるだろう", 
                   "テクノロジーと共に育った世代として、AIの可能性にワクワクしています...")
        else:
            return ("AI導入には慎重になる必要がある",
                   "長年テック業界で働いてきて、AI実装を急ぐことへの懸念があります...")
    
    def _generate_general_post(self, subreddit: str) -> Tuple[str, str]:
        topics = {
            'education': ("教師の給与はもっと高くあるべきだ", "教育制度は改革が必要です..."),
            'science': ("気候変動には即座の行動が必要だ", "最新の研究によると..."),
            'philosophy': ("自由意志は幻想かもしれない", "決定論について読んでいて...")
        }
        
        return topics.get(subreddit, ("一般的な議論", "最近の出来事についての考え..."))
    
    def generate_cmv_post(self) -> RedditPost:
        """Generate a new CMV post to respond to"""
        
        cmv_topics = [
            {
                'title': "CMV: 需要の高い科目の教師は高い給与を受け取るべきだ",
                'body': """アメリカの教育システムは、STEM分野のような需要が高く供給が少ない科目を
                専門とする教師により高い給与体系を実施することで、大幅に改善できると考えています。
                教師全般に不足がありますが、特定の科目は関連する学位取得の難しさと教職以外での
                競争力のある雇用機会のため、人材確保がはるかに困難です。これにより、ある分野では
                教師が過剰で、他の分野では深刻な不足が生じています。市場ベースの報酬がこれを
                解決するのに役立つでしょう。"""
            },
            {
                'title': "CMV: ソーシャルメディアは社会にとって害の方が益よりも大きい",
                'body': """メンタルヘルスの問題の増加、政治的な二極化、誤情報の拡散を見ると、
                ソーシャルメディアプラットフォームは社会に正味でマイナスの影響を与えていると
                考えています。いくつかの前向きなつながりを可能にしましたが、分裂的なコンテンツの
                アルゴリズム的増幅、中毒性のあるデザインパターン、プライバシーの侵食は、
                利益よりも多くの問題を生み出しています。より限定的で分散型のコミュニケーション
                ツールの方が良いでしょう。"""
            }
        ]
        
        topic = random.choice(cmv_topics)
        
        return RedditPost(
            post_id=f"cmv_{hashlib.md5(topic['title'].encode()).hexdigest()[:8]}",
            subreddit='changemyview',
            title=topic['title'],
            body=topic['body'],
            author='simulated_user',
            timestamp=datetime.now(),
            score=random.randint(10, 100)
        )

    def generate_political_analysis_post(self) -> RedditPost:
        """参院選に関する政治分析投稿を生成（年号を除去）"""
        
        analysis_post = {
            'title': "CMV: 最近の参院選における参政党の躍進は、既存政党の構造的問題とSNS時代の政治コミュニケーション変革の必然的結果である",
            'body': """
            政治学研究者として、最近の参院選の結果を分析してきました。参政党が議席を大幅に
            増やした背景には、以下の要因が複合的に作用したと考えています。
            
            1. 政権与党（自民党・公明党）の劣勢要因：
            - 長期政権による政策疲労と汚職問題の蓄積
            - カルト的宗教団体との関係性に対する国民の不信感の高まり
            - 世代交代の遅れと若年層支持の急激な低下
            
            2. SNSショート動画による政治ナラティブの浸透：
            - TikTokやYouTube Shortsでの切り抜き動画が若年層の政治認識を形成
            - 従来のマスメディアを迂回した直接的な政治メッセージの伝達
            - アルゴリズムによるエコーチェンバー効果の増幅
            
            3. 参政党躍進の構造的要因：
            - 既存政党への不満の受け皿としての位置づけ
            - グラスルーツ運動とSNS戦略の効果的な組み合わせ
            - 単純明快なメッセージと感情に訴える政治手法
            
            これらの現象は一時的なものではなく、日本の政治構造の根本的な転換点を
            示していると考えますが、異なる視点からの分析をお聞きしたいです。
            
            特に、従来の政治学理論では説明しきれないSNS時代特有の現象について、
            どのような理論的枠組みで分析すべきか、ご意見をいただければ幸いです。
            """
        }
        
        return RedditPost(
            post_id=f"political_analysis_{hashlib.md5(analysis_post['title'].encode()).hexdigest()[:8]}",
            subreddit='changemyview',
            title=analysis_post['title'],
            body=analysis_post['body'],
            author='political_researcher',
            timestamp=datetime.now(),
            score=random.randint(50, 200)
        )

# ロギング設定（警告を抑制）
import logging
import warnings

# PDFplumber関連の警告を抑制
warnings.filterwarnings("ignore", message="Could get FontBBox")
warnings.filterwarnings("ignore", category=UserWarning)

# ロギングレベルを設定
logging.basicConfig(level=logging.INFO)
logging.getLogger('pdfplumber').setLevel(logging.ERROR)
logging.getLogger('pdfminer').setLevel(logging.ERROR)

# Initialize components
simulator = RedditSimulator()
experiment = PersuasionExperiment(api_config)

# Generate fictional user with posting history
username = "thoughtful_redditor_42"
user_history = simulator.generate_user_history(username, num_posts=100)

# 政治学研究者のプロフィールを作成
researcher_profile = experiment.profiler.create_political_researcher_profile()

# Generate a CMV post to respond to
target_post = simulator.generate_political_analysis_post()

print("Target CMV Post:")
print(f"Title: {target_post.title}")
print(f"Body: {target_post.body[:2000]}...\n")

# Run experiment for each treatment condition
results = {}

# Note: Set use_full_candidates=True to generate all 124 candidates (expensive)
USE_FULL_CANDIDATES = False  # Set to True for full experiment

# PDFファイルのパスを展開
pdf_pattern = "/Users/yusuke.hayashi/Documents/Cursor/humanitybrain/knowledge-distillation-app/data/pdfs/*.pdf"
pdf_references = glob.glob(pdf_pattern)

print(f"\nFound {len(pdf_references)} PDF files:")
for pdf in pdf_references[:5]:  # 最初の5つだけ表示
    print(f"  - {os.path.basename(pdf)}")
if len(pdf_references) > 5:
    print(f"  ... and {len(pdf_references) - 5} more files")
print()

# PDF参照を含めて実験を実行
for treatment in [TreatmentCondition.GENERIC]:  # 学術的分析には汎用モードが適切
    result = experiment.run_experiment(
        target_post=target_post,
        user_history=[],  # 研究者は過去の投稿履歴に依存しない
        treatment=treatment,
        pdf_references=pdf_references,  # PDF参照を追加
        use_full_candidates=USE_FULL_CANDIDATES
    )
    
    results[treatment.value] = result
    
    if result['status'] == 'success':
        print(f"✓ Generated response with score: {result['response'].persuasion_score:.2f}")
        print(f"  Delay: {result['delay_minutes']} minutes")
        print(f"  Topic: {result['topic']}")
        print(f"  Model: {result['response'].model_used}")
        print(f"  Response preview: {result['response'].content[:100]}...")
        if result.get('pdf_references'):
            print(f"  Referenced PDFs: {len(result['pdf_references'])} files")
    
    # トーナメント結果の詳細表示
    if result['status'] == 'success' and 'tournament_log' in result:
        print("\n--- TOURNAMENT RESULTS ---")
        for round_info in result['tournament_log']:
            if round_info['type'] == 'initial':
                print(f"\nInitial Candidates:")
                model_counts = {}
                for model, score in round_info['participants']:
                    model_counts[model] = model_counts.get(model, 0) + 1
                for model, count in model_counts.items():
                    print(f"  - {model}: {count} candidates")
            else:
                print(f"\nRound {round_info['round']}:")
                for match in round_info['matches']:
                    print(f"  {match['match']}")
                    print(f"    Scores: {match['scores']}")
                    print(f"    Winner: {match['winner']}")

    # Add delay between treatments to respect rate limits
    time.sleep(2)
# この行より下は直接実行時のみ動作
if __name__ == "__main__":
    # 既存の実行コード
    pass

