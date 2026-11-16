# integrated_translator.py
import logging
from typing import Dict, Tuple, Optional, List, Any
import asyncio
import spacy
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse

# Import your existing components
from translator import translate_text, batch_translate, translation_memory, Translation
from bubble_tracker import bubble_tracker, BubbleInfo
from virtual_manga_page import ContextAwareMangaProcessor, ReadingOrder

class TranslationAnalyzer:
    """
    Enhanced translation analyzer with spaCy integration
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.SPACY_AVAILABLE = True
            logging.info("spaCy translation analyzer initialized")
        except OSError:
            logging.warning("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.SPACY_AVAILABLE = False
            self.nlp = None
    
    def analyze_translation_quality(self, original_text: str, translated_text: str) -> Dict[str, Any]:
        """Comprehensive analysis of translation quality"""
        if not self.SPACY_AVAILABLE or not translated_text.strip():
            return {"error": "spaCy not available or empty text"}
        
        doc = self.nlp(translated_text)
        
        return {
            "basic_metrics": self._get_basic_metrics(doc),
            "readability": self._analyze_readability(doc),
            "grammar_issues": self._find_grammar_issues(doc),
            "entities": self._extract_entities(doc),
            "quality_score": self._calculate_quality_score(doc, original_text)
        }
    
    def _get_basic_metrics(self, doc) -> Dict[str, Any]:
        """Get basic text metrics"""
        sentences = list(doc.sents)
        tokens = [token for token in doc if token.is_alpha]
        
        return {
            "sentence_count": len(sentences),
            "word_count": len(tokens),
            "avg_sentence_length": len(tokens) / max(len(sentences), 1),
            "unique_words": len(set(token.lemma_.lower() for token in tokens))
        }
    
    def _analyze_readability(self, doc) -> Dict[str, float]:
        """Analyze text readability for manga dialogue"""
        metrics = self._get_basic_metrics(doc)
        
        # Manga dialogue should be conversational - score based on ideal ranges
        score_factors = []
        
        # Ideal sentence length for dialogue: 5-15 words
        avg_length = metrics["avg_sentence_length"]
        if 5 <= avg_length <= 15:
            score_factors.append(1.0)
        else:
            score_factors.append(max(0, 1.0 - abs(avg_length - 10) / 20))
        
        # Lexical diversity (higher is better for natural speech)
        diversity = metrics["unique_words"] / max(metrics["word_count"], 1)
        score_factors.append(min(diversity * 2, 1.0))  # Scale to 0-1
        
        return {
            "readability_score": sum(score_factors) / len(score_factors),
            "sentence_length_score": score_factors[0],
            "diversity_score": score_factors[1]
        }
    
    def _find_grammar_issues(self, doc) -> List[Dict[str, Any]]:
        """Find potential grammar and style issues"""
        issues = []
        
        # Check for passive voice
        passive_verbs = [token for token in doc if token.dep_ == "nsubjpass"]
        if passive_verbs:
            issues.append({
                "type": "passive_voice",
                "severity": "medium",
                "message": f"Passive construction detected",
                "count": len(passive_verbs)
            })
        
        # Check for long noun chunks (can be awkward in dialogue)
        long_chunks = [chunk for chunk in doc.noun_chunks if len(chunk) > 4]
        if long_chunks:
            issues.append({
                "type": "complex_noun_phrase", 
                "severity": "low",
                "message": f"Complex noun phrases: {[chunk.text for chunk in long_chunks[:2]]}",
                "examples": [chunk.text for chunk in long_chunks[:2]]
            })
        
        # Check for sentence fragments
        sentences = list(doc.sents)
        fragments = []
        for sent in sentences:
            has_verb = any(token.pos_ == "VERB" for token in sent)
            has_subject = any(token.dep_ in ("nsubj", "nsubjpass") for token in sent)
            if not has_verb and len(sent) > 3:
                fragments.append(sent.text)
        
        if fragments:
            issues.append({
                "type": "sentence_fragment",
                "severity": "high",
                "message": "Sentence fragments may sound unnatural",
                "examples": fragments[:2]
            })
        
        return issues
    
    def _extract_entities(self, doc) -> List[Dict[str, str]]:
        """Extract named entities that should be preserved"""
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "description": spacy.explain(ent.label_)
            })
        return entities
    
    def _calculate_quality_score(self, doc, original_text: str) -> float:
        """Calculate overall translation quality score (0-1)"""
        if len(doc) == 0:
            return 0.5
        
        readability = self._analyze_readability(doc)
        metrics = self._get_basic_metrics(doc)
        issues = self._find_grammar_issues(doc)
        
        # Base score from readability
        score = readability["readability_score"] * 0.6
        
        # Penalty for grammar issues
        issue_penalty = min(len(issues) * 0.1, 0.3)
        score -= issue_penalty
        
        # Bonus for good sentence structure
        if metrics["sentence_count"] > 0 and metrics["avg_sentence_length"] > 5:
            score += 0.1
        
        return max(0.1, min(1.0, score))

class ContextConsistencyAnalyzer:
    """
    Analyzes context consistency across manga bubbles
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.SPACY_AVAILABLE = True
        except OSError:
            self.SPACY_AVAILABLE = False
    
    def analyze_context_consistency(self, current_bubble: str, context_bubbles: List[Dict]) -> Dict[str, Any]:
        """Analyze how well current translation fits with context"""
        if not self.SPACY_AVAILABLE:
            return {"error": "spaCy not available"}
        
        current_doc = self.nlp(current_bubble)
        
        analysis = {
            "similarity_scores": [],
            "entity_flow": self._analyze_entity_flow(current_doc, context_bubbles),
            "topic_continuity": self._analyze_topic_continuity(current_doc, context_bubbles),
            "consistency_issues": []
        }
        
        # Calculate similarity with context
        for i, context in enumerate(context_bubbles[-3:]):  # Last 3 context bubbles
            context_text = context.get('translated_text') or context.get('text') or ''
            if context_text:
                context_doc = self.nlp(context_text)
                similarity = current_doc.similarity(context_doc)
                analysis["similarity_scores"].append({
                    "context_index": i,
                    "similarity": round(similarity, 3),
                    "context_preview": context_text[:30] + "..." if len(context_text) > 30 else context_text
                })
        
        # Overall consistency assessment
        if analysis["similarity_scores"]:
            avg_similarity = sum(s["similarity"] for s in analysis["similarity_scores"]) / len(analysis["similarity_scores"])
            analysis["average_similarity"] = round(avg_similarity, 3)
            
            if avg_similarity < 0.3:
                analysis["consistency_issues"].append("Low semantic similarity with context")
            if avg_similarity > 0.8:
                analysis["consistency_issues"].append("Very high similarity - possible repetition")
        
        return analysis
    
    def _analyze_entity_flow(self, current_doc, context_bubbles) -> Dict[str, Any]:
        """Analyze how entities flow across dialogue"""
        current_entities = {(ent.text.lower(), ent.label_) for ent in current_doc.ents}
        
        # Collect entities from recent context
        context_entities = set()
        for context in context_bubbles[-3:]:
            context_text = context.get('translated_text') or context.get('text') or ''
            if context_text:
                context_doc = self.nlp(context_text)
                context_entities.update({(ent.text.lower(), ent.label_) for ent in context_doc.ents})
        
        shared_entities = current_entities.intersection(context_entities)
        new_entities = current_entities - context_entities
        
        return {
            "shared_entities": [{"text": ent[0], "type": ent[1]} for ent in shared_entities],
            "new_entities": [{"text": ent[0], "type": ent[1]} for ent in new_entities],
            "entity_continuity": len(shared_entities) / max(len(current_entities), 1)
        }
    
    def _analyze_topic_continuity(self, current_doc, context_bubbles) -> Dict[str, Any]:
        """Analyze topic continuity using key nouns and verbs"""
        current_topics = {
            "nouns": [token.lemma_.lower() for token in current_doc if token.pos_ == "NOUN"],
            "verbs": [token.lemma_.lower() for token in current_doc if token.pos_ == "VERB" and token.lemma_ not in ("be", "have", "do")]
        }
        
        # Analyze context topics
        context_topics = {"nouns": [], "verbs": []}
        for context in context_bubbles[-3:]:
            context_text = context.get('translated_text') or context.get('text') or ''
            if context_text:
                context_doc = self.nlp(context_text)
                context_topics["nouns"].extend(token.lemma_.lower() for token in context_doc if token.pos_ == "NOUN")
                context_topics["verbs"].extend(token.lemma_.lower() for token in context_doc 
                                            if token.pos_ == "VERB" and token.lemma_ not in ("be", "have", "do"))
        
        # Calculate topic overlap
        noun_overlap = len(set(current_topics["nouns"]) & set(context_topics["nouns"]))
        verb_overlap = len(set(current_topics["verbs"]) & set(context_topics["verbs"]))
        
        total_current_topics = len(current_topics["nouns"]) + len(current_topics["verbs"])
        
        return {
            "noun_overlap": noun_overlap,
            "verb_overlap": verb_overlap,
            "topic_continuity_score": (noun_overlap + verb_overlap) / max(total_current_topics, 1)
        }

class EnhancedMangaTranslator:
    """
    Main enhanced translator that integrates all analysis components
    """
    
    def __init__(self):
        self.translation_analyzer = TranslationAnalyzer()
        self.context_analyzer = ContextConsistencyAnalyzer()
        self.context_processor = ContextAwareMangaProcessor()
        
        # Track analysis history for continuous improvement
        self.analysis_history = []
    
    async def translate_with_full_analysis(self, 
                                         text: str, 
                                         source_language: str = "auto", 
                                         bubble_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform translation with comprehensive quality and context analysis
        """
        # Step 1: Perform initial translation
        translated_text, detected_lang = await translate_text(text, source_language, bubble_id)
        
        # Step 2: Analyze translation quality
        quality_analysis = self.translation_analyzer.analyze_translation_quality(text, translated_text)
        
        # Step 3: Analyze context consistency if bubble_id provided
        context_analysis = {}
        if bubble_id:
            context_bubbles = self.context_processor.get_context_for_translation(bubble_id)
            context_analysis = self.context_analyzer.analyze_context_consistency(translated_text, context_bubbles)
        
        # Step 4: Calculate overall scores
        overall_quality = self._calculate_overall_quality(quality_analysis, context_analysis)
        improvement_suggestions = self._generate_improvement_suggestions(quality_analysis, context_analysis)
        
        # Step 5: Update tracking systems
        self._update_tracking_systems(bubble_id, translated_text, overall_quality, quality_analysis)
        
        result = {
            "translation_result": {
                "original_text": text,
                "translated_text": translated_text,
                "detected_language": detected_lang,
                "bubble_id": bubble_id
            },
            "quality_analysis": quality_analysis,
            "context_analysis": context_analysis,
            "overall_assessment": {
                "quality_score": overall_quality,
                "confidence_level": self._assess_confidence(overall_quality),
                "improvement_suggestions": improvement_suggestions,
                "recommended_actions": self._get_recommended_actions(quality_analysis, context_analysis)
            }
        }
        
        # Store analysis for learning
        self.analysis_history.append(result)
        
        return result
    
    def _calculate_overall_quality(self, quality_analysis: Dict, context_analysis: Dict) -> float:
        """Calculate overall quality score combining all factors"""
        base_score = quality_analysis.get("quality_score", 0.5)
        
        # Adjust based on context consistency
        if context_analysis and "average_similarity" in context_analysis:
            context_score = context_analysis["average_similarity"]
            # Weighted average: 70% translation quality, 30% context consistency
            overall = (base_score * 0.7) + (context_score * 0.3)
        else:
            overall = base_score
        
        return round(overall, 3)
    
    def _generate_improvement_suggestions(self, quality_analysis: Dict, context_analysis: Dict) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        # Quality-based suggestions
        if "readability" in quality_analysis:
            readability = quality_analysis["readability"]
            if readability.get("readability_score", 0) < 0.6:
                suggestions.append("Consider simplifying sentence structure for better readability")
        
        if "grammar_issues" in quality_analysis:
            issues = quality_analysis["grammar_issues"]
            for issue in issues:
                if issue.get("severity") == "high":
                    suggestions.append(f"Address: {issue.get('message', 'Grammar issue')}")
        
        # Context-based suggestions
        if context_analysis and "consistency_issues" in context_analysis:
            suggestions.extend(context_analysis["consistency_issues"])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _assess_confidence(self, quality_score: float) -> str:
        """Assess confidence level based on quality score"""
        if quality_score >= 0.8:
            return "high"
        elif quality_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_recommended_actions(self, quality_analysis: Dict, context_analysis: Dict) -> List[str]:
        """Get recommended actions based on analysis"""
        actions = []
        
        if quality_analysis.get("quality_score", 0) < 0.7:
            actions.append("Review translation for natural flow")
        
        if context_analysis and context_analysis.get("average_similarity", 0) < 0.4:
            actions.append("Check context consistency with surrounding dialogue")
        
        if not actions:
            actions.append("Translation appears good - no action needed")
        
        return actions
    
    def _update_tracking_systems(self, bubble_id: str, translated_text: str, 
                               quality_score: float, analysis: Dict):
        """Update bubble tracker and virtual page with analysis results"""
        if bubble_id:
            # Update bubble tracker with quality information
            try:
                bubble_tracker.update_bubble_translation(
                    bubble_id=bubble_id,
                    translated_text=translated_text,
                    target_language="en",
                    confidence=quality_score,
                    success=quality_score > 0.5
                )
            except Exception as e:
                logging.warning(f"Could not update bubble tracker: {e}")
            
            # Update virtual page
            try:
                self.context_processor.update_bubble_translation(
                    bubble_id, translated_text, quality_score
                )
            except Exception as e:
                logging.warning(f"Could not update virtual page: {e}")
    
    async def batch_translate_with_analysis(self, 
                                          texts: List[str], 
                                          source_language: str = "auto", 
                                          bubble_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Batch translation with analysis"""
        results = []
        
        for i, text in enumerate(texts):
            bubble_id = bubble_ids[i] if bubble_ids and i < len(bubble_ids) else None
            result = await self.translate_with_full_analysis(text, source_language, bubble_id)
            results.append(result)
        
        return results
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get analytics about translation quality over time"""
        if not self.analysis_history:
            return {"message": "No analysis data available"}
        
        quality_scores = [item["overall_assessment"]["quality_score"] 
                         for item in self.analysis_history]
        
        return {
            "total_translations_analyzed": len(self.analysis_history),
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "quality_trend": self._calculate_quality_trend(quality_scores),
            "common_issues": self._identify_common_issues(),
            "high_quality_ratio": len([s for s in quality_scores if s >= 0.8]) / len(quality_scores)
        }
    
    def _calculate_quality_trend(self, scores: List[float]) -> str:
        """Calculate if quality is improving over time"""
        if len(scores) < 2:
            return "insufficient_data"
        
        recent = scores[-5:]  # Last 5 translations
        earlier = scores[:5]  # First 5 translations
        
        if len(recent) < 2 or len(earlier) < 2:
            return "stable"
        
        avg_recent = sum(recent) / len(recent)
        avg_earlier = sum(earlier) / len(earlier)
        
        if avg_recent > avg_earlier + 0.1:
            return "improving"
        elif avg_recent < avg_earlier - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _identify_common_issues(self) -> List[str]:
        """Identify common issues across all translations"""
        all_issues = []
        for analysis in self.analysis_history[-20:]:  # Last 20 analyses
            suggestions = analysis["overall_assessment"].get("improvement_suggestions", [])
            all_issues.extend(suggestions)
        
        # Count issue frequency
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue] += 1
        
        return [issue for issue, count in sorted(issue_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:3]]

# Integration with FastAPI app
def setup_enhanced_translator_endpoints(app):
    """Add enhanced translator endpoints to FastAPI app"""
    enhanced_translator = EnhancedMangaTranslator()
    
    @app.post('/translate_with_analysis')
    async def translate_with_analysis(request: Request):
        """Enhanced translation endpoint with full analysis"""
        try:
            data = await request.json()
            text = data.get('text', '')
            source_lang = data.get('source_language', 'auto')
            bubble_id = data.get('bubble_id')
            
            if not text:
                return JSONResponse(status_code=400, content={"error": "No text provided"})
            
            # Run async function directly
            result = await enhanced_translator.translate_with_full_analysis(
                text, source_lang, bubble_id
            )
            
            return JSONResponse(content=result)
            
        except Exception as e:
            logging.error(f"Enhanced translation error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
    
    @app.post('/batch_translate_with_analysis')
    async def batch_translate_with_analysis(request: Request):
        """Batch translation with analysis"""
        try:
            data = await request.json()
            texts = data.get('texts', [])
            source_lang = data.get('source_language', 'auto')
            bubble_ids = data.get('bubble_ids', [])
            
            if not texts:
                return JSONResponse(status_code=400, content={"error": "No texts provided"})
            
            results = await enhanced_translator.batch_translate_with_analysis(
                texts, source_lang, bubble_ids
            )
            
            return JSONResponse(content={"translations": results})
            
        except Exception as e:
            logging.error(f"Batch enhanced translation error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
    
    @app.get('/translation_analytics')
    async def get_translation_analytics():
        """Get analytics about translation quality"""
        analytics = enhanced_translator.get_system_analytics()
        return JSONResponse(content=analytics)
    
    @app.post('/analyze_existing_translation')
    async def analyze_existing_translation(request: Request):
        """Analyze an existing translation without re-translating"""
        try:
            data = await request.json()
            original_text = data.get('original_text', '')
            translated_text = data.get('translated_text', '')
            bubble_id = data.get('bubble_id')
            
            if not translated_text:
                return JSONResponse(status_code=400, content={"error": "No translated text provided"})
            
            # Analyze quality
            quality_analysis = enhanced_translator.translation_analyzer.analyze_translation_quality(
                original_text, translated_text
            )
            
            # Analyze context if available
            context_analysis = {}
            if bubble_id:
                context_bubbles = enhanced_translator.context_processor.get_context_for_translation(bubble_id)
                context_analysis = enhanced_translator.context_analyzer.analyze_context_consistency(
                    translated_text, context_bubbles
                )
            
            overall_quality = enhanced_translator._calculate_overall_quality(quality_analysis, context_analysis)
            
            return JSONResponse(content={
                "quality_analysis": quality_analysis,
                "context_analysis": context_analysis,
                "overall_quality_score": overall_quality
            })
            
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
