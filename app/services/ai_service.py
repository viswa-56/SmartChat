# # app/services/ai_service.py
# import requests
# import json
# from typing import Dict, Any
# from app.models.schemas import QueryRequest, QueryResponse
# from app.utils.config import settings


# class AIService:
#     def __init__(self):
#         # self.api_url = "https://router.huggingface.co/together/v1/chat/completions"
#         self.api_url = "https://api-inference.huggingface.co/models/"
#         self.headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
#                         "Content-Type": "application/json"}
    
#     async def generate_response(self, query: str, context: str, sources: list) -> QueryResponse:
#         """Generate response using free AI model with strict prompt engineering"""
        
#         # Engineered prompt to restrict responses to document content only
#         prompt = f"""
# {settings.SYSTEM_PROMPT}

# Context from documents:
# {context}

# Question: {query}

# Instructions:
# 1. Answer ONLY based on the provided context
# 2. If the answer is not in the context, say "I cannot answer this question based on the provided documents."
# 3. Always mention which document(s) you're referencing
# 4. Be concise and accurate

# Answer:"""

#         try:
#             # Use Hugging Face Inference API (free tier)
#             response = await self._call_huggingface_api(prompt)
            
#             # Calculate confidence based on context relevance
#             confidence = self._calculate_confidence(query, context, response)
            
#             return QueryResponse(
#                 answer=response,
#                 sources=sources,
#                 confidence_score=confidence,
#                 session_id="default"
#             )
            
#         except Exception as e:
#             return QueryResponse(
#                 answer="I apologize, but I'm unable to process your request at the moment. Please try again.",
#                 sources=[],
#                 confidence_score=0.0,
#                 session_id="default"
#             )
    
#     async def _call_huggingface_api(self, prompt: str) -> str:
#         """Call Hugging Face API"""
#         url = f"{self.api_url}{settings.AI_MODEL}"
        
#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": 512,
#                 "temperature": 0.1,
#                 "do_sample": True,
#                 "return_full_text": False
#             }
#         }
        
#         response = requests.post(url, headers=self.headers, json=payload)
#         # print('airesponse'+ response+ 'airesponse End')
#         print(response)
#         if response.status_code == 200:
#             result = response.json()
#             if isinstance(result, list) and len(result) > 0:
#                 return result[0].get('generated_text', '').strip()
#             return str(result)
#         else:
#             raise Exception(f"API call failed: {response.status_code}")
    
#     def _calculate_confidence(self, query: str, context: str, response: str) -> float:
#         """Calculate confidence score based on context relevance"""
#         if not context or "cannot answer" in response.lower():
#             return 0.0
        
#         # Simple confidence calculation based on context overlap
#         query_words = set(query.lower().split())
#         context_words = set(context.lower().split())
#         response_words = set(response.lower().split())
        
#         context_overlap = len(query_words.intersection(context_words))
#         response_overlap = len(response_words.intersection(context_words))
        
#         confidence = min(1.0, (context_overlap + response_overlap) / (len(query_words) + 1))
#         return confidence


# app/services/ai_service.py
import os
import requests
import json
import logging
from typing import Dict, Any, List
from app.models.schemas import QueryRequest, QueryResponse
from app.utils.config import settings

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        # Use the working chat completions endpoint
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        # "https://router.huggingface.co/together/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        # self.api_url = "https://api.together.xyz/v1/chat/completions"
        # self.headers = {
        # #     "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        # #     "Content-Type": "application/json"
        # }
        # self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        # print(os.getenv('TOGETHER_API_KEY'))
        self.model = "meta-llama/Llama-3.3-70B-Instruct"

    async def generate_response(self, query: str, context: str, sources: list) -> QueryResponse:
        """Generate response using chat completions API"""
        
        if not context.strip():
            return QueryResponse(
                answer="I cannot answer this question based on the provided documents. No relevant content was found.",
                sources=[],
                confidence_score=0.0,
                session_id="default"
            )
        
        try:
            # Create messages in OpenAI format
            messages = self._create_chat_messages(query, context)
            
            # Call the API
            response_text = await self._call_chat_completions_api(messages)
            
            # Calculate confidence
            confidence = self._calculate_confidence(query, context, response_text)
            
            return QueryResponse(
                answer=response_text,
                sources=sources,
                confidence_score=confidence,
                session_id="default"
            )
            
        except Exception as e:
            logger.error(f"AI Service Error: {str(e)}")
            return QueryResponse(
                answer=f"I encountered an error processing your request: {str(e)}",
                sources=[],
                confidence_score=0.0,
                session_id="default"
            )
    
    def _create_chat_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """Create chat messages in OpenAI format"""
        system_message = {
            "role": "system",
            "content": f"""You are a helpful assistant that answers questions based ONLY on the provided context. 
If the question cannot be answered from the context, respond with "I cannot answer this question based on the provided documents."
Always cite the source of your information.

Context from documents:
{context}"""
        }
        
        user_message = {
            "role": "user",
            "content": query
        }
        
        return [system_message, user_message]
    
    async def _call_chat_completions_api(self, messages: List[Dict[str, str]]) -> str:
        """Call the chat completions API"""
        payload = {
            "messages": messages,
            "model": settings.AI_MODEL,
            # self.model,
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9
        }
        
        logger.info(f"Calling chat completions API with model: {settings.AI_MODEL}")
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            logger.info(f"API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the response from the chat completions format
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0]["message"]
                    return message["content"].strip()
                else:
                    raise Exception("No choices in API response")
            
            elif response.status_code == 401:
                raise Exception("API authentication failed - check your API key")
            elif response.status_code == 429:
                raise Exception("API rate limit exceeded")
            else:
                error_detail = response.text
                raise Exception(f"API call failed with status {response.status_code}: {error_detail}")
                
        except requests.exceptions.Timeout:
            raise Exception("API request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to API")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from API")
    
    def _calculate_confidence(self, query: str, context: str, response: str) -> float:
        """Calculate confidence score based on context relevance"""
        if not context or "cannot answer" in response.lower():
            return 0.0
        
        # Simple confidence calculation based on context overlap
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        
        context_overlap = len(query_words.intersection(context_words))
        response_overlap = len(response_words.intersection(context_words))
        
        confidence = min(1.0, (context_overlap + response_overlap) / (len(query_words) + 1))
        return confidence
