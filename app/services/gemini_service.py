import google.generativeai as genai
from ..core.config import settings
import json
import re

class GeminiPolicyProcessor:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.embedding_model = 'models/text-embedding-004'

    def _parse_json_response(self, text: str) -> dict:
        """Safely parse JSON from a string that might contain markdown."""
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = text
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback for non-standard JSON, this is a simplified handler
            print(f"Warning: Could not parse JSON response directly. Response text: {text}")
            return {"error": "Failed to parse LLM response", "raw_response": text}

    async def extract_entities(self, query_text: str) -> dict:
        """Extract structured entities from natural language query"""
        prompt = f"""
        Extract structured information from this insurance query: "{query_text}"
        
        Return JSON with:
        {{
            "age": "integer or null",
            "gender": "'M' | 'F' | 'Other' | null",
            "procedure": "string or null",
            "location": "string or null",
            "policy_duration_months": "integer or null"
        }}
        
        Only extract explicitly mentioned information. Use null for missing data.
        Format the output as a JSON object inside a '```json' markdown block.
        """
        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)
    
    async def analyze_policy_clauses(self, query: dict, document_chunks: list) -> list:
        """Analyze policy clauses against query using Gemini"""
        prompt = f"""
        Analyze these policy document sections against the insurance query. Strictly follow all instructions.

        Query: {json.dumps(query)}

        Policy Sections:
        {json.dumps(document_chunks)}

        For each relevant section, return a JSON array of objects with the following structure:
        [{{
            "clause_id": "A unique reference or the first 10 words of the clause",
            "relevance_score": "A float from 0 to 1 indicating relevance to the query",
            "clause_type": "'inclusion' (provides coverage), 'exclusion' (denies coverage), 'condition' (a prerequisite), or 'general'",
            "matched_criteria": ["list", "of", "query", "criteria", "it", "matches"],
            "extracted_rules": {{
                "waiting_period_months": "integer, null if not mentioned",
                "pre_existing_condition_clause": "boolean, true if it relates to pre-existing conditions",
                "coverage_amount": "integer, null if not mentioned",
                "exclusions_mentioned": ["list", "of", "specific", "exclusions"],
                "conditions_mentioned": ["list", "of", "specific", "conditions"]
            }},
            "reasoning": "A brief explanation of how this clause applies to the query."
        }}]

        Instructions:
        1.  **Prioritize Identification**: First, identify if a clause is an inclusion, exclusion, or a condition for coverage.
        2.  **Extract Key Rules**: Pay close attention to waiting periods, pre-existing conditions, and specific coverage amounts.
        3.  **Be Precise**: If a value is not explicitly mentioned, use null.
        4.  **Return Empty Array**: If no clauses are relevant, return an empty array [] in the JSON format.
        
        Format the output as a JSON object inside a '```json' markdown block.
        """
        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)
    
    async def generate_embeddings(self, text: str, task_type="retrieval_document") -> list:
        """Generate embeddings using Gemini's embedding capabilities"""
        response = await genai.embed_content_async(
            model=self.embedding_model,
            content=text,
            task_type=task_type
        )
        return response['embedding']

    async def generate_embeddings_batch(self, texts: list[str], task_type="retrieval_document") -> list:
        """Generate embeddings for a batch of texts for improved efficiency."""
        response = await genai.embed_content_async(
            model=self.embedding_model,
            content=texts,
            task_type=task_type
        )
        return response['embedding']
        
    async def final_decision_reasoning(self, query: dict, analyzed_clauses: list) -> dict:
        """Generate final decision with detailed reasoning"""
        prompt = f"""
        Make a final insurance claim decision based on the query and the policy analysis, following a strict set of rules.

        Query: {json.dumps(query)}
        Policy Analysis: {json.dumps(analyzed_clauses)}

        **Decision-Making Rules (MUST be followed in this order):**
        1.  **Exclusion Priority**: If any relevant clause is an 'exclusion', the decision MUST be 'rejected', regardless of other clauses.
        2.  **Waiting Period Validation**: Check if a waiting period is required. If the policy duration in the query is less than the required waiting period, the decision MUST be 'rejected'.
        3.  **Pre-existing Conditions**: If the query mentions a pre-existing condition and a relevant clause excludes it, the decision MUST be 'rejected'.
        4.  **Approval Logic**: If no exclusions or unmet conditions apply, and there is at least one 'inclusion' clause, the decision is 'approved'.
        5.  **Review Required**: If the information is insufficient to make a clear decision, the status is 'requires_review'.

        Return a single JSON object with the following structure:
        {{
            "decision": "'approved' | 'rejected' | 'requires_review'",
            "confidence_score": "A float from 0 to 1, reflecting certainty in the decision",
            "approved_amount": "The final approved amount as an integer, or 0 if rejected",
            "reasoning": "A detailed, step-by-step explanation for the decision, explicitly referencing the rules applied.",
            "risk_factors": ["A list of identified risk factors, if any, such as 'Exclusion clause applied' or 'Waiting period not met'"],
            "recommendations": ["A list of recommendations, e.g., 'request more documents for clarity on pre-existing condition'"]
        }}

        Be conservative and justify every decision by citing the rule that was triggered. Format the output as a JSON object inside a '```json' markdown block.
        """
        response = await self.model.generate_content_async(prompt)
        return self._parse_json_response(response.text)

    async def generate_answer_from_context(self, question: str, context_chunks: list[str]) -> str:
        """Generates a structured, formal answer to an insurance policy question based on provided context.
        
        Returns:
            str: A well-formatted answer with policy details, conditions, and references.
        """
        context_text = "\n".join(context_chunks)
        prompt = f"""You are an expert insurance policy analyst. Provide a clear, formal answer to the question based EXCLUSIVELY on the provided policy document context.
        
INSTRUCTIONS:
1. Answer in complete, grammatically correct sentences.
2. Be precise and avoid vague language (no "might be" or "could be").
3. Include specific policy terms, conditions, and limitations.
4. Format numbers and dates consistently (e.g., "thirty (30) days").
5. If information is not found in the context, state: "This information is not specified in the provided policy document."
        
STRUCTURE YOUR ANSWER AS FOLLOWS:
[Summary] A brief, direct answer to the question.

[Conditions] Any applicable conditions, waiting periods, or requirements.

[Limitations] Any coverage limits, sub-limits, or exclusions.

[Reference] The specific section or clause from the policy document, if available.

Question: {question}

Policy Document Context:
---
{context_text}
---

Provide your response in the specified format. Only include information that is explicitly stated in the context."""
        
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.2,  # Lower temperature for more focused, deterministic responses
                    "top_p": 0.8,
                    "max_output_tokens": 1024,
                }
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
