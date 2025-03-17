"""
LLM Integration module for Cognito that adds AI-powered code analysis.

This module enhances Cognito with:
1. Smart code review recommendations
2. Interactive code assistant for Q&A
3. Natural language explanations of code issues
4. Automated refactoring suggestions
"""

import os
from typing import Dict, List, Any, Optional, Union
import logging

from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configure logging
logger = logging.getLogger(__name__)

class CodeAssistant:
    """AI-powered code assistant using LangChain and LLMs."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """
        Initialize the code assistant.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for generation (0.0-1.0)
        """
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key found. LLM features will be limited.")
            self.llm_available = False
            return
            
        self.llm_available = True
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key,
            streaming=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        
        # Initialize embeddings for semantic search
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the chains and agent
        self._initialize_chains()
        self._initialize_agent()
        
        logger.info(f"Code Assistant initialized with model: {model_name}")
    
    def _initialize_chains(self):
        """Initialize the various LLM chains for code tasks."""
        # Chain for generating code explanations
        self.explanation_prompt = PromptTemplate(
            input_variables=["code"],
            template="""
            You are a code explanation expert. Explain the following code in a clear,
            concise manner that would help a junior developer understand it:
            
            ```
            {code}
            ```
            
            Focus on:
            1. The purpose of the code
            2. How it works step by step
            3. Any important patterns or techniques used
            4. Potential issues or improvements
            """
        )
        
        self.explanation_chain = LLMChain(
            llm=self.llm,
            prompt=self.explanation_prompt,
            output_key="explanation"
        )
        
        # Chain for code review feedback
        self.review_prompt = PromptTemplate(
            input_variables=["code", "analysis_results"],
            template="""
            You are an expert code reviewer. Review the following code and provide
            constructive feedback. I've already run some automated analysis with these results:
            
            ANALYSIS RESULTS:
            {analysis_results}
            
            CODE:
            ```
            {code}
            ```
            
            Provide your feedback on:
            1. Code quality and readability
            2. Potential bugs or edge cases
            3. Architecture and design patterns
            4. Performance considerations
            5. Security concerns
            
            For each issue, explain why it matters and how to fix it.
            """
        )
        
        self.review_chain = LLMChain(
            llm=self.llm,
            prompt=self.review_prompt,
            output_key="review_feedback"
        )
        
        # Chain for refactoring suggestions
        self.refactor_prompt = PromptTemplate(
            input_variables=["code", "issue"],
            template="""
            You are a code refactoring expert. Given the following code and an issue description,
            provide a specific refactoring suggestion with before and after code examples.
            
            CODE:
            ```
            {code}
            ```
            
            ISSUE: {issue}
            
            Provide your refactoring suggestion with:
            1. A brief explanation of the refactoring
            2. The original code snippet that needs refactoring
            3. The refactored code snippet
            4. Benefits of this change
            """
        )
        
        self.refactor_chain = LLMChain(
            llm=self.llm,
            prompt=self.refactor_prompt,
            output_key="refactoring"
        )
    
    def _initialize_agent(self):
        """Initialize the code assistant agent."""
        if not self.llm_available:
            return
            
        # Define tools the agent can use
        tools = [
            Tool(
                name="ExplainCode",
                func=lambda code: self.explanation_chain.run(code=code),
                description="Explains code in simple terms. Input should be a code snippet."
            ),
            Tool(
                name="ReviewCode",
                func=lambda x: self.review_chain.run(
                    code=x.get("code", ""),
                    analysis_results=x.get("analysis_results", "No analysis results provided.")
                ),
                description="Reviews code and provides feedback. Input should be a JSON with 'code' and 'analysis_results' keys."
            ),
            Tool(
                name="SuggestRefactoring",
                func=lambda x: self.refactor_chain.run(
                    code=x.get("code", ""),
                    issue=x.get("issue", "Improve code quality")
                ),
                description="Suggests ways to refactor code. Input should be a JSON with 'code' and 'issue' keys."
            )
        ]
        
        # Create the agent
        system_message = SystemMessage(
            content="""You are CognitoGPT, an AI coding assistant integrated with the Cognito code analysis tool.
            You can explain code, review code, and suggest refactorings. You're helpful, concise, and technical.
            When providing code suggestions, focus on practical, proven solutions.
            Always format code blocks using markdown syntax with the appropriate language.
            """
        )
        
        self.agent = initialize_agent(
            tools,
            self.llm,
            agent="chat-conversational-react-description",
            verbose=True,
            memory=self.memory,
            agent_kwargs={
                "system_message": system_message,
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
            }
        )
        
        logger.info("Code Assistant agent initialized successfully")
    
    def create_vectorstore_from_code(self, code_files: Dict[str, str]) -> None:
        """
        Create a vector store from a collection of code files for semantic search.
        
        Args:
            code_files: Dictionary of filename to code content
        """
        if not self.llm_available:
            logger.warning("LLM features not available. Skipping vector store creation.")
            return
            
        # Combine all files into documents
        documents = []
        for file_path, content in code_files.items():
            documents.append({
                "content": content,
                "metadata": {"source": file_path}
            })
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        texts = []
        metadatas = []
        for doc in documents:
            chunks = text_splitter.split_text(doc["content"])
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({"source": doc["metadata"]["source"]})
        
        # Create vector store
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Vector store created with {len(texts)} chunks from {len(code_files)} files")
    
    def semantic_code_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for code semantically using the query.
        
        Args:
            query: Natural language query
            n_results: Number of results to return
            
        Returns:
            List of relevant code chunks with metadata
        """
        if not self.llm_available or not hasattr(self, "vectorstore"):
            return [{"content": "LLM features not available or no code indexed.", "metadata": {}}]
            
        results = self.vectorstore.similarity_search(query, k=n_results)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    
    def explain_code(self, code: str) -> str:
        """
        Generate a natural language explanation of the provided code.
        
        Args:
            code: Code snippet to explain
            
        Returns:
            Natural language explanation
        """
        if not self.llm_available:
            return "LLM features not available. Please set OPENAI_API_KEY."
            
        return self.explanation_chain.run(code=code)
    
    def review_code(self, code: str, analysis_results: str) -> str:
        """
        Generate detailed code review feedback.
        
        Args:
            code: Code to review
            analysis_results: Results from Cognito analysis
            
        Returns:
            Detailed review feedback
        """
        if not self.llm_available:
            return "LLM features not available. Please set OPENAI_API_KEY."
            
        return self.review_chain.run(code=code, analysis_results=analysis_results)
    
    def suggest_refactoring(self, code: str, issue: str) -> str:
        """
        Generate refactoring suggestions for specific issues.
        
        Args:
            code: Code to refactor
            issue: Description of the issue to fix
            
        Returns:
            Refactoring suggestion
        """
        if not self.llm_available:
            return "LLM features not available. Please set OPENAI_API_KEY."
            
        return self.refactor_chain.run(code=code, issue=issue)
    
    def chat(self, user_input: str) -> str:
        """
        Interactive chat with the code assistant.
        
        Args:
            user_input: User's question or request
            
        Returns:
            Assistant's response
        """
        if not self.llm_available:
            return "LLM features not available. Please set OPENAI_API_KEY."
            
        return self.agent.run(user_input)


class CognitoLLMEnhancer:
    """
    Enhances Cognito analysis results with LLM-powered insights.
    """
    
    def __init__(self):
        """Initialize the LLM enhancer."""
        self.code_assistant = CodeAssistant()
    
    def enhance_analysis(self, code: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance analysis results with LLM-generated insights.
        
        Args:
            code: The code being analyzed
            analysis_results: Original Cognito analysis results
            
        Returns:
            Enhanced analysis results with LLM insights
        """
        if not self.code_assistant.llm_available:
            analysis_results["llm_insights"] = {
                "available": False,
                "message": "LLM features not available. Please set OPENAI_API_KEY."
            }
            return analysis_results
        
        # Convert analysis results to a string for the LLM
        analysis_str = self._format_analysis_for_llm(analysis_results)
        
        # Get LLM review
        llm_review = self.code_assistant.review_code(code, analysis_str)
        
        # Extract suggestions for top issues
        top_issues = self._extract_top_issues(analysis_results)
        refactoring_suggestions = {}
        
        for issue_key, issue_desc in top_issues.items():
            suggestion = self.code_assistant.suggest_refactoring(code, issue_desc)
            refactoring_suggestions[issue_key] = suggestion
        
        # Add LLM insights to results
        analysis_results["llm_insights"] = {
            "available": True,
            "overall_review": llm_review,
            "refactoring_suggestions": refactoring_suggestions,
            "code_explanation": self.code_assistant.explain_code(code)
        }
        
        return analysis_results
    
    def _format_analysis_for_llm(self, analysis_results: Dict[str, Any]) -> str:
        """Format the analysis results for the LLM in a readable format."""
        formatted = "ANALYSIS RESULTS:\n"
        
        if "language" in analysis_results:
            formatted += f"Language: {analysis_results['language']}\n\n"
        
        if "summary" in analysis_results:
            formatted += "SUMMARY:\n"
            summary = analysis_results["summary"]
            for key, value in summary.items():
                if isinstance(value, dict):
                    formatted += f"- {key.replace('_', ' ').title()}:\n"
                    for k, v in value.items():
                        formatted += f"  - {k.replace('_', ' ').title()}: {v}\n"
                else:
                    formatted += f"- {key.replace('_', ' ').title()}: {value}\n"
            formatted += "\n"
        
        if "suggestions" in analysis_results:
            formatted += "SUGGESTIONS:\n"
            for suggestion in analysis_results["suggestions"]:
                category = suggestion.get("category", "General")
                message = suggestion.get("message", "")
                priority = suggestion.get("priority", "medium")
                formatted += f"- [{priority.upper()}] {category}: {message}\n"
        
        return formatted
    
    def _extract_top_issues(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Extract the top issues from the analysis results."""
        top_issues = {}
        
        if "suggestions" in analysis_results:
            suggestions = analysis_results["suggestions"]
            high_priority = [s for s in suggestions if s.get("priority") == "high"]
            medium_priority = [s for s in suggestions if s.get("priority") == "medium"]
            
            # Take up to 3 high priority issues
            for i, issue in enumerate(high_priority[:3]):
                category = issue.get("category", "General")
                message = issue.get("message", "")
                key = f"high_{category}_{i}"
                top_issues[key] = message
            
            # Fill with medium priority if needed
            if len(top_issues) < 3:
                for i, issue in enumerate(medium_priority[:3 - len(top_issues)]):
                    category = issue.get("category", "General")
                    message = issue.get("message", "")
                    key = f"medium_{category}_{i}"
                    top_issues[key] = message
        
        return top_issues


# Example usage
if __name__ == "__main__":
    # Sample code for testing
    sample_code = """
    def factorial(n):
        # Recursive factorial function
        if n <= 0:
            return 1
        else:
            return n * factorial(n-1)
    
    def process_data(file_path):
        # Process data from a file
        data = open(file_path).readlines()
        result = ""
        for line in data:
            result = result + line
        return result
    """
    
    # Sample analysis results
    sample_analysis = {
        "language": "python",
        "summary": {
            "maintainability": {
                "index": 65.5,
                "rating": "Good"
            },
            "security": "Medium Risk",
            "performance": "Needs Improvement"
        },
        "suggestions": [
            {
                "category": "Security",
                "message": "File is opened without proper error handling or context manager",
                "priority": "high"
            },
            {
                "category": "Performance",
                "message": "Inefficient string concatenation in loop",
                "priority": "medium"
            },
            {
                "category": "Performance",
                "message": "Recursive function without memoization",
                "priority": "low"
            }
        ]
    }
    
    # Initialize enhancer
    enhancer = CognitoLLMEnhancer()
    
    # Enhance analysis
    enhanced_results = enhancer.enhance_analysis(sample_code, sample_analysis)
    
    # Print enhanced results
    import json
    print(json.dumps(enhanced_results, indent=2))