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

# Updated imports for LangChain
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent

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
        try:
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key,
                streaming=True
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
        except Exception as e:
            logger.error(f"Error initializing LLM components: {str(e)}")
            self.llm_available = False
    
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
        
        try:    
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
            
            # Create the agent - using simple string for system message to avoid template issues
            system_message = "You are CognitoGPT, an AI coding assistant integrated with the Cognito code analysis tool. You can explain code, review code, and suggest refactorings. You're helpful, concise, and technical. When providing code suggestions, focus on practical, proven solutions. Always format code blocks using markdown syntax with the appropriate language."
            
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
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            # Graceful degradation - we'll continue without the agent
            self.agent = None
    
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
        
        try:
            return self.explanation_chain.run(code=code)
        except Exception as e:
            logger.error(f"Error in code explanation: {e}")
            return f"Could not generate explanation: {str(e)}"
    
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
        
        try:
            return self.review_chain.run(code=code, analysis_results=analysis_results)
        except Exception as e:
            logger.error(f"Error in code review: {e}")
            return f"Could not generate review: {str(e)}"
    
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
        
        try:
            return self.refactor_chain.run(code=code, issue=issue)
        except Exception as e:
            logger.error(f"Error in refactoring suggestion: {e}")
            return f"Could not generate refactoring suggestion: {str(e)}"
    
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
        
        if self.agent is None:
            return "Chat functionality not available due to initialization error."
            
        try:
            return self.agent.run(user_input)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Could not process chat: {str(e)}"