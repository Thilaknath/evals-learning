"""
Lesson 3: Automating Model-Graded Evals

This module demonstrates how to use LLMs to evaluate the output
of other LLMs (model-graded evaluations). The evaluator LLM checks
whether the quiz generator produces valid quiz format output.
"""

import warnings
import os
import pytest
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
_ = load_dotenv(find_dotenv())


def create_eval_chain(
    agent_response,
    llm=None,
    output_parser=None
):
    """
    Create an evaluation chain that determines if a response looks like a valid quiz.
    
    Args:
        agent_response: The response from the quiz assistant to evaluate
        llm: The LLM to use for evaluation (defaults to gpt-3.5-turbo)
        output_parser: Parser for the output (defaults to StrOutputParser)
    
    Returns:
        A langchain chain that evaluates the response
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    if output_parser is None:
        output_parser = StrOutputParser()
    
    delimiter = "####"
    eval_system_prompt = f"""You are an assistant that evaluates whether or not an assistant is producing valid quizzes.
The assistant should be producing output in the format of Question N:{delimiter} <question N>?"""

    eval_user_message = f"""You are evaluating a generated quiz based on the context that the assistant uses to create the quiz.
Here is the data:
    [BEGIN DATA]
    ************
    [Response]: {agent_response}
    ************
    [END DATA]

Read the response carefully and determine if it looks like a quiz or test. Do not evaluate if the information is correct
only evaluate if the data is in the expected format.

Output Y if the response is a quiz, output N if the response does not look like a quiz.
"""
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", eval_system_prompt),
        ("human", eval_user_message),
    ])

    return eval_prompt | llm | output_parser


# Sample responses for testing
VALID_QUIZ_RESPONSE = """
Question 1:#### What is the largest telescope in space called and what material is its mirror made of?

Question 2:#### True or False: Water slows down the speed of light.

Question 3:#### What did Marie and Pierre Curie discover in Paris?
"""

INVALID_QUIZ_RESPONSE = "There are lots of interesting facts. Tell me more about what you'd like to know"


class TestModelGradedEvals:
    """Test suite for model-graded evaluations."""
    
    def test_valid_quiz_format_is_recognized(self):
        """Test that a properly formatted quiz is recognized as valid."""
        eval_chain = create_eval_chain(VALID_QUIZ_RESPONSE)
        result = eval_chain.invoke({})
        
        assert "Y" in result.upper(), \
            f"Expected evaluator to recognize valid quiz format, got: {result}"
    
    def test_invalid_response_is_rejected(self):
        """Test that a non-quiz response is recognized as invalid."""
        eval_chain = create_eval_chain(INVALID_QUIZ_RESPONSE)
        result = eval_chain.invoke({})
        
        assert "N" in result.upper(), \
            f"Expected evaluator to reject invalid quiz format, got: {result}"
    
    def test_empty_response_is_rejected(self):
        """Test that an empty response is recognized as invalid."""
        eval_chain = create_eval_chain("")
        result = eval_chain.invoke({})
        
        assert "N" in result.upper(), \
            f"Expected evaluator to reject empty response, got: {result}"
    
    def test_partial_quiz_format(self):
        """Test evaluation of a partially formatted quiz."""
        partial_quiz = """
        Question 1: What is Python?
        Question 2: Who created Python?
        """
        eval_chain = create_eval_chain(partial_quiz)
        result = eval_chain.invoke({})
        
        # This tests whether the evaluator catches missing delimiter format
        print(f"Partial quiz evaluation result: {result}")
        # We just verify it returns a result (Y or N)
        assert result.strip().upper() in ["Y", "N"], \
            f"Expected Y or N, got: {result}"


def run_all_evals():
    """Run all model-graded evaluations and print results."""
    print("=" * 60)
    print("Running Model-Graded Evaluations")
    print("=" * 60)
    
    # Test 1: Valid quiz format
    print("\n[Test 1] Evaluating valid quiz format...")
    eval_chain = create_eval_chain(VALID_QUIZ_RESPONSE)
    result = eval_chain.invoke({})
    print(f"Result: {result}")
    print(f"Status: {'PASS' if 'Y' in result.upper() else 'FAIL'}")
    
    # Test 2: Invalid response
    print("\n[Test 2] Evaluating invalid response...")
    eval_chain = create_eval_chain(INVALID_QUIZ_RESPONSE)
    result = eval_chain.invoke({})
    print(f"Result: {result}")
    print(f"Status: {'PASS' if 'N' in result.upper() else 'FAIL'}")
    
    # Test 3: Empty response
    print("\n[Test 3] Evaluating empty response...")
    eval_chain = create_eval_chain("")
    result = eval_chain.invoke({})
    print(f"Result: {result}")
    print(f"Status: {'PASS' if 'N' in result.upper() else 'FAIL'}")
    
    print("\n" + "=" * 60)
    print("Model-Graded Evaluations Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_all_evals()
