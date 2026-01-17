"""
Lesson 4: Comprehensive Testing Framework

This module demonstrates comprehensive testing with hallucination detection
and dataset-based evaluations using LLM graders.
"""

import warnings
import os
import pytest
import pandas as pd
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
_ = load_dotenv(find_dotenv())

# Import quiz_bank from app.py
from app import quiz_bank, assistant_chain


# ============================================================================
# Evaluation Chain for Hallucination Detection
# ============================================================================

def create_hallucination_eval_chain(context, agent_response):
    """
    Create an evaluation chain that checks if a quiz contains only facts
    from the question bank (no hallucinations).
    """
    eval_system_prompt = """You are an assistant that evaluates \
    how well the quiz assistant creates quizzes for a user by looking at the set of \
    facts available to the assistant.
    Your primary concern is making sure that ONLY facts \
    available are used. Quizzes that contain facts outside
    the question bank are BAD quizzes and harmful to the student."""

    eval_user_message = """You are evaluating a generated quiz \
    based on the context that the assistant uses to create the quiz.
    Here is the data:
        [BEGIN DATA]
        ************
        [Question Bank]: {context}
        ************
        [Quiz]: {agent_response}
        ************
        [END DATA]

Compare the content of the submission with the question bank \
using the following steps

1. Review the question bank carefully. \
    These are the only facts the quiz can reference
2. Compare the quiz to the question bank.
3. Ignore differences in grammar or punctuation
4. If a fact is in the quiz, but not in the question bank \
    the quiz is bad.

Remember, the quizzes need to only include facts the assistant \
    is aware of. It is dangerous to allow made up facts.

Output Y if the quiz only contains facts from the question bank, \
output N if it contains facts that are not in the question bank.
"""
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", eval_system_prompt),
        ("human", eval_user_message),
    ])

    return eval_prompt | ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0) | StrOutputParser()


# ============================================================================
# Detailed Evaluation Chain with Explanations
# ============================================================================

def create_detailed_eval_chain():
    """
    Create an evaluation chain that provides both a decision and explanation
    for whether a quiz only uses facts from the question bank.
    """
    eval_system_prompt = """You are an assistant that evaluates \
    how well the quiz assistant creates quizzes for a user by looking at the set of \
    facts available to the assistant.
    Your primary concern is making sure that ONLY facts \
    available are used.
    Helpful quizzes only contain facts in the test set."""

    eval_user_message = """You are evaluating a generated quiz based on the question bank that the assistant uses to create the quiz.
    Here is the data:
        [BEGIN DATA]
        ************
        [Question Bank]: {context}
        ************
        [Quiz]: {agent_response}
        ************
        [END DATA]

## Examples of quiz questions
Subject: <subject>
    Categories: <category1>, <category2>
    Facts:
    - <fact 1>
    - <fact 2>

## Steps to make a decision
Compare the content of the submission with the question bank using the following steps

1. Review the question bank carefully. These are the only facts the quiz can reference
2. Compare the information in the quiz to the question bank.
3. Ignore differences in grammar or punctuation

Remember, the quizzes should only include information from the question bank.


## Additional rules
- Output an explanation of whether the quiz only references information in the context.
- Make the explanation brief only include a summary of your reasoning for the decsion.
- Include a clear "Yes" or "No" as the first paragraph.
- Reference facts from the quiz bank if the answer is yes

Separate the decision and the explanation. For example:

************
Decision: <Y>
************
Explanation: <Explanation>
************
"""
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", eval_system_prompt),
        ("human", eval_user_message),
    ])

    return eval_prompt | ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0) | StrOutputParser()


# ============================================================================
# Test Dataset
# ============================================================================

TEST_DATASET = [
    {
        "input": "I'm trying to learn about science, can you give me a quiz to test my knowledge",
        "response": "science",
        "subjects": ["davinci", "telescope", "physics", "curie"]
    },
    {
        "input": "I'm an geography expert, give a quiz to prove it?",
        "response": "geography",
        "subjects": ["paris", "france", "louvre"]
    },
    {
        "input": "Quiz me about Art",
        "response": "art",
        "subjects": ["mona lisa", "starry night", "davinci", "van gogh"]
    },
]


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_dataset(dataset, quiz_bank, assistant, evaluator):
    """
    Evaluate a dataset of test cases using the quiz assistant and evaluator.
    
    Returns a list of evaluation results with input, output, and grader response.
    """
    eval_results = []
    for row in dataset:
        eval_result = {}
        user_input = row["input"]
        answer = assistant.invoke({"question": user_input})
        eval_response = evaluator.invoke({"context": quiz_bank, "agent_response": answer})

        eval_result["input"] = user_input
        eval_result["output"] = answer
        eval_result["grader_response"] = eval_response
        eval_results.append(eval_result)
    return eval_results


def generate_report(eval_results, output_path=None):
    """
    Generate an HTML report from evaluation results.
    
    Args:
        eval_results: List of evaluation result dictionaries
        output_path: Optional path to save HTML report
    
    Returns:
        HTML string of the report
    """
    df = pd.DataFrame(eval_results)
    df_html = df.to_html().replace("\\n", "<br>")
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Test Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Comprehensive Test Evaluation Report</h1>
    {df_html}
</body>
</html>
""")
    return df_html


# ============================================================================
# Pytest Test Classes
# ============================================================================

class TestHallucinationDetection:
    """Test suite for hallucination detection in quiz generation."""

    def test_hallucination_detection_unknown_topic(self):
        """
        Test that asking about a topic not in the quiz bank
        is flagged as hallucination (returns N).
        """
        assistant = assistant_chain()
        quiz_request = "Write me a quiz about books."
        result = assistant.invoke({"question": quiz_request})
        print(f"\nQuiz generated for 'books' request:\n{result}\n")
        
        eval_agent = create_hallucination_eval_chain(quiz_bank, result)
        eval_response = eval_agent.invoke({"context": quiz_bank, "agent_response": result})
        print(f"Evaluator response: {eval_response}")
        
        # Should detect that books is not in the quiz bank
        assert "N" in eval_response.upper(), \
            f"Expected hallucination detection (N) for unknown topic 'books', got: {eval_response}"

    def test_valid_science_quiz_no_hallucination(self):
        """
        Test that a valid science quiz is not flagged as hallucination.
        """
        assistant = assistant_chain()
        quiz_request = "Give me a quiz about science"
        result = assistant.invoke({"question": quiz_request})
        print(f"\nQuiz generated for 'science' request:\n{result}\n")
        
        eval_agent = create_hallucination_eval_chain(quiz_bank, result)
        eval_response = eval_agent.invoke({"context": quiz_bank, "agent_response": result})
        print(f"Evaluator response: {eval_response}")
        
        # Science is in the quiz bank, should pass
        assert "Y" in eval_response.upper(), \
            f"Expected valid quiz (Y) for science topic, got: {eval_response}"

    def test_valid_art_quiz_no_hallucination(self):
        """
        Test that a valid art quiz is not flagged as hallucination.
        """
        assistant = assistant_chain()
        quiz_request = "Create a quiz about art"
        result = assistant.invoke({"question": quiz_request})
        print(f"\nQuiz generated for 'art' request:\n{result}\n")
        
        eval_agent = create_hallucination_eval_chain(quiz_bank, result)
        eval_response = eval_agent.invoke({"context": quiz_bank, "agent_response": result})
        print(f"Evaluator response: {eval_response}")
        
        # Art is in the quiz bank, should pass
        assert "Y" in eval_response.upper(), \
            f"Expected valid quiz (Y) for art topic, got: {eval_response}"


class TestDatasetEvaluations:
    """Test suite for dataset-based evaluations."""

    def test_dataset_evaluation_runs_successfully(self):
        """
        Test that dataset evaluation completes without errors.
        """
        assistant = assistant_chain()
        evaluator = create_detailed_eval_chain()
        
        eval_results = evaluate_dataset(TEST_DATASET, quiz_bank, assistant, evaluator)
        
        assert len(eval_results) == len(TEST_DATASET), \
            f"Expected {len(TEST_DATASET)} results, got {len(eval_results)}"
        
        for result in eval_results:
            assert "input" in result
            assert "output" in result
            assert "grader_response" in result
            print(f"\nInput: {result['input']}")
            print(f"Grader Response: {result['grader_response'][:200]}...")

    def test_generate_report(self):
        """
        Test that report generation works correctly.
        """
        assistant = assistant_chain()
        evaluator = create_detailed_eval_chain()
        
        eval_results = evaluate_dataset(TEST_DATASET[:1], quiz_bank, assistant, evaluator)
        html_report = generate_report(eval_results)
        
        assert "<table" in html_report.lower(), "Report should contain HTML table"
        assert "input" in html_report.lower(), "Report should contain input column"


# ============================================================================
# Standalone Execution
# ============================================================================

def run_comprehensive_tests():
    """Run all comprehensive tests and generate a report."""
    print("=" * 70)
    print("Running Comprehensive Testing Framework")
    print("=" * 70)

    # Test 1: Hallucination Detection
    print("\n[Test 1] Hallucination Detection - Unknown Topic (books)")
    print("-" * 50)
    assistant = assistant_chain()
    quiz_request = "Write me a quiz about books."
    result = assistant.invoke({"question": quiz_request})
    print(f"Generated Quiz:\n{result}\n")
    
    eval_agent = create_hallucination_eval_chain(quiz_bank, result)
    eval_response = eval_agent.invoke({"context": quiz_bank, "agent_response": result})
    print(f"Evaluator Response: {eval_response}")
    print(f"Status: {'PASS' if 'N' in eval_response.upper() else 'FAIL'}")

    # Test 2: Valid Quiz Check
    print("\n[Test 2] Valid Science Quiz - No Hallucination Expected")
    print("-" * 50)
    quiz_request = "Give me a quiz about science"
    result = assistant.invoke({"question": quiz_request})
    print(f"Generated Quiz:\n{result}\n")
    
    eval_agent = create_hallucination_eval_chain(quiz_bank, result)
    eval_response = eval_agent.invoke({"context": quiz_bank, "agent_response": result})
    print(f"Evaluator Response: {eval_response}")
    print(f"Status: {'PASS' if 'Y' in eval_response.upper() else 'FAIL'}")

    # Test 3: Dataset Evaluation
    print("\n[Test 3] Dataset Evaluation with Detailed Explanations")
    print("-" * 50)
    evaluator = create_detailed_eval_chain()
    eval_results = evaluate_dataset(TEST_DATASET, quiz_bank, assistant, evaluator)
    
    for i, result in enumerate(eval_results, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {result['input']}")
        print(f"Grader Response:\n{result['grader_response'][:300]}...")

    # Generate Report
    print("\n[Test 4] Generating HTML Report")
    print("-" * 50)
    report_path = "/tmp/comprehensive_eval_results.html"
    generate_report(eval_results, report_path)
    print(f"Report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("Comprehensive Testing Complete")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_tests()
