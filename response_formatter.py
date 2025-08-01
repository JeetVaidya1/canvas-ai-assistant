# response_formatter.py - Phase 1: Basic formatting improvements

import re

def format_response_basic(response: str, question_type: str = "explanation") -> str:
    """
    Basic response formatting to improve readability
    """
    # Step 1: Clean up whitespace
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
    response = re.sub(r'[ \t]+', ' ', response)
    
    # Step 2: Add emojis for different sections
    section_replacements = [
        (r'\*\*Summary:\*\*', '📋 **Summary:**'),
        (r'\*\*Key Points:\*\*', '🎯 **Key Points:**'),
        (r'\*\*Examples:\*\*', '💫 **Examples:**'),
        (r'\*\*Follow-up Questions:\*\*', '❓ **Follow-up Questions:**'),
        (r'\*\*Important:\*\*', '⚠️ **Important:**'),
        (r'\*\*Note:\*\*', '📝 **Note:**'),
    ]
    
    for pattern, replacement in section_replacements:
        response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
    
    # Step 3: Improve numbered lists
    response = re.sub(r'^(\d+)\.\s*\*\*([^*]+)\*\*:?\s*', r'### \1. \2\n\n', response, flags=re.MULTILINE)
    
    # Step 4: Improve bullet points
    response = re.sub(r'^-\s*\*\*([^*]+)\*\*:?\s*', r'• **\1:** ', response, flags=re.MULTILINE)
    response = re.sub(r'^-\s*([^*\n]+)', r'• \1', response, flags=re.MULTILINE)
    
    # Step 5: Add content type indicator
    type_indicators = {
        "definition": "📚",
        "explanation": "💡", 
        "example": "🔍",
        "comparison": "⚖️",
        "application": "🛠️"
    }
    
    if question_type in type_indicators:
        icon = type_indicators[question_type]
        response = f"{icon} **{question_type.title()} Guide**\n\n{response}"
    
    # Step 6: Add simple study tip
    if "follow-up" not in response.lower():
        response += f"""

💡 **Quick Study Tip:** Try explaining this concept to someone else in your own words!
"""
    
    return response.strip()