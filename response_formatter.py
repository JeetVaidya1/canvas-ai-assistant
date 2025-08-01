# response_formatter.py - Clean, natural conversational formatting

import re

def format_response_clean(response: str, question_type: str = "explanation") -> str:
    """
    Clean, natural response formatting like ChatGPT/Claude
    """
    # Step 1: Remove excessive formatting and emojis
    response = clean_excessive_formatting(response)
    
    # Step 2: Fix structure and flow
    response = improve_natural_flow(response)
    
    # Step 3: Clean up redundant sections
    response = remove_redundant_sections(response)
    
    return response.strip()

def clean_excessive_formatting(response: str) -> str:
    """Remove excessive emojis, headers, and formatting"""
    
    # Remove emoji-heavy headers like "💡 **Explanation Guide**" 
    response = re.sub(r'^[🎯💡📚🔍⚖️🛠️🧠]\s*\*\*[^*]+\*\*\s*\n+', '', response)
    
    # Remove excessive emoji bullet points - keep the bullet, remove emoji
    response = re.sub(r'^\s*•\s*\*\*([^*]+)\*\*:\s*([🎯💡📚🔍⚖️🛠️🧠💫📋❓⚠️📝])\s*', r'**\1:** ', response, flags=re.MULTILINE)
    
    # Clean up numbered sections with emojis and excessive formatting
    response = re.sub(r'^###\s*\d+\.\s*([^\n]+)\n+', r'## \1\n\n', response, flags=re.MULTILINE)
    
    # Remove redundant "Step-by-Step Breakdown" headers
    response = re.sub(r'^###?\s*\d*\.?\s*Step-by-Step Breakdown:?\s*\n+', '', response, flags=re.MULTILINE)
    
    # Clean up excessive asterisks and formatting
    response = re.sub(r'\*\*([^*]+)\*\*:\s*•', r'**\1:**', response)
    
    return response

def improve_natural_flow(response: str) -> str:
    """Make the response flow more naturally"""
    
    # Convert clunky headers to natural transitions
    response = re.sub(r'^###?\s*\d*\.?\s*Underlying Principles:?\s*\n+', '\n## Key Principles\n\n', response, flags=re.MULTILINE)
    response = re.sub(r'^###?\s*\d*\.?\s*Real-World Analogies:?\s*\n+', '\n## Think of it this way\n\n', response, flags=re.MULTILINE)
    response = re.sub(r'^###?\s*\d*\.?\s*Reference to Diagrams:?\s*\n+', '\nBased on your course materials, ', response, flags=re.MULTILINE)
    
    # Make bullet points more natural
    response = re.sub(r'^\s*•\s*\*\*What is it\?\*\*:\s*', '', response, flags=re.MULTILINE)
    response = re.sub(r'^\s*•\s*\*\*([^*]+)\*\*:\s*', r'**\1:** ', response, flags=re.MULTILINE)
    
    # Fix awkward double colons
    response = re.sub(r'(\*\*[^*]+\*\*):\s*:', r'\1:', response)
    
    return response

def remove_redundant_sections(response: str) -> str:
    """Remove redundant and repetitive sections"""
    
    # Remove redundant summary if it just repeats what was already said
    summary_pattern = r'\n+###?\s*Summary:?\s*\n+In summary,.*?(?=\n\n|\n###|\n\*\*|$)'
    response = re.sub(summary_pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove generic follow-up questions section
    followup_pattern = r'\n+###?\s*Follow[-\s]?up Questions:?\s*\n+.*?(?=\n\n[A-Z]|$)'
    response = re.sub(followup_pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove "Feel free to ask" conclusions
    response = re.sub(r'\n+Feel free to ask.*?$', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up excessive whitespace
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
    
    return response

def format_response_natural(response: str, question_type: str = "explanation") -> str:
    """
    Main formatting function - makes responses natural and conversational
    """
    # Apply cleaning
    cleaned = format_response_clean(response, question_type)
    
    # Add minimal, natural improvements
    cleaned = add_natural_improvements(cleaned)
    
    return cleaned

def add_natural_improvements(response: str) -> str:
    """Add minimal, natural improvements without clutter"""
    
    # Only add very subtle improvements
    # Convert awkward "Think of components as..." to more natural language
    response = re.sub(r'Think of ([^.]+) as ([^.]+)\.', r'\1 are essentially \2.', response)
    
    # Make examples flow more naturally
    response = re.sub(r'For example, in ([^,]+), ([^.]+)\.', r'For instance, consider \1 - \2.', response)
    
    # Clean up any remaining awkward transitions
    response = re.sub(r'\n\n\*\*([^*]+)\*\*\n\n', r'\n\n**\1**\n\n', response)
    
    return response

# Main function to use in your system
def format_ai_response(response: str, question_type: str = "explanation") -> str:
    """
    Clean, natural response formatting that looks like ChatGPT/Claude
    """
    return format_response_natural(response, question_type)