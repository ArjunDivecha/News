"""
Parse structured JSON output from LLM into report data structure.
"""

import json
import re
from typing import Dict, List, Any, Optional


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM response, handling markdown code blocks and extra text.
    
    Args:
        text: Raw LLM response
        
    Returns:
        Parsed JSON dict, or None if extraction fails
    """
    # Try to find JSON in markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common issues
        json_str = json_str.strip()
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


def validate_report_structure(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate that the report structure matches expected format.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check top-level keys
    required_keys = ['executive_synthesis', 'flash_headlines', 'sections']
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    # Validate executive synthesis
    if 'executive_synthesis' in data:
        es = data['executive_synthesis']
        if 'single_most_important' not in es:
            errors.append("executive_synthesis missing 'single_most_important'")
        if 'key_takeaways' not in es or not isinstance(es['key_takeaways'], list):
            errors.append("executive_synthesis missing 'key_takeaways' list")
        if 'what_to_watch' not in es or not isinstance(es['what_to_watch'], list):
            errors.append("executive_synthesis missing 'what_to_watch' list")
    
    # Validate sections
    if 'sections' in data:
        if not isinstance(data['sections'], list):
            errors.append("'sections' must be a list")
        else:
            for i, section in enumerate(data['sections']):
                if not isinstance(section, dict):
                    errors.append(f"Section {i} is not a dict")
                    continue
                if 'title' not in section:
                    errors.append(f"Section {i} missing 'title'")
                if 'narrative' not in section:
                    errors.append(f"Section {i} missing 'narrative'")
                if 'tables' in section:
                    if not isinstance(section['tables'], list):
                        errors.append(f"Section {i} 'tables' must be a list")
                    else:
                        for j, table in enumerate(section['tables']):
                            if not isinstance(table, dict):
                                errors.append(f"Section {i}, Table {j} is not a dict")
                                continue
                            if 'headers' not in table:
                                errors.append(f"Section {i}, Table {j} missing 'headers'")
                            if 'rows' not in table:
                                errors.append(f"Section {i}, Table {j} missing 'rows'")
    
    return len(errors) == 0, errors


def parse_structured_report(llm_response: str) -> Dict[str, Any]:
    """
    Parse LLM response into structured report data.
    
    Args:
        llm_response: Raw LLM response text
        
    Returns:
        Parsed report structure dict
    """
    data = extract_json_from_response(llm_response)
    
    if data is None:
        raise ValueError("Could not extract JSON from LLM response")
    
    is_valid, errors = validate_report_structure(data)
    if not is_valid:
        raise ValueError(f"Invalid report structure: {', '.join(errors)}")
    
    return data
