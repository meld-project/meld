"""CAPE JSON format adapter for Nebula baselines.

This module provides functions to convert CAPE JSON reports to formats
compatible with Nebula baseline models (Neurlux, QuoVadis, DMDS).
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Union

LOGGER = logging.getLogger(__name__)


def extract_api_sequence_from_cape(report: Union[Dict, str]) -> List[str]:
    """Extract API call sequence from CAPE JSON report.
    
    Args:
        report: CAPE JSON report (dict or file path)
        
    Returns:
        List of API names (lowercased)
    """
    if isinstance(report, str):
        try:
            with open(report, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except Exception as e:
            LOGGER.warning(f"Failed to load JSON from {report}: {e}")
            return []
    
    if not isinstance(report, dict):
        return []
    
    api_sequence = []
    
    # CAPE format: behavior.processes[].calls[].api
    if 'behavior' in report and isinstance(report['behavior'], dict):
        processes = report['behavior'].get('processes', [])
        if isinstance(processes, list):
            for process in processes:
                if isinstance(process, dict):
                    calls = process.get('calls', [])
                    if isinstance(calls, list):
                        for call in calls:
                            if isinstance(call, dict):
                                api_name = call.get('api')
                                if api_name:
                                    api_sequence.append(api_name.lower())
    
    return api_sequence


def parse_api_calls_from_cape(report: Union[Dict, str]) -> List[Dict]:
    """Parse API calls with arguments from CAPE JSON report.
    
    Args:
        report: CAPE JSON report (dict or file path)
        
    Returns:
        List of dicts with 'api_name' and 'args' keys
    """
    if isinstance(report, str):
        try:
            with open(report, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except Exception as e:
            LOGGER.warning(f"Failed to load JSON from {report}: {e}")
            return []
    
    if not isinstance(report, dict):
        return []
    
    api_calls = []
    
    # CAPE format: behavior.processes[].calls[].api and .arguments
    if 'behavior' in report and isinstance(report['behavior'], dict):
        processes = report['behavior'].get('processes', [])
        if isinstance(processes, list):
            for process in processes:
                if isinstance(process, dict):
                    calls = process.get('calls', [])
                    if isinstance(calls, list):
                        for call in calls:
                            if isinstance(call, dict):
                                api_name = call.get('api')
                                arguments = call.get('arguments', [])
                                
                                if api_name:
                                    # Convert arguments to list of strings
                                    args_list = []
                                    if isinstance(arguments, list):
                                        for arg in arguments:
                                            if isinstance(arg, (str, int, float)):
                                                args_list.append(str(arg))
                                            elif isinstance(arg, dict):
                                                # Flatten dict arguments
                                                args_list.append(json.dumps(arg, sort_keys=True))
                                            else:
                                                args_list.append(str(arg))
                                    elif isinstance(arguments, dict):
                                        args_list.append(json.dumps(arguments, sort_keys=True))
                                    else:
                                        args_list.append(str(arguments))
                                    
                                    api_calls.append({
                                        'api_name': api_name.lower(),
                                        'args': args_list
                                    })
    
    return api_calls


def cape_to_speakeasy_format(report: Union[Dict, str]) -> Optional[Dict]:
    """Convert CAPE JSON to Speakeasy-like format.
    
    This creates a format compatible with nebula's original preprocessing.
    
    Args:
        report: CAPE JSON report (dict or file path)
        
    Returns:
        Speakeasy-like format dict with 'entry_points' containing 'apis'
    """
    if isinstance(report, str):
        try:
            with open(report, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except Exception as e:
            LOGGER.warning(f"Failed to load JSON from {report}: {e}")
            return None
    
    if not isinstance(report, dict):
        return None
    
    # Extract API calls
    api_calls = parse_api_calls_from_cape(report)
    
    if not api_calls:
        return None
    
    # Convert to Speakeasy format: entry_points[].apis[]
    speakeasy_format = {
        'entry_points': [{
            'apis': [
                {
                    'api_name': call['api_name'],
                    'args': call['args']
                }
                for call in api_calls
            ]
        }]
    }
    
    return speakeasy_format


def cape_json_to_text(report: Union[Dict, str]) -> str:
    """Convert CAPE JSON report to text string.
    
    This is used by Neurlux which processes JSON as text.
    
    Args:
        report: CAPE JSON report (dict or file path)
        
    Returns:
        JSON string representation
    """
    if isinstance(report, str):
        try:
            with open(report, 'r', encoding='utf-8') as f:
                report = json.load(f)
        except Exception as e:
            LOGGER.warning(f"Failed to load JSON from {report}: {e}")
            return "{}"
    
    if isinstance(report, dict):
        return json.dumps(report, ensure_ascii=False, sort_keys=True)
    
    return str(report)

