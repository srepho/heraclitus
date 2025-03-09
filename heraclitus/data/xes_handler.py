"""
XES (eXtensible Event Stream) format handler for Heraclitus.

This module provides functions to import and export XES files,
which is the IEEE 1849-2016 standard for process mining event logs.

References:
    - IEEE 1849-2016: https://standards.ieee.org/standard/1849-2016.html
    - XES Standard Definition: http://www.xes-standard.org/
"""
from typing import Optional, Dict, Any, List, Union, Set, Tuple
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import datetime
import warnings
import pandas as pd

from heraclitus.data.event_log import EventLog

# XES namespaces
XES_NS = {"xes": "http://www.xes-standard.org/"}

# XES Extension URIs
XES_EXTENSION_CONCEPT = "http://www.xes-standard.org/org/concept.xesext"
XES_EXTENSION_TIME = "http://www.xes-standard.org/org/time.xesext"
XES_EXTENSION_ORGANIZATIONAL = "http://www.xes-standard.org/org/organizational.xesext"
XES_EXTENSION_LIFECYCLE = "http://www.xes-standard.org/org/lifecycle.xesext"
XES_EXTENSION_SEMANTIC = "http://www.xes-standard.org/org/semantic.xesext"


class XESAttribute:
    """Class representing an XES attribute."""
    
    def __init__(self, key: str, value: Any, type_name: str = "string"):
        """
        Initialize an XES attribute.
        
        Args:
            key: Attribute key
            value: Attribute value
            type_name: Type of the attribute (string, date, int, float, boolean)
        """
        self.key = key
        self.value = value
        self.type_name = type_name
    
    def to_xml(self) -> ET.Element:
        """
        Convert the attribute to an XML element.
        
        Returns:
            ElementTree Element representing the attribute
        """
        if self.type_name == "date":
            if isinstance(self.value, str):
                # Try to parse the string as a datetime
                try:
                    dt = datetime.datetime.fromisoformat(self.value.replace('Z', '+00:00'))
                    value_str = dt.isoformat()
                except ValueError:
                    value_str = self.value
            elif isinstance(self.value, (datetime.datetime, datetime.date)):
                value_str = self.value.isoformat()
            else:
                value_str = str(self.value)
            
            attr = ET.Element(f"{self.type_name}", {"key": self.key, "value": value_str})
        else:
            attr = ET.Element(f"{self.type_name}", {"key": self.key, "value": str(self.value)})
        
        return attr


def parse_xes_attribute(element: ET.Element) -> Tuple[str, Any]:
    """
    Parse an XES attribute element.
    
    Args:
        element: ElementTree Element representing an XES attribute
        
    Returns:
        Tuple of (key, value)
    """
    key = element.attrib.get("key")
    value = element.attrib.get("value")
    
    # Convert value based on element tag (type)
    if element.tag == "string":
        pass  # Keep as string
    elif element.tag == "date":
        try:
            value = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            warnings.warn(f"Could not parse date value: {value}")
    elif element.tag == "int":
        value = int(value)
    elif element.tag == "float":
        value = float(value)
    elif element.tag == "boolean":
        value = value.lower() == "true"
    
    return key, value


def import_xes(file_path: str, timestamp_format: Optional[str] = None) -> EventLog:
    """
    Import an XES file and convert it to a Heraclitus EventLog.
    
    Args:
        file_path: Path to the XES file
        timestamp_format: Optional format string for parsing timestamps
        
    Returns:
        Heraclitus EventLog object
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a valid XES file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XES file not found: {file_path}")
    
    # Parse the XML file
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid XES file: {str(e)}")
    
    # Check if it's an XES file
    if root.tag != "log" and not root.tag.endswith("}log"):
        raise ValueError(f"Not an XES file: root element is '{root.tag}' instead of 'log'")
    
    # Extract global attributes and extensions
    extensions = {}
    global_trace_attrs = {}
    global_event_attrs = {}
    
    for child in root:
        if child.tag == "extension" or child.tag.endswith("}extension"):
            name = child.attrib.get("name")
            prefix = child.attrib.get("prefix")
            uri = child.attrib.get("uri")
            extensions[prefix] = {"name": name, "uri": uri}
        
        elif child.tag == "global" or child.tag.endswith("}global"):
            scope = child.attrib.get("scope")
            if scope == "trace":
                for attr in child:
                    key, value = parse_xes_attribute(attr)
                    global_trace_attrs[key] = {"value": value, "type": attr.tag}
            elif scope == "event":
                for attr in child:
                    key, value = parse_xes_attribute(attr)
                    global_event_attrs[key] = {"value": value, "type": attr.tag}
    
    # Initialize default column names
    case_id_key = "concept:name"  # Default case ID attribute
    activity_key = "concept:name"  # Default activity attribute
    timestamp_key = "time:timestamp"  # Default timestamp attribute
    
    # Extract traces and events
    events_data = []
    
    for trace in root.findall(".//trace") + root.findall(".//{http://www.xes-standard.org/}trace"):
        trace_attrs = {}
        
        # First, apply global trace attributes
        for key, attr_info in global_trace_attrs.items():
            trace_attrs[key] = attr_info["value"]
        
        # Then, override with specific trace attributes
        for attr in trace:
            if attr.tag not in ("event", "{http://www.xes-standard.org/}event"):
                key, value = parse_xes_attribute(attr)
                trace_attrs[key] = value
        
        # Extract case ID (default is concept:name)
        case_id = trace_attrs.get(case_id_key, "UNKNOWN")
        
        # Process events
        for event in trace.findall("./event") + trace.findall("./{http://www.xes-standard.org/}event"):
            event_attrs = {}
            
            # First, apply global event attributes
            for key, attr_info in global_event_attrs.items():
                event_attrs[key] = attr_info["value"]
            
            # Then, apply trace attributes (inherit from trace)
            for key, value in trace_attrs.items():
                if key != case_id_key:  # Don't duplicate case ID
                    event_attrs[key] = value
            
            # Add case ID
            event_attrs[case_id_key] = case_id
            
            # Then, override with specific event attributes
            for attr in event:
                if attr.tag not in ("event", "{http://www.xes-standard.org/}event"):
                    key, value = parse_xes_attribute(attr)
                    event_attrs[key] = value
            
            events_data.append(event_attrs)
    
    # Convert to DataFrame
    df = pd.DataFrame(events_data)
    
    # Ensure required columns exist
    if case_id_key not in df.columns:
        raise ValueError(f"Case ID attribute '{case_id_key}' not found in the XES file")
    
    if activity_key not in df.columns:
        raise ValueError(f"Activity attribute '{activity_key}' not found in the XES file")
    
    if timestamp_key not in df.columns:
        warnings.warn(f"Timestamp attribute '{timestamp_key}' not found in the XES file")
    
    # Create EventLog
    return EventLog(
        df,
        case_id_column=case_id_key,
        activity_column=activity_key,
        timestamp_column=timestamp_key if timestamp_key in df.columns else None
    )


def export_xes(event_log: EventLog, file_path: str, pretty_print: bool = True) -> None:
    """
    Export a Heraclitus EventLog to an XES file.
    
    Args:
        event_log: The EventLog to export
        file_path: Path to save the XES file
        pretty_print: Whether to format the XML with indentation
        
    Raises:
        ValueError: If the EventLog is invalid or missing required columns
    """
    # Check if the EventLog has required columns
    df = event_log.to_dataframe()
    required_columns = [
        event_log.case_id_column,
        event_log.activity_column
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the EventLog")
    
    # Create root element
    root = ET.Element("log")
    root.set("xes.version", "1.0")
    root.set("xmlns", "http://www.xes-standard.org/")
    
    # Add standard extensions
    extensions = [
        {"name": "Concept", "prefix": "concept", "uri": XES_EXTENSION_CONCEPT},
        {"name": "Time", "prefix": "time", "uri": XES_EXTENSION_TIME},
        {"name": "Organizational", "prefix": "org", "uri": XES_EXTENSION_ORGANIZATIONAL},
        {"name": "Lifecycle", "prefix": "lifecycle", "uri": XES_EXTENSION_LIFECYCLE}
    ]
    
    for ext in extensions:
        ext_elem = ET.SubElement(root, "extension")
        ext_elem.set("name", ext["name"])
        ext_elem.set("prefix", ext["prefix"])
        ext_elem.set("uri", ext["uri"])
    
    # Group by case ID to create traces
    for case_id, case_df in df.groupby(event_log.case_id_column):
        trace = ET.SubElement(root, "trace")
        
        # Add trace attributes (case ID)
        trace_id = XESAttribute("concept:name", case_id)
        trace.append(trace_id.to_xml())
        
        # Add events
        for _, row in case_df.iterrows():
            event = ET.SubElement(trace, "event")
            
            # Add mandatory attributes
            activity = XESAttribute("concept:name", row[event_log.activity_column])
            event.append(activity.to_xml())
            
            # Add timestamp if available
            if event_log.timestamp_column and event_log.timestamp_column in row:
                timestamp = XESAttribute("time:timestamp", row[event_log.timestamp_column], "date")
                event.append(timestamp.to_xml())
            
            # Add other attributes
            for col in row.index:
                # Skip already processed columns
                if col in (event_log.case_id_column, event_log.activity_column, event_log.timestamp_column):
                    continue
                
                value = row[col]
                
                # Skip NaN/None values
                if pd.isna(value):
                    continue
                
                # Determine attribute type
                if isinstance(value, (int, np.integer)):
                    attr = XESAttribute(col, value, "int")
                elif isinstance(value, (float, np.floating)):
                    attr = XESAttribute(col, value, "float")
                elif isinstance(value, bool):
                    attr = XESAttribute(col, value, "boolean")
                elif isinstance(value, (datetime.datetime, datetime.date)):
                    attr = XESAttribute(col, value, "date")
                else:
                    attr = XESAttribute(col, value, "string")
                
                event.append(attr.to_xml())
    
    # Write to file
    tree = ET.ElementTree(root)
    
    if pretty_print:
        xml_string = ET.tostring(root, encoding="utf-8")
        pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)
    else:
        tree.write(file_path, encoding="utf-8", xml_declaration=True)


# Add EventLog extension methods
def extend_event_log_class():
    """
    Extend the EventLog class with methods for XES import/export.
    This function is called when the module is imported.
    """
    # Add from_xes class method
    if not hasattr(EventLog, "from_xes"):
        @classmethod
        def from_xes(cls, file_path: str, **kwargs) -> EventLog:
            """
            Create an EventLog from an XES file.
            
            Args:
                file_path: Path to the XES file
                **kwargs: Additional arguments to pass to import_xes
                
            Returns:
                EventLog object
            """
            return import_xes(file_path, **kwargs)
        
        EventLog.from_xes = from_xes
    
    # Add to_xes instance method
    if not hasattr(EventLog, "to_xes"):
        def to_xes(self, file_path: str, pretty_print: bool = True) -> None:
            """
            Export this EventLog to an XES file.
            
            Args:
                file_path: Path to save the XES file
                pretty_print: Whether to format the XML with indentation
            """
            export_xes(self, file_path, pretty_print)
        
        EventLog.to_xes = to_xes


# Extend EventLog class when this module is imported
extend_event_log_class()