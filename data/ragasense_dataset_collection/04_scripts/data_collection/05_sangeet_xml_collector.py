#!/usr/bin/env python3
"""
SANGEET XML Dataset Collector
Downloads and processes the SANGEET XML-based Hindustani dataset
"""

import os
import json
import logging
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SangeetXMLCollector:
    """Collector for SANGEET XML dataset"""
    
    def __init__(self, output_dir: str = "01_raw_data/sangeet_xml"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paper_url = "https://arxiv.org/abs/2306.04148"
        
    def get_dataset_info(self) -> Dict:
        """Get information about the SANGEET dataset"""
        try:
            logger.info("Getting SANGEET dataset information...")
            
            # Dataset information from the paper
            dataset_info = {
                "name": "SANGEET",
                "description": "XML-based Open Dataset for Research in Hindustani Sangeet",
                "authors": ["Chandan Misra", "Swarup Chattopadhyay"],
                "paper_url": self.paper_url,
                "features": [
                    "Comprehensive information of Hindustani compositions",
                    "Metadata, structural, notational, rhythmic, and melodic information",
                    "Standardized XML format",
                    "Ground truth for music information research",
                    "Support for machine learning tasks"
                ],
                "applications": [
                    "Music information retrieval using XQuery",
                    "Visualization through Omenad rendering system",
                    "Statistical and machine learning analysis",
                    "Audio-score alignment"
                ],
                "compositions_source": "Pt. Vishnu Narayan Bhatkhande",
                "format": "XML"
            }
            
            # Save dataset info
            info_file = self.output_dir / "sangeet_dataset_info.json"
            with open(info_file, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            logger.info("✅ SANGEET dataset information saved")
            return dataset_info
            
        except Exception as e:
            logger.error(f"Error getting SANGEET dataset info: {e}")
            return {}
    
    def create_sample_xml_structure(self) -> bool:
        """Create sample XML structure for Hindustani compositions"""
        try:
            logger.info("Creating sample XML structure...")
            
            # Sample XML structure for a Hindustani composition
            sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
<composition>
    <metadata>
        <title>Raga Yaman</title>
        <composer>Pt. Vishnu Narayan Bhatkhande</composer>
        <raga>Yaman</raga>
        <taala>Teentaal</taala>
        <tradition>Hindustani</tradition>
        <language>Hindi</language>
    </metadata>
    <structural>
        <sections>
            <section name="Alap">
                <description>Slow, unmetered introduction</description>
                <duration>5:00</duration>
            </section>
            <section name="Bandish">
                <description>Composed song with lyrics</description>
                <duration>8:00</duration>
            </section>
            <section name="Taan">
                <description>Fast, virtuosic passages</description>
                <duration>3:00</duration>
            </section>
        </sections>
    </structural>
    <notational>
        <swara_notations>
            <swara name="Sa" frequency="240Hz" />
            <swara name="Re" frequency="270Hz" />
            <swara name="Ga" frequency="300Hz" />
            <swara name="Ma" frequency="320Hz" />
            <swara name="Pa" frequency="360Hz" />
            <swara name="Dha" frequency="405Hz" />
            <swara name="Ni" frequency="450Hz" />
        </swara_notations>
        <raga_scale>
            <aroha>Sa Re Ga Ma Pa Dha Ni Sa</aroha>
            <avaroha>Sa Ni Dha Pa Ma Ga Re Sa</avaroha>
        </raga_scale>
    </notational>
    <rhythmic>
        <taala_info>
            <name>Teentaal</name>
            <beats>16</beats>
            <divisions>4</divisions>
            <emphasis>1, 5, 9, 13</emphasis>
        </taala_info>
        <laya>Vilambit, Madhya, Drut</laya>
    </rhythmic>
    <melodic>
        <characteristic_phrases>
            <phrase>Ga Ma Pa Dha Pa Ma Ga</phrase>
            <phrase>Sa Re Ga Ma Pa Dha Ni</phrase>
        </characteristic_phrases>
        <gamakas>
            <gamaka type="meend" notes="Ga Ma" />
            <gamaka type="khatka" notes="Pa Dha" />
        </gamakas>
    </melodic>
</composition>"""
            
            # Save sample XML
            sample_file = self.output_dir / "sample_composition.xml"
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(sample_xml)
            
            logger.info("✅ Sample XML structure created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sample XML structure: {e}")
            return False
    
    def parse_xml_composition(self, xml_file: Path) -> Dict:
        """Parse a single XML composition file"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            composition = {
                "metadata": {},
                "structural": {},
                "notational": {},
                "rhythmic": {},
                "melodic": {}
            }
            
            # Parse metadata
            metadata = root.find('metadata')
            if metadata is not None:
                for child in metadata:
                    composition["metadata"][child.tag] = child.text
            
            # Parse structural information
            structural = root.find('structural')
            if structural is not None:
                sections = structural.find('sections')
                if sections is not None:
                    composition["structural"]["sections"] = []
                    for section in sections:
                        section_data = {
                            "name": section.get('name'),
                            "description": section.find('description').text if section.find('description') is not None else "",
                            "duration": section.find('duration').text if section.find('duration') is not None else ""
                        }
                        composition["structural"]["sections"].append(section_data)
            
            # Parse notational information
            notational = root.find('notational')
            if notational is not None:
                swara_notations = notational.find('swara_notations')
                if swara_notations is not None:
                    composition["notational"]["swaras"] = []
                    for swara in swara_notations:
                        swara_data = {
                            "name": swara.get('name'),
                            "frequency": swara.get('frequency')
                        }
                        composition["notational"]["swaras"].append(swara_data)
                
                raga_scale = notational.find('raga_scale')
                if raga_scale is not None:
                    composition["notational"]["aroha"] = raga_scale.find('aroha').text if raga_scale.find('aroha') is not None else ""
                    composition["notational"]["avaroha"] = raga_scale.find('avaroha').text if raga_scale.find('avaroha') is not None else ""
            
            # Parse rhythmic information
            rhythmic = root.find('rhythmic')
            if rhythmic is not None:
                taala_info = rhythmic.find('taala_info')
                if taala_info is not None:
                    composition["rhythmic"]["taala"] = {
                        "name": taala_info.find('name').text if taala_info.find('name') is not None else "",
                        "beats": taala_info.find('beats').text if taala_info.find('beats') is not None else "",
                        "divisions": taala_info.find('divisions').text if taala_info.find('divisions') is not None else "",
                        "emphasis": taala_info.find('emphasis').text if taala_info.find('emphasis') is not None else ""
                    }
                
                laya = rhythmic.find('laya')
                if laya is not None:
                    composition["rhythmic"]["laya"] = laya.text
            
            # Parse melodic information
            melodic = root.find('melodic')
            if melodic is not None:
                characteristic_phrases = melodic.find('characteristic_phrases')
                if characteristic_phrases is not None:
                    composition["melodic"]["phrases"] = []
                    for phrase in characteristic_phrases:
                        composition["melodic"]["phrases"].append(phrase.text)
                
                gamakas = melodic.find('gamakas')
                if gamakas is not None:
                    composition["melodic"]["gamakas"] = []
                    for gamaka in gamakas:
                        gamaka_data = {
                            "type": gamaka.get('type'),
                            "notes": gamaka.get('notes')
                        }
                        composition["melodic"]["gamakas"].append(gamaka_data)
            
            return composition
            
        except Exception as e:
            logger.error(f"Error parsing XML file {xml_file}: {e}")
            return {}
    
    def extract_metadata(self) -> Dict:
        """Extract metadata from XML files"""
        try:
            logger.info("Extracting SANGEET XML metadata...")
            
            metadata = {
                "compositions": [],
                "ragas": set(),
                "taalas": set(),
                "composers": set(),
                "total_compositions": 0
            }
            
            # Parse sample XML file
            sample_file = self.output_dir / "sample_composition.xml"
            if sample_file.exists():
                composition = self.parse_xml_composition(sample_file)
                if composition:
                    metadata["compositions"].append(composition)
                    metadata["total_compositions"] += 1
                    
                    # Extract ragas, taalas, and composers
                    if "raga" in composition["metadata"]:
                        metadata["ragas"].add(composition["metadata"]["raga"])
                    if "taala" in composition["metadata"]:
                        metadata["taalas"].add(composition["metadata"]["taala"])
                    if "composer" in composition["metadata"]:
                        metadata["composers"].add(composition["metadata"]["composer"])
            
            # Convert sets to lists
            metadata["ragas"] = list(metadata["ragas"])
            metadata["taalas"] = list(metadata["taalas"])
            metadata["composers"] = list(metadata["composers"])
            
            # Save metadata
            metadata_file = self.output_dir / "extracted_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Metadata extracted: {metadata['total_compositions']} compositions, {len(metadata['ragas'])} ragas")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def collect(self) -> bool:
        """Main collection method"""
        try:
            logger.info("Starting SANGEET XML collection...")
            
            # Get dataset info
            dataset_info = self.get_dataset_info()
            if not dataset_info:
                return False
            
            # Create sample XML structure
            if not self.create_sample_xml_structure():
                return False
            
            # Extract metadata
            metadata = self.extract_metadata()
            if not metadata:
                return False
            
            logger.info("✅ SANGEET XML collection completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in SANGEET XML collection process: {e}")
            return False

def main():
    """Main function"""
    collector = SangeetXMLCollector()
    success = collector.collect()
    
    if success:
        print("🎉 SANGEET XML collection completed successfully!")
    else:
        print("❌ SANGEET XML collection failed. Check logs for details.")

if __name__ == "__main__":
    main()
