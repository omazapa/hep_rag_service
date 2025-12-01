#!/usr/bin/env python3
"""
Test script to verify that methods, attributes and parameters are being extracted from Doxygen
"""

from elasticsearch import Elasticsearch
import json

def test_method_extraction():
    """Test that Doxygen methods and attributes are being extracted"""
    
    es = Elasticsearch(["http://localhost:9200"])
    
    # Wait for index to be available
    try:
        count = es.count(index="root-documentation")
        print(f"Total documents in index: {count['count']}")
    except Exception as e:
        print(f"Error: Index not ready yet - {e}")
        return
    
    # Test cases for classes known to have methods and attributes
    test_cases = [
        {
            "name": "TH1 - Histogram class with many methods",
            "query": "TH1 Fill",
            "expected_in_content": ["SIGNATURE", "MEMBERS", "Fill", "method", "histogram"]
        },
        {
            "name": "TTree - Tree class with branches",
            "query": "TTree Branch",
            "expected_in_content": ["SIGNATURE", "MEMBERS", "Branch", "method"]
        },
        {
            "name": "TCanvas - Canvas drawing class",
            "query": "TCanvas Draw",
            "expected_in_content": ["SIGNATURE", "Draw", "method"]
        },
        {
            "name": "TFile - File I/O class",
            "query": "TFile Open Close",
            "expected_in_content": ["SIGNATURE", "Open", "method"]
        }
    ]
    
    print("\n" + "="*100)
    print("Testing Method and Attribute Extraction from Doxygen Documentation")
    print("="*100)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*100}")
        print(f"Test Case {i}: {test['name']}")
        print(f"Query: {test['query']}")
        print(f"{'='*100}")
        
        # Search for documents
        try:
            response = es.search(
                index="root-documentation",
                body={
                    "query": {
                        "multi_match": {
                            "query": test['query'],
                            "fields": ["title^3", "content^2", "code_snippets"]
                        }
                    },
                    "size": 3
                }
            )
        except Exception as e:
            print(f"❌ Search failed: {e}")
            continue
        
        hits = response['hits']['hits']
        
        if not hits:
            print(f"❌ No results found")
            continue
        
        for j, hit in enumerate(hits, 1):
            doc = hit['_source']
            content = doc.get('content', '')
            title = doc.get('title', 'No title')
            url = doc.get('url', 'No URL')
            score = hit['_score']
            
            print(f"\n--- Result {j} (score: {score:.2f}) ---")
            print(f"Title: {title}")
            print(f"URL: {url}")
            print(f"Content length: {len(content)} chars")
            
            # Check for expected content
            found_terms = []
            missing_terms = []
            
            for term in test['expected_in_content']:
                if term in content:
                    found_terms.append(term)
                else:
                    missing_terms.append(term)
            
            if found_terms:
                print(f"✓ Found expected content: {', '.join(found_terms)}")
            
            if missing_terms:
                print(f"⚠ Missing expected content: {', '.join(missing_terms)}")
            
            # Check structure
            has_signature = "SIGNATURE:" in content
            has_members = "MEMBERS:" in content
            has_description = "DESCRIPTION:" in content
            
            print(f"\nStructured content detection:")
            print(f"  - Has DESCRIPTION section: {'✓' if has_description else '✗'}")
            print(f"  - Has SIGNATURE section: {'✓' if has_signature else '✗'}")
            print(f"  - Has MEMBERS section: {'✓' if has_members else '✗'}")
            
            # Show content preview
            print(f"\nContent preview (first 800 chars):")
            print("-" * 100)
            preview = content[:800]
            # Show first part of DESCRIPTION
            if "DESCRIPTION:" in preview:
                desc_start = preview.find("DESCRIPTION:")
                desc_preview = preview[desc_start:desc_start+400]
                print(desc_preview)
            # Show first part of MEMBERS
            if "MEMBERS:" in content:
                members_start = content.find("MEMBERS:")
                members_preview = content[members_start:members_start+600]
                print("\n...\n")
                print(members_preview)
            else:
                print(preview)
            print("-" * 100)
    
    # Additional analysis: Sample documents with MEMBERS section
    print(f"\n{'='*100}")
    print("Statistical Analysis")
    print(f"{'='*100}")
    
    # Count documents with structured content
    try:
        with_members = es.count(
            index="root-documentation",
            body={
                "query": {
                    "match": {
                        "content": "MEMBERS:"
                    }
                }
            }
        )
        
        with_signature = es.count(
            index="root-documentation",
            body={
                "query": {
                    "match": {
                        "content": "SIGNATURE:"
                    }
                }
            }
        )
        
        total = es.count(index="root-documentation")
        
        print(f"Total documents: {total['count']}")
        print(f"Documents with MEMBERS section: {with_members['count']} ({with_members['count']/total['count']*100:.1f}%)")
        print(f"Documents with SIGNATURE section: {with_signature['count']} ({with_signature['count']/total['count']*100:.1f}%)")
        
        if with_members['count'] > 0:
            print("\n✓ Method and attribute extraction is working!")
        else:
            print("\n⚠ No structured member information found - extraction may need adjustment")
            
    except Exception as e:
        print(f"Error in statistical analysis: {e}")

if __name__ == "__main__":
    test_method_extraction()
