#!/usr/bin/env python3
"""
Test script to validate the quality of extracted content from ROOT documentation
"""

from elasticsearch import Elasticsearch


def test_content_quality():
    """Test the quality of extracted content"""

    es = Elasticsearch(["http://localhost:9200"])

    # Test queries to check content quality
    test_cases = [
        {
            "query": "TActivation",
            "expected_terms": ["activation", "function", "neuron", "TMVA"],
            "avoid_terms": ["Loading...", "Searching...", "No Matches", "SVG", "try Firefox"],
        },
        {
            "query": "TTree",
            "expected_terms": ["tree", "branch", "ROOT", "data"],
            "avoid_terms": ["Loading...", "Searching...", "No Matches"],
        },
        {
            "query": "histogram",
            "expected_terms": ["TH1", "bin", "axis", "fill"],
            "avoid_terms": ["Loading...", "Searching...", "SVG warning"],
        },
    ]

    print("=" * 80)
    print("Content Quality Test for ROOT Documentation")
    print("=" * 80)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}: {test['query']}")
        print(f"{'='*80}")

        # Search for documents
        response = es.search(
            index="root-documentation",
            body={
                "query": {"multi_match": {"query": test["query"], "fields": ["title^2", "content"]}},
                "size": 3,
            },
        )

        hits = response["hits"]["hits"]

        if not hits:
            print(f"❌ No results found for '{test['query']}'")
            continue

        for j, hit in enumerate(hits, 1):
            doc = hit["_source"]
            content = doc.get("content", "")
            title = doc.get("title", "No title")
            url = doc.get("url", "No URL")

            print(f"\n--- Result {j} ---")
            print(f"Title: {title}")
            print(f"URL: {url}")
            print(f"Content length: {len(content)} chars")
            print(f"Category: {doc.get('category', 'N/A')}")
            print(f"Type: {doc.get('content_type', 'N/A')}")

            # Check for unwanted content
            found_bad_terms = []
            for term in test["avoid_terms"]:
                if term.lower() in content.lower():
                    found_bad_terms.append(term)

            if found_bad_terms:
                print(f"❌ Found unwanted terms: {', '.join(found_bad_terms)}")
            else:
                print(f"✓ No unwanted UI/SVG terms found")

            # Check for expected terms
            found_good_terms = []
            for term in test["expected_terms"]:
                if term.lower() in content.lower():
                    found_good_terms.append(term)

            if found_good_terms:
                print(f"✓ Found expected terms: {', '.join(found_good_terms)}")
            else:
                print(f"⚠ Expected terms not found in content")

            # Show content preview
            print(f"\nContent preview (first 500 chars):")
            print("-" * 80)
            print(content[:500])
            print("-" * 80)

    # Additional statistics
    print(f"\n{'='*80}")
    print("Overall Statistics")
    print(f"{'='*80}")

    # Count total documents
    count_response = es.count(index="root-documentation")
    print(f"Total indexed documents: {count_response['count']}")

    # Sample random documents to check for bad content
    sample_response = es.search(index="root-documentation", body={"query": {"match_all": {}}, "size": 10})

    bad_content_count = 0
    for hit in sample_response["hits"]["hits"]:
        content = hit["_source"].get("content", "")
        if any(term in content for term in ["Loading...", "Searching...", "This browser is not able to show SVG"]):
            bad_content_count += 1

    print(f"Documents with UI artifacts (from sample of 10): {bad_content_count}")

    if bad_content_count == 0:
        print("✓ Content quality looks good!")
    else:
        print("⚠ Some documents still contain UI artifacts")


if __name__ == "__main__":
    test_content_quality()
