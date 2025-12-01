#!/usr/bin/env python3
"""
Validate generated URLs by sampling from the Elasticsearch index
"""

import logging

import requests
from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def validate_urls(es_host="http://localhost:9200", index_name="root-documentation", sample_size=50):
    """
    Sample URLs from index and validate they exist

    Args:
        es_host: Elasticsearch host
        index_name: Index name
        sample_size: Number of URLs to check per category
    """
    es = Elasticsearch([es_host])

    # Get sample documents from each category
    categories = ["html", "macros", "notebooks", "pyzdoc"]

    logger.info("=" * 70)
    logger.info("URL Validation Report")
    logger.info("=" * 70)

    total_checked = 0
    total_valid = 0
    total_invalid = 0

    for category in categories:
        logger.info(f"\n📁 Category: {category}")
        logger.info("-" * 70)

        # Sample documents from this category
        query = {
            "size": sample_size,
            "query": {"term": {"category": category}},
            "_source": ["title", "url", "category"],
        }

        try:
            response = es.search(index=index_name, body=query)
            hits = response["hits"]["hits"]

            if not hits:
                logger.info(f"  ⚠️  No documents found in category '{category}'")
                continue

            # Check each URL
            valid_count = 0
            invalid_urls = []

            for hit in hits[:10]:  # Check first 10 from sample
                url = hit["_source"]["url"]
                title = hit["_source"]["title"]

                try:
                    # HEAD request to check if URL exists
                    resp = requests.head(url, timeout=5, allow_redirects=True)

                    if resp.status_code == 200:
                        valid_count += 1
                        total_valid += 1
                        logger.info(f"  ✓ {url}")
                    else:
                        invalid_urls.append((url, title, resp.status_code))
                        total_invalid += 1

                except requests.RequestException as e:
                    invalid_urls.append((url, title, str(e)))
                    total_invalid += 1

                total_checked += 1

            # Report for this category
            logger.info(f"\n  Valid: {valid_count}/{len(hits[:10])}")

            if invalid_urls:
                logger.info(f"\n  ✗ Invalid URLs:")
                for url, title, error in invalid_urls[:3]:  # Show first 3
                    logger.info(f"    - {url}")
                    logger.info(f"      Title: {title}")
                    logger.info(f"      Error: {error}")

        except Exception as e:
            logger.error(f"  ✗ Error querying category '{category}': {e}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"Total URLs checked: {total_checked}")
    logger.info(f"Valid: {total_valid} ({100*total_valid/total_checked:.1f}%)")
    logger.info(f"Invalid: {total_invalid} ({100*total_invalid/total_checked:.1f}%)")
    logger.info("=" * 70)


if __name__ == "__main__":
    validate_urls()
