#!/usr/bin/env python3
"""
Example script showing the improved content structure with methods and attributes
"""

import requests
from bs4 import BeautifulSoup

# Example: Show what content is being extracted from a real Doxygen page
url = "https://root.cern/doc/master/classTH1.html"

print("=" * 100)
print("Example: Extracting Methods and Attributes from Doxygen HTML")
print("=" * 100)
print(f"\nFetching: {url}\n")

try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Extract structured content
    content_parts = []
    methods_and_attributes = []

    content_div = soup.find("div", {"class": "contents"}) or soup.find("div", {"id": "doc-content"})

    if content_div:
        # 1. Extract class description
        print("1. CLASS DESCRIPTION:")
        print("-" * 100)
        for textblock in content_div.find_all("div", class_="textblock", recursive=True):
            if not textblock.find_parent("div", class_="memitem"):
                text = textblock.get_text(separator=" ", strip=True)
                if text and len(text) > 20:
                    print(text[:500])  # First 500 chars
                    print("...")
                    break

        # 2. Extract methods and attributes
        print("\n2. METHODS AND ATTRIBUTES (first 5):")
        print("-" * 100)

        count = 0
        for memitem in content_div.find_all("div", class_="memitem"):
            if count >= 5:
                break

            member_info = []

            # Get signature
            memproto = memitem.find("div", class_="memproto")
            if memproto:
                memname_table = memproto.find("table", class_="memname")
                if memname_table:
                    signature = memname_table.get_text(separator=" ", strip=True)
                    if signature and len(signature) > 5:
                        member_info.append(f"SIGNATURE: {signature}")

            # Get documentation
            memdoc = memitem.find("div", class_="memdoc")
            if memdoc:
                for p in memdoc.find_all("p", recursive=False):
                    desc = p.get_text(separator=" ", strip=True)
                    if desc and len(desc) > 10:
                        member_info.append(f"DESC: {desc[:200]}")
                        break

            if member_info:
                count += 1
                print(f"\nMethod/Attribute #{count}:")
                for info in member_info:
                    print(f"  {info}")

        print(f"\n... and {len(content_div.find_all('div', class_='memitem')) - 5} more members")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 100)
print("This structured information will be stored in Elasticsearch for better search results")
print("=" * 100)
