#!/usr/bin/env python3
"""
Test script for Geant4 Doxygen documentation search
"""

from index_geant4_doxygen import Geant4DoxygenIndexer


def test_geant4_doxygen_search():
    """Test search functionality with sample queries"""
    
    indexer = Geant4DoxygenIndexer(
        es_host="http://localhost:9200",
        index_name="geant4-doxygen"
    )
    
    print("=" * 80)
    print("🧪 Geant4 Doxygen Documentation Search Tests")
    print("=" * 80)
    
    # Test queries for different categories
    test_queries = [
        # Geometry classes
        "G4Box solid geometry",
        "G4Tubs cylindrical shape",
        "G4LogicalVolume properties",
        
        # Physics classes
        "G4VPhysicalVolume placement",
        "G4Material definition",
        "G4Element atomic number",
        
        # Particle classes
        "G4ParticleDefinition properties",
        "G4Electron particle",
        "G4Gamma photon",
        
        # Process classes
        "G4VProcess physics process",
        "G4Transportation particle transport",
        "G4Cerenkov radiation",
        
        # Manager classes
        "G4RunManager simulation",
        "G4EventManager event loop",
        "G4TrackingManager tracking",
        
        # Header files
        "G4Types header file",
        "G4ThreeVector vector class",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        results = indexer.search(query, k=3, hybrid=True)
        
        if not results:
            print("❌ No results found")
            continue
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['title']}")
            print(f"    🎯 Score: {result['score']:.3f}")
            print(f"    📁 Category: {result['category']}")
            print(f"    🔗 URL: {result['url']}")
            print(f"    📄 Preview: {result['content'][:200]}...")
    
    print(f"\n{'='*80}")
    print("✓ Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_geant4_doxygen_search()
