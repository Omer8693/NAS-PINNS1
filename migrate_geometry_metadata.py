#!/usr/bin/env python3
"""
migrate_geometry_metadata.py

Add geometry metadata to existing v2 slice JSON files.
This allows the web interface to properly display domains without re-running training.

Usage:
  python3 migrate_geometry_metadata.py
"""

import os
import json
import sys

# Geometry definitions from domains_3d.py
GEOMETRY_METADATA = {
    "rectangular": {
        "type": "rectangular",
        "params": {
            "Lx": 1.3,
            "Ly": 0.6,
            "Lz": 0.4,
        }
    },
    "cylinder": {
        "type": "cylinder",
        "params": {
            "R": 0.25,
            "H": 0.6,
        }
    },
    "stacked": {
        "type": "stacked",
        "params": {
            "L_cube": 0.5,
            "N_stack": 2,
            "Lz": 1.0,
        }
    },
    "lshape": {
        "type": "lshape",
        "params": {
            "Lx": 0.8,
            "Ly": 0.8,
            "Lz": 0.4,
            "cut_x": 0.3,
            "cut_y": 0.3,
        }
    },
}

def migrate_file(filepath):
    """
    Add geometry metadata to a single slice.json file.
    Returns (success, message)
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failed to read {filepath}: {e}"
    
    # Skip if geometry already exists
    if 'geometry' in data:
        return True, f"Already has geometry: {filepath}"
    
    # Extract domain name from filename
    # Format: {domain}_{arch}_skip{s}_slice.json
    basename = os.path.basename(filepath)
    domain = basename.split('_')[0]
    
    if domain not in GEOMETRY_METADATA:
        return False, f"Unknown domain '{domain}' in {filepath}"
    
    # Add geometry field
    data['geometry'] = GEOMETRY_METADATA[domain]
    
    # Write back
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f)
        return True, f"Migrated: {filepath}"
    except Exception as e:
        return False, f"Failed to write {filepath}: {e}"

def main():
    v2_dir = os.path.join(
        os.path.dirname(__file__),
        "NAS-PINNS3",
        "level8_nas_mco_pinn",
        "results",
        "v2"
    )
    
    if not os.path.isdir(v2_dir):
        print(f"❌ Directory not found: {v2_dir}")
        sys.exit(1)
    
    # Find all slice.json files
    slice_files = [
        os.path.join(v2_dir, f)
        for f in os.listdir(v2_dir)
        if f.endswith("_slice.json")
    ]
    
    print(f"Found {len(slice_files)} slice files to migrate\n")
    
    success_count = 0
    already_migrated = 0
    failed = []
    
    for filepath in sorted(slice_files):
        success, msg = migrate_file(filepath)
        
        if success:
            if "Already has geometry" in msg:
                already_migrated += 1
                print(f"  ⏭️  {msg}")
            else:
                success_count += 1
                print(f"  ✅ {msg}")
        else:
            failed.append((filepath, msg))
            print(f"  ❌ {msg}")
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully migrated: {success_count}")
    print(f"⏭️  Already migrated: {already_migrated}")
    print(f"❌ Failed: {len(failed)}")
    print(f"{'='*60}")
    
    if failed:
        print("\nFailed files:")
        for path, msg in failed:
            print(f"  - {path}: {msg}")
        sys.exit(1)
    
    print("\n✨ All files migrated successfully!")
    print("\nYou can now use the updated web interface:")
    print("  cd NAS-PINNS3/web && python3 app.py")
    print("  Visit: http://localhost:5000/demo")

if __name__ == "__main__":
    main()
