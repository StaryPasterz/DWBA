import json
import os
from dataclasses import dataclass
from potential_core import CorePotentialParams

@dataclass
class AtomEntry:
    name: str           
    Z: float            
    core_params: CorePotentialParams
    default_n: int      
    default_l: int      
    ip_ev: float        
    description: str    

# --- Database Loader ---

DB_FILE = os.path.join(os.path.dirname(__file__), "atoms.json")
ATOM_DATABASE = {}

def load_database():
    """Refreshes the database from atoms.json."""
    global ATOM_DATABASE
    ATOM_DATABASE.clear()
    
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            data = json.load(f)
            
        for key, val in data.items():
            # Skip metadata entries (like _comment)
            if not isinstance(val, dict):
                continue
            
            # Convert list back to CorePotentialParams
            a = val["a_params"]
            params = CorePotentialParams(
                Zc=val["Zc"],
                a1=a[0], a2=a[1], a3=a[2], a4=a[3], a5=a[4], a6=a[5]
            )
            
            entry = AtomEntry(
                name=val["name"],
                Z=val["Z"],
                core_params=params,
                default_n=val["default_n"],
                default_l=val["default_l"],
                ip_ev=val["ip_ev"],
                description=val.get("description", "")
            )
            ATOM_DATABASE[key] = entry
    else:
        print(f"[WARNING] Database file {DB_FILE} not found!")

# Initial load
load_database()

def get_atom_list():
    """Returns a list of available atom names."""
    # Reload every time? No, maybe just rely on initial load unless forced.
    # But for interactive fit tool, reload is good.
    load_database() 
    return list(ATOM_DATABASE.keys())

def get_atom(name: str) -> AtomEntry:
    """Retrieves an atom entry by name."""
    if name not in ATOM_DATABASE:
        load_database() # Retry
        
    if name not in ATOM_DATABASE:
        raise ValueError(f"Atom '{name}' not found in database.")
    return ATOM_DATABASE[name]

