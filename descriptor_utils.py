import numpy as np
from itertools import combinations


# Atomic number → element symbol
ATOMIC_MAP = {
1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",
11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",
19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",
27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",32:"Ge",33:"As",34:"Se",
35:"Br",36:"Kr",53:"I"
}


def normalize_atom(atom):
    """Convert atomic number or symbol to proper element symbol"""

    atom = atom.strip()

    # if atomic number
    if atom.isdigit():
        num = int(atom)
        return ATOMIC_MAP.get(num, atom)

    # symbol formatting
    atom = atom.capitalize()

    return atom


def read_xyz(file):
    """
    Reads xyz files in multiple formats:

    Format 1:
    N
    comment
    C 0.0 0.0 0.0

    Format 2:
    C 0.0 0.0 0.0

    Format 3:
    6 0.0 0.0 0.0
    """

    lines = file.read().decode().splitlines()

    atoms = []
    coords = []

    for line in lines:

        parts = line.split()

        # must have at least atom + 3 coords
        if len(parts) < 4:
            continue

        atom = normalize_atom(parts[0])

        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except:
            continue

        atoms.append(atom)
        coords.append([x, y, z])

    return atoms, np.array(coords)


def distance(a, b):
    return np.linalg.norm(a - b)


def angle(a, b, c):

    ba = a - b
    bc = c - b

    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    return np.degrees(np.arccos(cosang))


def extract_descriptors(atoms, coords):

    if "Co" not in atoms:
        raise ValueError("No Co atom found")

    co_index = atoms.index("Co")
    co_coord = coords[co_index]

    candidates = []

    for i, (atom, coord) in enumerate(zip(atoms, coords)):

        if i == co_index:
            continue

        d = distance(co_coord, coord)

        # Hydrogen rule
        if atom == "H":

            if 1.4 <= d <= 1.9:
                candidates.append((atom, i, d))

        # Other atoms
        else:

            if 1.6 <= d <= 2.875:
                candidates.append((atom, i, d))

    if len(candidates) < 3:
        raise ValueError("Less than 3 donor atoms detected")

    # sort by distance
    candidates = sorted(candidates, key=lambda x: x[2])

    # take 3 closest donors
    candidates = candidates[:3]

    bond_lengths = [c[2] for c in candidates]
    donor_indices = [c[1] for c in candidates]

    BL = sorted(bond_lengths)

    angles = []

    for i, j in combinations(donor_indices, 2):

        ang = angle(coords[i], co_coord, coords[j])
        angles.append(ang)

    BA = sorted(angles)

    return {
        "BL1": BL[0],
        "BL2": BL[1],
        "BL3": BL[2],
        "BA1": BA[0],
        "BA2": BA[1],
        "BA3": BA[2]
    }
