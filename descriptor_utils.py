import numpy as np
from itertools import combinations

# allowed donor atoms
DONORS = ["N","O","S","P","Cl","Br","I","F"]

# atomic number mapping
ATOMIC_NUMBERS = {
"H":1,"C":6,"N":7,"O":8,"F":9,
"P":15,"S":16,"Cl":17,
"Br":35,"I":53,
"Co":27
}


def normalize_atom(atom):
    """Convert atomic number to element symbol"""

    atom = atom.strip()

    # if atomic number
    if atom.isdigit():
        num = int(atom)
        for k,v in ATOMIC_NUMBERS.items():
            if v == num:
                return k
        return atom

    # normalize case
    return atom.capitalize()


def read_xyz(file):

    lines = file.read().decode().splitlines()

    atoms = []
    coords = []

    for line in lines[2:]:

        parts = line.split()

        atom = normalize_atom(parts[0])

        atoms.append(atom)

        coords.append(list(map(float, parts[1:4])))

    return atoms, np.array(coords)


def distance(a,b):
    return np.linalg.norm(a-b)


def angle(a,b,c):

    ba = a-b
    bc = c-b

    cosang = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))

    return np.degrees(np.arccos(cosang))


def extract_descriptors(atoms,coords):

    # find Co index
    if "Co" not in atoms:
        raise ValueError("No Co atom found")

    co_index = atoms.index("Co")
    co_coord = coords[co_index]

    candidates = []

    for i,(atom,coord) in enumerate(zip(atoms,coords)):

        if i == co_index:
            continue

        d = distance(co_coord,coord)

        # main bond window
        if 1.6 <= d <= 2.875:

            # ignore H unless very close
            if atom == "H":

                if d <= 1.9:
                    candidates.append((atom,i,d))

            else:
                candidates.append((atom,i,d))

    # remove H if non-H donors exist
    nonH = [c for c in candidates if c[0] != "H"]

    if len(nonH) >= 3:
        candidates = nonH

    # keep nearest atoms
    candidates = sorted(candidates, key=lambda x: x[2])[:3]

    if len(candidates) != 3:
        raise ValueError("Structure does not contain exactly 3 valid donors")

    bond_lengths = [c[2] for c in candidates]
    donor_indices = [c[1] for c in candidates]

    BL = sorted(bond_lengths)

    angles = []

    for i,j in combinations(donor_indices,2):

        ang = angle(coords[i],co_coord,coords[j])
        angles.append(ang)

    BA = sorted(angles)

    return {
        "BL1":BL[0],
        "BL2":BL[1],
        "BL3":BL[2],
        "BA1":BA[0],
        "BA2":BA[1],
        "BA3":BA[2]
    }
