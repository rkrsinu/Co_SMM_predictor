import numpy as np
from itertools import combinations


# atomic number → element
ATOMIC_MAP = {
1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",
11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",
19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",
27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",32:"Ge",33:"As",34:"Se",
35:"Br",36:"Kr",53:"I"
}


def normalize_atom(atom):

    atom = atom.strip()

    if atom.isdigit():
        return ATOMIC_MAP.get(int(atom), atom)

    return atom.capitalize()


def read_xyz(file):

    lines = file.read().decode().splitlines()

    atoms = []
    coords = []

    for line in lines:

        parts = line.split()

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
        coords.append([x,y,z])

    return atoms, np.array(coords)


def distance(a,b):
    return np.linalg.norm(a-b)


def angle(a,b,c):

    ba = a-b
    bc = c-b

    cosang = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))

    return np.degrees(np.arccos(cosang))


def find_donors(atoms,coords):

    METALS = [
        "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
        "Y","Zr","Nb","Mo","Ru","Rh","Pd","Ag","Cd",
        "La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy"
    ]

    metals = [a for a in atoms if a in METALS]

    co_indices = [i for i,a in enumerate(atoms) if a=="Co"]

    # ---- validation ----

    if len(co_indices)!=1 or len(metals)!=1:

        message = (
        "⚠️ This predictor is designed only for "
        "mononuclear three-coordinate Co complexes. "
        "Please upload a structure containing only one Co center."
        )

        return None,None,message

    co_index = co_indices[0]
    co_coord = coords[co_index]

    candidates=[]

    for i,(atom,coord) in enumerate(zip(atoms,coords)):

        if i==co_index:
            continue

        d = distance(co_coord,coord)

        if atom=="H":

            if 1.4<=d<=1.9:
                candidates.append((i,d))

        else:

            if 1.6<=d<=2.875:
                candidates.append((i,d))

    if len(candidates)<3:

        return None,None,"⚠️ Unable to detect three donor atoms around Co."

    candidates = sorted(candidates,key=lambda x:x[1])

    donors = candidates[:3]

    return co_index, donors, None


def compute_descriptors(coords,co_index,donor_indices):

    co = coords[co_index]

    BL=[]

    for i in donor_indices:
        BL.append(distance(co,coords[i]))

    BL_sorted = sorted(BL)

    angles=[]

    for i,j in combinations(donor_indices,2):

        ang = angle(coords[i],co,coords[j])
        angles.append(ang)

    BA_sorted = sorted(angles)

    return BL_sorted, BA_sorted
