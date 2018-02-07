#!/usr/bin/env python3

import copy
import json
import numpy
import random
import scipy.spatial
import sys
import time

# Default threshold distance, in angstroms, for a bond between two atoms
DEFAULT_BOND_RADIUS = 1.8

# Periodic Table (parsed from http://periodic.lanl.gov/index.shtml)
ELEMENTS = [
    None, # no 0th element
    ("H",  1.008),  ("He", 4.003),  ("Li", 6.94),   ("Be", 9.012),
    ("B",  10.81),  ("C",  12.01),  ("N",  14.01),  ("O",  16.0),
    ("F",  19.0),   ("Ne", 20.18),  ("Na", 22.99),  ("Mg", 24.31),
    ("Al", 26.98),  ("Si", 28.09),  ("P",  30.97),  ("S",  32.06),
    ("Cl", 35.45),  ("Ar", 39.95),  ("K",  39.1),   ("Ca", 40.08),
    ("Sc", 44.96),  ("Ti", 47.88),  ("V",  50.94),  ("Cr", 52.0),
    ("Mn", 54.94),  ("Fe", 55.85),  ("Co", 58.93),  ("Ni", 58.69),
    ("Cu", 63.55),  ("Zn", 65.39),  ("Ga", 69.72),  ("Ge", 72.64),
    ("As", 74.92),  ("Se", 78.96),  ("Br", 79.9),   ("Kr", 83.79),
    ("Rb", 85.47),  ("Sr", 87.62),  ("Y8", 8.91),   ("Zr", 91.22),
    ("Nb", 92.91),  ("Mo", 95.96),  ("Tc", 98.0),   ("Ru", 101.1),
    ("Rh", 102.9),  ("Pd", 106.4),  ("Ag", 107.9),  ("Cd", 112.4),
    ("In", 114.8),  ("Sn", 118.7),  ("Sb", 121.8),  ("Te", 127.6),
    ("I",  126.9),  ("Xe", 131.3),  ("Cs", 132.9),  ("Ba", 137.3),
    ("La", 138.9),  ("Ce", 140.1),  ("Pr", 140.9),  ("Nd", 144.2),
    ("Pm", 145.0),  ("Sm", 150.4),  ("Eu", 152.0),  ("Gd", 157.2),
    ("Tb", 158.9),  ("Dy", 162.5),  ("Ho", 164.9),  ("Er", 167.3),
    ("Tm", 168.9),  ("Yb", 173.0),  ("Lu", 175.0),  ("Hf", 178.5),
    ("Ta", 180.9),  ("W",  183.9),  ("Re", 186.2),  ("Os", 190.2),
    ("Ir", 192.2),  ("Pt", 195.1),  ("Au", 197.0),  ("Hg", 200.5),
    ("Tl", 204.38), ("Pb", 207.2),  ("Bi", 209.0),  ("Po", 209.0),
    ("At", 210.0),  ("Rn", 222.0),  ("Fr", 223.0),  ("Ra", 226.0),
    ("Ac", 227.0),  ("Th", 232.0),  ("Pa", 231.0),  ("U",  238.0),
    ("Np", 237.0),  ("Pu", 244.0),  ("Am", 243.0),  ("Cm", 247.0),
    ("Bk", 247.0),  ("Cf", 251.0),  ("Es", 252.0),  ("Fm", 257.0),
    ("Md", 258.0),  ("No", 259.0),  ("Lr", 262.0),  ("Rf", 267.0),
    ("Db", 268.0),  ("Sg", 269.0),  ("Bh", 270.0),  ("Hs", 277.0),
    ("Mt", 278.0),  ("Ds", 281.0),  ("Rg", 282.0),  ("Cn", 285.0),
    ("Nh", 286.0),  ("Fl", 289.0),  ("Mc", 289.0),  ("Lv", 293.0),
    ("Ts", 294.0),  ("Og", 294.0)
]

# The width, in angstroms, of the margin around the bounding box
BOUNDARY_MARGIN = 0.8

DEFAULT_XYZ_COMMENT = "generated with nme"

def symbol_to_atomic_number(symbol):
    """ Convert an element's symbol to its atomic number

    Args:
        symbol: A string containing the symbol of an element

    Returns:
        An integer corresponding to the element's atomic number
    """

    for i in range(1, len(ELEMENTS)):
        if (ELEMENTS[i][0].lower() == symbol.lower()):
            return i

class Atom(object):

    def __init__(self, element, xyz):
        """ Initialize Atom object

        Args:
            element:
            xyz: A tuple or list of size 3, corresponding to the atom's
                coordinates
        """

        self.element = element
        self.xyz = numpy.array(xyz)
        self.attributes = {}

    # Dict-like storage of values
    def __setitem__(self, key, value):
        self.attributes[key] = value
    def __getitem__(self, key):
        return self.attributes[key]
    def __delitem__(self, key):
        del self.attributes[key]

class Molecule(object):

    def __init__(self, atoms = None):
        """ Initialize Molecule object

        Args:
            atoms: An array of atoms
        """

        if (atoms == None):
            self.atoms = []
        else:
            self.atoms = atoms

        self.attributes = {}

    def copy(self):
        """ Create a copy of this Molecule

        Returns:
            A copy of this Molecule
        """

        return copy.deepcopy(self)

    def append(self, other):
        """ Append atoms or molecules to the atoms array

        Args:
            other: An Atom or Molecule object or list of objects
        """

        if (type(other) != list):
            other = [other]
        for object_ in other:
            if (type(object_) == Atom):
                self.atoms.append(object_)
            elif (type(object_) == Molecule):
                self.atoms += object_.copy().atoms
            else:
                raise TypeError(
                    "Object of type %s cannot be appended to Molecule"
                    % type(object_)
                )

    def offset(self, offset):
        """ Offset the current molecule

        Args:
            offset: A tuple or list of size 3, corresponding to the translation
                that should be applied to all of the molecule's atoms
        """

        np_offset = numpy.array(offset)

        for atom in self.atoms:
            atom.xyz += np_offset

    def move_to(self, xyz):
        """ Move the molecule such that its centroid is at the desired position

        Args:
            xyz: A tuple or list of size 3, corresponding to the location that
                the molecule should be moved to
        """

        self.offset(numpy.array(xyz) - self.centroid)

    def find_bonds(self, bond_radius = DEFAULT_BOND_RADIUS):
        """ Find bonds between molecules

        Args:
            bond_radius: The distance, in angstroms, between two atoms under
                which a bond will be assumed

        Returns:
            A two-dimensional array of atoms where each pair of atoms
            represents a bond
        """

        all_coords = [atom.xyz for atom in self]

        print(
            "computing euclidean distance matrix for %d atoms" % len(self.atoms)
        )
        distances = scipy.spatial.distance.cdist(
            all_coords, all_coords, "euclidean"
        )

        return [
            [self.atoms[bond[0]], self.atoms[bond[1]]]
            for bond in numpy.argwhere(
                (distances < bond_radius) & (distances != 0)
            )
        ]

    def write_xyz(self, filename, comment = DEFAULT_XYZ_COMMENT):
        """ Write the current molecule to an XYZ file

        Args:
            filename: The path that the XYZ data should be written to
            comment: A comment to include in the XYZ file
        """

        with open(filename, "w") as f:
            f.write("%d\n%s\n" % (len(self.atoms), comment))
            for atom in self.atoms:
                f.write("%d\t%f\t%f\t%f\n" % (
                    atom.element, atom.xyz[0], atom.xyz[1], atom.xyz[2]
                ))

    def write_lammps(self, filename):
        """ Syntactic sugar """

        write_lammps(self, filename)

    @property
    def bonds(self):
        return self.find_bonds()

    @property
    def centroid(self):
        return sum([atom.xyz for atom in self.atoms]) / len(self.atoms)

    # Iteration procedures
    def __iter__(self):
        self.atom_iter_index = -1
        return self
    def __next__(self):
        self.atom_iter_index += 1
        if (self.atom_iter_index == len(self.atoms)):
            raise StopIteration
        else:
            return self.atoms[self.atom_iter_index]

    # Dict-like storage of values
    def __setitem__(self, key, value):
        self.attributes[key] = value
    def __getitem__(self, key):
        return self.attributes[key]
    def __delitem__(self, key):
        del self.attributes[key]

    # Syntactic sugar
    def __iadd__(self, other):
        self.append(other)
        return self
    def __add__(self, other):
        new = Molecule()
        new += self
        new += other
        return new

class Workspace(object):

    def __init__(self):
        self.molecules = []
        self.attributes = {}

    def append(self, object_):
        """ Append atoms or molecules to the workspace

        Args:
            other: An Atom or Molecule object or list of objects
        """

        if (type(object_) == Molecule):
            self.molecules.append(object_)
        elif (type(object_) == Atom):
            new_molecule = Molecule()
            new_molecule.append(object_)
            self.molecules.append(new_molecule)
        else:
            raise TypeError(
                "Object of type %s cannot be appended to Workspace"
                % type(object_)
            )

    def write_xyz(self, filename, comment = DEFAULT_XYZ_COMMENT):
        """ Write the current workspace to an XYZ file

        Args:
            filename: The path that the XYZ data should be written to
            comment: A comment to include in the XYZ file
        """

        with open(filename, "w") as f:
            f.write("%d\n%s\n" % (
                sum([
                    len(molecule.atoms)
                    for molecule in self.molecules
                ]),
                comment
            ))
            for molecule in self.molecules:
                for atom in molecule.atoms:
                    f.write("%d\t%f\t%f\t%f\n" % (
                        atom.element, atom.xyz[0], atom.xyz[1], atom.xyz[2]
                    ))

    def write_lammps(self, filename):
        """ Syntactic sugar """

        write_lammps(self.molecules, filename)

    # Dict-like storage of values
    def __setitem__(self, key, value):
        self.attributes[key] = value
    def __getitem__(self, key):
        return self.attributes[key]
    def __delitem__(self, key):
        del self.attributes[key]

    # Syntactic sugar
    def __iadd__(self, other):
        self.append(other)
        return self
    def __add__(self, other):
        new = Workspace()
        new += self
        new += other
        return new

def read_xyz(filepath):
    """ Read a .xyz file into an array of atoms

    Args:
        filepath: The path of the xyz file

    Returns:
        A Molecule object of the atoms in the .xyz file
    """

    atoms = []
    i = 0

    with open(filepath, "r") as f:
        for line in f.readlines():
            row = line.rstrip().split()
            try:

                try:
                    atomic_number = int(row[0])
                except:
                    atomic_number = symbol_to_atomic_number(row[0])

                atoms.append(
                    Atom(
                        atomic_number, # atomic number
                        numpy.array([
                            float(row[i])
                            for i in range(1, 4)
                        ]) # coordinates
                    )
                )

                i += 1
                sys.stdout.write("\rfound %d atoms" % i)
                sys.stdout.flush()
            except ValueError:
                pass
            except IndexError:
                pass

    print("")

    return Molecule(atoms)

def write_lammps(molecules, filename):
    """ Write the given molecules to a LAMMPS data file

    Args:
        molecules: A molecule or array of molecules to be written
        filename: The path that the LAMMPS data should be written to
    """

    if (type(molecules) != list):
        molecules = [molecules]

    all_atoms = []
    for molecule in molecules:
        all_atoms += molecule.atoms

    all_x = [atom.xyz[0] for atom in molecule]
    all_y = [atom.xyz[1] for atom in molecule]
    all_z = [atom.xyz[2] for atom in molecule]

    unique_elements = list(
        sorted(
            set([
                atom.element
                for atom in all_atoms
            ])
        )
    )

    atom_element_map = dict(
        zip(
            unique_elements, range(1, len(unique_elements) + 1)
        )
    )

    i_atom = 0
    i_mol = 0
    with open(filename, "w") as f:

        # Header
        f.write("LAMMPS data file\n\n")

        # Summary
        f.write(
            "%d atoms\n0 bonds\n0 angles\n0 dihedrals\n0 impropers\n\n"
            % len(molecule.atoms)
        )

        # Summary - types
        f.write(
            "%d atom types\n0 bond types\n0 angle types\n0 dihedral types\n0 improper types\n\n"
            % len(unique_elements)
        )

        # Bounds
        f.write("%0.6f %0.6f xlo xhi\n" % (
            min(all_x) - BOUNDARY_MARGIN, max(all_x) + BOUNDARY_MARGIN
        ))
        f.write("%0.6f %0.6f ylo yhi\n" % (
            min(all_y) - BOUNDARY_MARGIN, max(all_y) + BOUNDARY_MARGIN
        ))
        f.write("%0.6f %0.6f zlo zhi\n\n" % (
            min(all_z) - BOUNDARY_MARGIN, max(all_z) + BOUNDARY_MARGIN
        ))

        # Masses of involved elements
        f.write("Masses\n\n")
        for element in atom_element_map:
            f.write("%d %s\n" % (
                atom_element_map[element], ELEMENTS[element][1]
            ))

        # Individual atoms
        f.write("\nAtoms\n\n")
        for molecule in molecules:
            i_mol += 1
            for atom in molecule:
                i_atom += 1
                xyz = atom.xyz
                atom_number = atom_element_map[atom.element]
                f.write("%d\t%d\t%d\t%s\t%0.6f\t%0.6f\t%0.6f\n" % (
                    i_atom, i_mol, atom_number, 0.0, xyz[0], xyz[1], xyz[2]
                ))
