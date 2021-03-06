#!/usr/bin/env python3

"""
.. |xyz| replace:: A tuple, list, or numpy array of size 3
"""

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
    ("Rb", 85.47),  ("Sr", 87.62),  ("Y",  88.91),  ("Zr", 91.22),
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
DEFAULT_BOUNDARY_MARGIN = 0.8

DEFAULT_XYZ_COMMENT = "generated with nme"

## WRAPPER FUNCTIONS FOR INTERACTING WITH ELEMENTS ARRAY #######################

def symbol_to_atomic_number(symbol):
    """ Convert an element's symbol to its atomic number """

    for i in range(1, len(ELEMENTS)):
        if (ELEMENTS[i][0].lower() == symbol.lower()):
            return i

def symbol_to_atomic_mass(symbol):
    """ Convert an element's symbol to its atomic mass """

    first = True
    for element in ELEMENTS:
        if (first):
            first = False
        elif (element[0].lower() == symbol.lower()):
            return element[1]

def atomic_number_to_symbol(atomic_number):
    """ Convert an atomic number to a symbol """

    return ELEMENTS[atomic_number][0]

def atomic_number_to_atomic_mass(atomic_number):
    """ Convert an atomic number to that element's atomic mass """

    return ELEMENTS[atomic_number][1]

def atomic_mass_to_atomic_number(atomic_mass):
    """ Convert an atomic mass to an atomic number by guessing what the element
    is by finding the element with the closest atomic mass """

    # return first item of sorted list of tuples where the first index is the
    # atomic number and the second index is the absolute value of the
    # difference in that element's atomic mass and the provided atomic mass
    return sorted(
        zip(
            range(len(ELEMENTS)),
            [
                abs(element[1] - atomic_mass)
                    if (element is not None)
                    else 1e100
                for element in ELEMENTS
            ]
        ),
        key = lambda x: x[1]
    )[0][0]

def atomic_mass_to_symbol(atomic_mass):
    """ Convert an atomic mass to a symbol """

    return atomic_number_to_symbol(
        atomic_mass_to_atomic_number(atomic_mass)
    )

################################################################################

class Atom(object):

    def __init__(self, element, xyz):
        """ Initialize an :class:`Atom`. This is the fundamental object of
        ``nme``.

        :param int element: The atomic number of the element
        :param xyz: |xyz| corresponding to the atom's coordinates
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
        """ Initialize a :class:`Molecule`. A :class:`Molecule` acts a
        container for :class:`Atom` objects and can be used to manipulate
        groups of atoms or write a group of atoms to the disk.

        :param list atoms: (optional) A list of :class:`Atom` objects. If no
            atoms are provied then an empty molecule is returned.
        """

        if (atoms == None):
            self.atoms = []
        else:
            self.atoms = atoms

        self.attributes = {}

    def copy(self):
        """ Create a copy of this molecule

        :return: A copy of this molecule
        :rtype: :class:`Molecule`
        """

        return copy.deepcopy(self)

    def append(self, other):
        """ Append atoms or molecules to the atoms array

        :param other: An :class:`Atom`, :class:`Molecule`, or a list of atoms
            and molecules
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

        :param offset: |xyz| corresponding to the translation that should be
            applied to all of the molecule's atoms
        """

        np_offset = numpy.array(offset)

        for atom in self.atoms:
            numpy.add(atom.xyz, np_offset, out = atom.xyz, casting = "unsafe")

    def move_to(self, xyz):
        """ Move the molecule such that its centroid is at the desired position

        :param xyz: |xyz|, corresponding to the location that the molecule
            should be moved to
        """

        self.offset(numpy.array(xyz) - self.centroid)

    def rotate(self, theta, phi, psi, degrees = False):
        """ Rotate the molecule about its centroid in 3D space

        :param float theta: The angle to rotate the molecule around its X axis
        :param float phi: The angle to rotate the molecule around its Y axis
        :param float psi: The angle to rotate the molecule around its Z axis
        :param bool degrees: (optional) If True, the angles given are assumed
            to be in degrees and will be converted to radians before being used
        """

        # https://www.siggraph.org/education/materials/HyperGraph/modeling/mod_tran/3drota.htm

        if (degrees == True):
            theta = numpy.radians(theta)
            phi = numpy.radians(phi)
            psi = numpy.radians(psi)

        rx = numpy.array([
            [1, 0, 0, 0],
            [0, numpy.cos(theta), numpy.sin(theta), 0],
            [0, -numpy.sin(theta), numpy.cos(theta), 0],
            [0, 0, 0, 1]
        ])
        ry = numpy.array([
            [numpy.cos(phi), 0, -numpy.sin(phi), 0],
            [0, 1, 0, 0],
            [numpy.sin(phi), 0, numpy.cos(phi), 0],
            [0, 0, 0, 1]
        ])
        rz = numpy.array([
            [numpy.cos(psi), numpy.sin(psi), 0, 0],
            [-numpy.sin(psi), numpy.cos(psi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        for atom in self.atoms:
            current_centroid = self.centroid
            self.move_to([0, 0, 0])

            transpose = numpy.append(atom.xyz[numpy.newaxis, :].T, 1)
            atom.xyz = (rx.dot(ry).dot(rz).dot(transpose)).T[:3]

            self.move_to(current_centroid)

    def find_bonds(self, bond_radius = DEFAULT_BOND_RADIUS):
        """ Find bonds between molecules

        :param float bond_radius: (optional) The distance, in angstroms,
            between two atoms under which a bond will be assumed
        :return: A two-dimensional array of atoms where each pair of atoms
            represents a bond
        :rtype: list
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

        :param str filename: The path that the XYZ data should be written to
        :param str comment: (optional) A comment to include in the XYZ file
        """

        with open(filename, "w") as f:
            f.write("%d\n%s\n" % (len(self.atoms), comment))
            for atom in self.atoms:
                f.write("%d\t%f\t%f\t%f\n" % (
                    atom.element, atom.xyz[0], atom.xyz[1], atom.xyz[2]
                ))

    def write_lammps(self, filename, *args, **kwargs):
        """ Syntactic sugar: see :meth:`nme.nme.write_lammps` """

        write_lammps(self, filename, *args, **kwargs)

    def bbox_intersects(self, molecule, padding = 0):
        """ Test whether the bounding box of the current molecule intersects
        the bounding box of another molecule

        :param Molecule molecule: The other molecule
        :param float padding: (optional) A number to pad the bounding boxes by
            to increase their size
        :return: True if the bounding box of this molecule intersects the
            bounding box of another molecule; False otherwise.
        :rtype: bool
        """

        bbox_self = self.bbox
        bbox_other = molecule.bbox

        if (padding != 0):
            padding_np = numpy.array([padding] * 3)
            bbox_self[0] -= padding_np
            bbox_self[1] += padding_np
            bbox_other[0] -= padding_np
            bbox_other[1] += padding_np

        return numpy.all(
            (bbox_self[1] >= bbox_other[0])
            & (bbox_self[0] <= bbox_other[1])
        )

    @property
    def bbox(self):
        coords = numpy.array([atom.xyz for atom in self.atoms])
        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        return numpy.array([
            [min(x), min(y), min(z)],
            [max(x), max(y), max(z)]
        ])

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
        """ Initialize a :class:`Workspace`. A Workspace is a container for
        :class:`Atom`s and :class:`Molecule`s. The primary purpose of a
        Workspace is to facilitate writing multiple atoms and molecules to a
        single output file. """

        self.molecules = []
        self.attributes = {}

    def append(self, object_):
        """ Append atoms or molecules to the workspace

        :param object_: An `Atom` or `Molecule`
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

        :param str filename: The path that the XYZ data should be written to
        :param str comment: (optional) A comment to include in the XYZ file
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

    def write_lammps(self, filename, *args, **kwargs):
        """ Syntactic sugar: see :meth:`nme.nme.write_lammps`"""

        write_lammps(self.molecules, filename, *args, **kwargs)

    @property
    def bbox(self):
        all_atoms = []
        for molecule in self.molecules:
            all_atoms += molecule.atoms
        all_x = [atom.xyz[0] for atom in all_atoms]
        all_y = [atom.xyz[1] for atom in all_atoms]
        all_z = [atom.xyz[2] for atom in all_atoms]
        return [
            numpy.array([min(all_x), min(all_y), min(all_z)]),
            numpy.array([max(all_x), max(all_y), max(all_z)])
        ]

    # Dict-like storage of values
    def __setitem__(self, key, value):
        self.attributes[key] = value
    def __getitem__(self, key):
        return self.attributes[key]
    def __delitem__(self, key):
        del self.attributes[key]

    # Iteration procedures
    def __iter__(self):
        self.molecule_iter_index = -1
        return self
    def __next__(self):
        self.molecule_iter_index += 1
        if (self.molecule_iter_index == len(self.molecules)):
            raise StopIteration
        else:
            return self.molecules[self.molecule_iter_index]

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

    :param str filepath: The path of the xyz file
    :return: A :class:`Molecule` object containing the atoms in the .xyz file
    :rtype: Molecule
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

def write_lammps(molecules, filename, bbox = None,
                 boundary_margin = DEFAULT_BOUNDARY_MARGIN):
    """ Write the given molecules to a LAMMPS data file

    :param molecules: A molecule or array of molecules to be written
    :param filename: The path that the LAMMPS data should be written to
    :param bbox: (optional) The bounding box of the simulation area, specified
        as a pair of cartesian coordinates corresponding to the minimum
        coordinates and the maximum coordinates (e.g. [ [0, 0, 0], [1, 1, 1] ])
    :param boundary_margin: (optional) If a bounding box is not supplied,
        compute a bounding box automatically by padding the area taken up by
        the molecules by this many atoms
    """

    if (type(molecules) != list):
        molecules = [molecules]

    all_atoms = []
    for molecule in molecules:
        all_atoms += molecule.atoms

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
            % len(all_atoms)
        )

        # Summary - types
        f.write(
            "%d atom types\n0 bond types\n0 angle types\n0 dihedral types\n0 improper types\n\n"
            % len(unique_elements)
        )

        # Compute bbox if necessary
        if (bbox is None):
            all_x = [atom.xyz[0] for atom in all_atoms]
            all_y = [atom.xyz[1] for atom in all_atoms]
            all_z = [atom.xyz[2] for atom in all_atoms]
            bbox = [
                [
                    min(all_x) - boundary_margin,
                    min(all_y) - boundary_margin,
                    min(all_z) - boundary_margin
                ],
                [
                    max(all_x) + boundary_margin,
                    max(all_y) + boundary_margin,
                    max(all_z) + boundary_margin
                ]
            ]
        f.write("%0.6f %0.6f xlo xhi\n" % (bbox[0][0], bbox[1][0]))
        f.write("%0.6f %0.6f ylo yhi\n" % (bbox[0][1], bbox[1][1]))
        f.write("%0.6f %0.6f zlo zhi\n\n" % (bbox[0][2], bbox[1][2]))

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
