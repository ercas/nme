.. NICE Lab Molecule Editor documentation master file, created by
   sphinx-quickstart on Tue Mar 27 12:13:57 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NICE Lab Molecule Editor
========================

``nme`` is a Python library that provides a simple, programmatic interface to
XYZ files. ``nme`` can also export LAMMPS input data files.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    api

Quick Start
-----------

.. code-block:: python

    import nme

Basic usage of ``nme`` involves loading and manipulating molecules from XYZ
files. A single molecule can be loaded with the function
:meth:`read_xyz <nme.nme.read_xyz>`.

``nme`` provides three main classes: :class:`Atom <nme.nme.Atom>`,
:class:`Molecule <nme.nme.Molecule>`, and :class:`Workspace
<nme.nme.Workspace>`.  An ``Atom`` is the basic unit of ``nme`` and can be
initialized as such:

.. code-block:: python

    # The first argument is the atomic number; the second argument is a list of
    # the atom's X, Y, and Z coordinates.
    carbon = nme.Atom(6, [0, 0, 1])

    # All Atoms, Molecules, and Workspaces can hold attributes which can be
    # set and accessed via indexing. These attributes are ignored when saving
    # to disk.
    carbon["used"] = False
    print(carbon["used"])

:class:`Molecule <nme.nme.Molecule>` objects consist of ``Atom`` objects and
can either be constructed by hand or loaded from an XYZ file. You can append
atoms or other molecules to ``Molecule`` objects - this can be done by using
either the :meth:`Molecule.append <nme.nme.Molecule.append>` function or the
``+=`` operator. Additionlly, ``Molecule`` objects can be saved to XYZ files
using :meth:`Molecule.write_xyz <nme.nme.Molecule.write_xyz>`.

.. code-block:: python

    glucose = nme.read_xyz("./samples/glucose.xyz")
    glucose.move_to([0, 0, 0])
    glucose += carbon
    glucose.write_xyz("./glucose_with_extra_carbon.xyz")

.. image:: ./static/glucose_with_extra_carbon.png

.. code-block:: python

    (glucose + nme.Atom(6, [0, 0, -1])).write_xyz("./glucose_plus_2_carbon.xyz")

.. image:: ./static/glucose_plus_2_carbon.png


:class:`Workspace <nme.nme.Workspace>` objects consist of ``Molecule`` and
``Atom`` objects. The primary purpose of a ``Workspace`` is to allow writing of
multiple objects to a single output file.

``Workspace`` objects have a :meth:`Workspace.write_xyz
<nme.nme.Workspace.write_xyz>` function that writes the entire workspace to a
single XYZ file as a single molecule. Additionally the :meth:`write_lammps
<nme.nme.write_lammps` method writes the entire workspace to a LAMMPS input
data file.

.. code-block:: python

    workspace = nme.Workspace()

    glucose = nme.read_xyz("./samples/glucose.xyz")
    glucose.move_to([0, 0, 0])
    workspace += glucose

    # Create an independent copy of the glucose molecule so that we can edit it
    # without affecting the state of the other glucose molecule
    glucose_2 = glucose.copy()
    glucose_2.move_to([0, 0, 1])
    workspace += glucose_2

    # Because we still have an independent reference to the original glucose
    # molecule, we can still access it directly
    glucose.move_to([0, 0, -2])

    workspace.write_xyz("2_glucose.xyz")
    workspace.write_lammps("2_glucose.dat")

.. image:: ./static/2_glucose.png
