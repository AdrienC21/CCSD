#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""create_tanimot_smiles.py: Code to create the molecule used in GDSS for Tanimoto on QM9.
"""
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import RWMol

if __name__ == "__main__":
    # Create molecule
    C = rdkit.Chem.rdchem.Atom("C")
    O = rdkit.Chem.rdchem.Atom("O")
    N = rdkit.Chem.rdchem.Atom("N")

    mol = RWMol()
    mol.AddAtom(C)
    mol.AddAtom(C)
    mol.AddAtom(C)
    mol.AddAtom(C)
    mol.AddAtom(C)
    mol.AddBond(0, 1, rdkit.Chem.rdchem.BondType.DOUBLE)
    mol.AddBond(1, 2, rdkit.Chem.rdchem.BondType.SINGLE)
    mol.AddBond(2, 3, rdkit.Chem.rdchem.BondType.SINGLE)
    mol.AddBond(3, 4, rdkit.Chem.rdchem.BondType.SINGLE)
    mol.AddBond(4, 0, rdkit.Chem.rdchem.BondType.SINGLE)
    mol.AddAtom(N)
    mol.AddAtom(C)
    mol.AddBond(3, 5, rdkit.Chem.rdchem.BondType.SINGLE)
    mol.AddBond(3, 6, rdkit.Chem.rdchem.BondType.SINGLE)
    mol.AddBond(5, 6, rdkit.Chem.rdchem.BondType.SINGLE)
    mol.AddAtom(C)
    mol.AddAtom(O)
    mol.AddBond(6, 7, rdkit.Chem.rdchem.BondType.SINGLE)
    mol.AddBond(7, 8, rdkit.Chem.rdchem.BondType.DOUBLE)

    # Sanitize molecule
    rdkit.Chem.SanitizeMol(mol)

    # Convert to SMILES
    smiles = Chem.MolToSmiles(mol)
    print(smiles)

    # Plot molecule
    mol_img = Draw.MolToImage(mol, size=(300, 300))
    plt.imshow(mol_img)
    plt.suptitle(f"SMILES: {smiles}")
    plt.show()
