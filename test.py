from src.basis import loader
from src.geometry import cartesian
from src.geometry import zmatrix
from src.integrals.python import base

# create a water molecule using cartesian coordinates
water = cartesian.Cartesian()
water.geometry = (
    ["H", "O", "H"],  # atoms
    [
        [0.000000, 0.763239, -0.477047],
        [0.000000, 0.000000, 0.119262],
        [0.000000, -0.763239, -0.477047]
    ],  # coordinates
    0, 1  # charge and multiplicity
)
print(water.atoms)

# create a hydrogen peroxide moleule using z-matrix
peroxide = zmatrix.ZMatrix()
peroxide.geometry = """
4 0 1
H
O 1 1.0
O 2 1.5 1 109.71
H 3 1.0 2 109.71 1 0.000
"""
print(peroxide)

# create a methane moleule using z-matrix
# test variable substitution
r2 = 1.0923
r3 = 1.5365
a3 = 110.40
r4 = 1.0923
a4 = 108.53
d4 = 238.88
r5 = 1.0923
a5 = 108.53
d5 = 121.12
r6 = 1.0923
a6 = 110.40
d6 = 300.00
r7 = 1.0923
a7 = 110.40
d7 = 60.00
r8 = 1.0923
a8 = 110.40
d8 = 180.00

ethane = zmatrix.ZMatrix()
ethane.geometry = f"""
8 0 1
H
C  1  {r2}
C  2  {r3}  1  {a3}
H  2  {r4}  1  {a4}  3  {d4}
H  2  {r5}  1  {a5}  3  {d5}
H  3  {r6}  2  {a6}  1  {d6}
H  3  {r7}  2  {a7}  1  {d7}
H  3  {r8}  2  {a8}  1  {d8}
"""
print(ethane)

# try loading a basis set
basis = loader.BasisSetLoader(basis_folder="/Users/hemanthharidas/Desktop/codes/hartee-fock/basis-sets")

# try loading D shell
kcl = cartesian.Cartesian()
kcl.geometry = (
    ["K", "Cl"],  # atoms
    [
        [-1.713138, 0.994968, 0.000000],
        [1.002682, 1.069375, 0.000000],
    ],  # coordinates
    0, 1  # charge and multiplicity
)
print(kcl)

test = basis.load(basis_name="6-31g", basis_type="gaussian94", molecule=water.geometry)

# print the basis set information
for shell in test:
    # print(shell)
    print(f"Shell Type      : {shell.angular_momentum:10s}\n",
          f"Exponents       : {shell.exponents}\n",
          f"Coefficients    : {shell.coefficients}\n"
          f"Normalizations  : {shell.normalized_coeffs.values()}\n"
          f"Center          : {shell.atom} at {shell.location}"
          )

# write the basis sets as an xml file
basis._write_xml()

# now test gaussian product theorem
gaussiancenters, gaussianintegrals, gaussianexponents = base.gaussiantheorem(
    centerA=test[0].location, exponentA=test[0].exponents,
    centerB=test[2].location, exponentB=test[2].exponents
)
