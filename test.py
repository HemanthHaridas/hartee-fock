from src.basis import loader
from src.geometry import cartesian

# create a water molecule
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
print(water)


basis = loader.BasisSetLoader(basis_folder="/Users/hemanthharidas/Desktop/codes/hartee-fock/basis-sets")
test = basis.load(basis_name="def2-tzvp", basis_type="gaussian94", atoms=["H", "O", "H"])
print(test)
