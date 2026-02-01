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
