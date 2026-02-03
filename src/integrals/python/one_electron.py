import numpy
from src.integrals.python import base_integrals


def overlap_md_3D(centerA: numpy.ndarray, exponentA: numpy.ndarray, shellA: numpy.ndarray, centerB: numpy.ndarray, exponentB: numpy.ndarray, shellB: numpy.ndarray) -> numpy.ndarray:
    # get product gaussians
    gaussian_centers, gaussian_integrals, gaussian_exponents = base_integrals.gaussian_product_theorem(centerA, exponentA, centerB, exponentB)

    # initialize recursion arrays for x, y, z
    maxLx = shellA[0] + shellB[0]
    maxLy = shellA[1] + shellB[1]
    maxLz = shellA[2] + shellB[2]

    # set up recursion arrays
    Ex = numpy.zeros((exponentA.shape[0], exponentB.shape[0], maxLx + 1))
    Ey = numpy.zeros((exponentA.shape[0], exponentB.shape[0], maxLy + 1))
    Ez = numpy.zeros((exponentA.shape[0], exponentB.shape[0], maxLz + 1))

    # base cases
    exponentA = numpy.ravel(exponentA)
    exponentB = numpy.ravel(exponentB)
    comb_exp = exponentA[:, None] + exponentB[None, :]
    Ex[..., 0] = gaussian_integrals * (numpy.pi / comb_exp)**1.5
    Ey[..., 0] = gaussian_integrals * (numpy.pi / comb_exp)**1.5
    Ez[..., 0] = gaussian_integrals * (numpy.pi / comb_exp)**1.5

    # recursion in x
    Px = gaussian_centers[..., 0]
    for lx in range(1, maxLx + 1):
        Ex[..., lx] = (Px - centerA[0, 0]) * Ex[..., lx - 1]
        if lx > 1:
            Ex[..., lx] += (lx - 1) / (2 * comb_exp) * Ex[..., lx - 2]

    # recursion in y
    Py = gaussian_centers[..., 1]
    for ly in range(1, maxLy + 1):
        Ey[..., ly] = (Py - centerA[1, 0]) * Ey[..., ly - 1]
        if ly > 1:
            Ey[..., ly] += (ly - 1) / (2 * comb_exp) * Ey[..., ly - 2]

    # recursion in z
    Pz = gaussian_centers[..., 0]
    for lz in range(1, maxLz + 1):
        Ez[..., lz] = (Pz - centerA[2, 0]) * Ez[..., lz - 1]
        if lz > 1:
            Ez[..., lz] += (lz - 1) / (2 * comb_exp) * Ez[..., lz - 2]

    # final overlap integral for given shells
    overlap = Ex[..., shellA[0] + shellB[0]] * Ey[..., shellA[1] + shellB[1]] * Ez[..., shellA[2] + shellB[2]]

    return overlap
