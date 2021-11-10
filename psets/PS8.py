import sympy as sp


def simp_dot(a, b):  # Taken from solutions
    expr = a.dot(b)
    return expr.simplify().trigsimp().factor().simplify()


if __name__ == "__main__":
    u, v = sp.symbols("u v", real=True)
    x = sp.Matrix(
        [
            u - sp.Pow(u, 3) / 3 + u * sp.Pow(v, 2),
            v - sp.Pow(v, 3) / 3 + v * sp.Pow(u, 2),
            sp.Pow(u, 2) - sp.Pow(v, 2),
        ]
    )

    x_u = sp.diff(x, u)
    x_uu = sp.diff(x_u, u)
    x_v = sp.diff(x, v)
    x_vv = sp.diff(x_v, v)
    x_uv = sp.diff(x_v, u)

    N = x_u.cross(x_v).normalized()
    N.simplify()

    E = simp_dot(x_u, x_u)
    F = simp_dot(x_u, x_v)
    G = simp_dot(x_v, x_v)
    e = simp_dot(N, x_uu)
    f = simp_dot(N, x_uv)
    g = simp_dot(N, x_vv)

    a11 = (f * F - e * G) / (E * G - F ** 2)
    a12 = (g * F - f * G) / (E * G - F ** 2)
    a21 = (e * F - f * E) / (E * G - F ** 2)
    a22 = (f * F - g * E) / (E * G - F ** 2)

    dN = sp.Matrix([[a11, a12], [a21, a22]])

    eigs = dN.eigenvals()

    print("Parts A, B, and C")
    print("---------------------")
    print(f"a) The values of E, F, and G are {E}, {F}, and {G} respectively.")
    print(f"b) The values of e, f, and g are {e}, {f}, and {g} respectively")
    print(
        f"c) The eigenvalues of dN (the principal curvatures) are {list(eigs.keys())}."
    )
