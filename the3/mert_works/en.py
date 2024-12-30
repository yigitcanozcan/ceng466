
from math import pi
from sympy import symbols, Eq, solve, log, exp

# d_0 = 15e-3
# l_0 = 120e-3
# F = 35e3

# d_max = 1.2e-5

# A = (pi*d_0**2)/4

# l = [
#     [70e9, 250e6, 0.33],
#     [105e9, 850e6, 0.36],
#     [205e9, 550e6, 0.27],
#     [45e9, 170e6, 0.35]
# ]

# stress = F/A
# print(f"stress: {stress:.3e}")

# for i in l:
#     E, Y, P = i

#     if Y < stress:
#         print(f"it is not possible: Y < stress: {Y:.3e} < {stress:.3e}")

#     e_axial = stress/E

#     delta_d = -(P*e_axial*d_0)

#     delta_d = abs(delta_d)

#     print(f"delta_d = {delta_d:.3e}")
#     print(f"d_max = {d_max:.3e}")
#     print("is it possible = ", delta_d < d_max)


# data = [
#     ["Aluminum",    35e6,   90e6, 40],
#     ["Copper",      69e6,  200e6, 45],
#     ["Brass(70Cu–30Zn)", 75e6, 300e6, 68],
#     ["Nickel",     138e6,  480e6, 40],
#     ["Steel (1020)", 180e6, 380e6, 25],
#     ["Titanium",    450e6,  520e6, 25],
# ]


# d_0 = 15e-3
# l_0 = 120e-3
# F = 15e3

# A = (pi*d_0**2)/4

# stress = F/A

# print(f"Stress: {stress:.3e}")

# for q in data:

#     matter, Y, T, Ductility = q

#     is_plastic_deformation = Y < stress
    
#     print("Material: ", matter, f"Yield: {Y:.3e}")
#     if is_plastic_deformation:
#         print("Plastic deformation occurs")
#     else:
#         print("Plastic deformation does not occur")
    

#     print("")

# data = [
#     ["Steel alloy",     830e6,   207000e6],
#     ["Brass alloy",     380e6,   97000e6],
#     ["Aluminum alloy",  275e6,   69000e6],
#     ["Titanium alloy",  690e6,   107000e6]
# ]

# for q in data:

#     matter, Y, E = q

#     U_r = Y**2/(2*E)

#     print("Material: ", matter, f"U_r (modulus of resilience): {U_r:.3e}")


# q_eng1 = 315e6
# e_eng1 = 0.105

# q_eng2 = 340e6
# e_eng2 = 0.220

# e_engtarget = 0.28

# q_true1 = q_eng1*(1+e_eng1)
# q_true2 = q_eng2*(1+e_eng2)

# print(f"q_true1: {q_true1:.3e}")
# print(f"q_true2: {q_true2:.3e}")

# e_true1 = log(1+e_eng1)
# e_true2 = log(1+e_eng2)
# e_truetarget = log(1+e_engtarget)

# print(f"e_true1: {e_true1:.3e}")
# print(f"e_true2: {e_true2:.3e}")
# print(f"e_target: {e_truetarget:.3e}")

# K, n = symbols('K n', real=True)
# eq1 = Eq(log(K)+n*log(e_true1), log(q_true1))
# eq2 = Eq(log(K)+n*log(e_true2), log(q_true2))

# result = solve((eq1, eq2), (K, n))[0]

# K, n = result

# q_truetarget = exp(log(K)+n*log(e_truetarget))
# print(f"q_target: {q_truetarget:.3e}")

# q_engtarget = q_truetarget/(1+e_engtarget)
# print(f"q_engtarget: {q_engtarget:.3e}")

q_y1 = 230e6
d1 = 1e-2
d1_ = d1**(-1/2)

q_y2 = 275e6
d2 = 6e-3
d2_ = d2**(-1/2)

print("d1_: ", d1_, "d2_: ", d2_)

q_0, k = symbols('q_0 k', real=True)
eq1 = Eq(q_0 + k*d1_, q_y1)
eq2 = Eq(q_0 + k*d2_, q_y2)

result = solve((eq1, eq2), (q_0, k))

q_0, k = result[q_0], result[k]

print(f"q_0: {q_0:.3e}")
print(f"k: {k:.3e}")

q_ytarget = 310e6

d_target = ((q_ytarget - q_0)/k)**(-2)

print(f"d_target: {d_target:.3e}")





