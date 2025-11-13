import math
from problem3_main import (
    compute_inertia_matrix,
    compute_coriolis_centrifugal,
    compute_gravity_terms,
    m1, m2, m3, l1, l2, l3, rho, r1, r2, r3, g as gravity
)

def print_matrix(M, name="Matrix", indent=2):
    spaces = " " * indent
    print(f"\n{name}:")
    if isinstance(M[0], list):
        for row in M:
            print(f"{spaces}[{row[0]:12.6e}, {row[1]:12.6e}, {row[2]:12.6e}]")
    else:
        print(f"{spaces}[{M[0]:12.6e}, {M[1]:12.6e}, {M[2]:12.6e}]")

def matrix_determinant_3x3(M):
    return (M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[2][1]) -
            M[0][1] * (M[1][0]*M[2][2] - M[1][2]*M[2][0]) +
            M[0][2] * (M[1][0]*M[2][1] - M[1][1]*M[2][0]))

def compute_and_print_configuration(config_num, q, dq):
    print("\n" + "="*70)
    print(f"Configuration {config_num}:")
    print(f"  θ = [{q[0]/math.pi:.4f}π, {q[1]/math.pi:.4f}π, {q[2]/math.pi:.4f}π] rad")
    print(f"    = [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}] rad")
    print(f"  θ̇ = [{dq[0]/math.pi:.4f}π, {dq[1]/math.pi:.4f}π, {dq[2]/math.pi:.4f}π] rad/s")
    print(f"    = [{dq[0]:.6f}, {dq[1]:.6f}, {dq[2]:.6f}] rad/s")
    print("="*70)
    
    D = compute_inertia_matrix(q)
    C_dq = compute_coriolis_centrifugal(q, dq)
    g_vec = compute_gravity_terms(q)
    
    print_matrix(D, "Inertia Matrix D(q)")
    det_D = matrix_determinant_3x3(D)
    print(f"  det(D) = {det_D:.6e}")
    
    print_matrix(C_dq, "Coriolis/Centrifugal C(q,q̇)·q̇")
    print_matrix(g_vec, "Gravity g(q)")
    
    return D, C_dq, g_vec

if __name__ == "__main__":
    print("="*70)
    print(" "*15 + "Problem 3")
    print("="*70)
    
    print("\nRobot Parameters:")
    print(f"  Density ρ = {rho} kg/m³")
    print(f"  Radius r₁ = r₂ = r₃ = {r1} m")
    print(f"  Length l₁ = l₂ = l₃ = {l1} m")
    print(f"  Mass m₁ = m₂ = m₃ = {m1:.6f} kg")
    print(f"  Gravity g = {gravity} m/s²")
    
    # Configuration 1
    q1 = [math.pi/3, math.pi/2, math.pi/3]
    dq1 = [0.0, 0.0, 0.0]
    D1, C1, g1 = compute_and_print_configuration(1, q1, dq1)
    
    # Configuration 2
    q2 = [math.pi/3, math.pi/2, math.pi/3]
    dq2 = [math.pi/4, math.pi/4, math.pi/2]
    D2, C2, g2 = compute_and_print_configuration(2, q2, dq2)
    
    # Configuration 3
    q3 = [0.0, math.pi/2, 0.0]
    dq3 = [math.pi/4, math.pi/4, math.pi/2]
    D3, C3, g3 = compute_and_print_configuration(3, q3, dq3)
    
    print("Calculation completed!")

